use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, OnceLock};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};
use uuid::Uuid;

use crate::db::repo::{
    active_job_for, characters_of, clips_of, get_asset, get_character, get_clip,
    get_project, get_secret, get_settings, insert_asset, insert_job, patch_clip,
    set_character_sheet, touch_project,
    CharacterRow, NewAsset,
};
use crate::db::DbPool;
use crate::jobs::{JobManager, QueueName};
use crate::paths::asset_abs_path;
use crate::services::audio::{probe_duration_ms, probe_size};
use crate::services::{elevenlabs, gemini, styles};

use crate::speech::align::{estimate_word_timings, strip_voice_tags};


/// The supervised Flow2API backend, when the app has one. Generation tasks run
/// outside a command, so they reach the manager through the app handle.
fn flow2api_manager(app: &AppHandle) -> Option<Arc<crate::flow2api_runtime::Flow2ApiManager>> {
    use tauri::Manager;
    app.try_state::<Arc<crate::flow2api_runtime::Flow2ApiManager>>().map(|s| s.inner().clone())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfInput {
    pub field: Option<String>,
    pub mode: String, // "none" | "first-frame" | "reference"
    pub max: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModel {
    pub id: String,
    pub name: String,
    pub family: String,
    pub kind: String, // "image" | "video"
    pub input: HfInput,
    #[serde(default)]
    pub required: Vec<String>,
    #[serde(default)]
    pub props: HashMap<String, Value>,
    pub durations: Option<Value>,
    #[serde(default)]
    pub tags: Vec<String>,
}

pub fn get_image_model(id: Option<&str>) -> HfModel {
    if let Some(target) = id {
        if target.starts_with("flow:") {
            let canonical = crate::services::flow2api::canonical_flow_id(target);
            if let Some(m) = crate::services::flow2api::flow_image_models().into_iter().find(|m| m.id == canonical) {
                return m;
            }
        }
    }
    crate::services::flow2api::flow_image_models().into_iter().next().expect("No flow image models configured")
}

pub fn get_video_model(id: Option<&str>) -> HfModel {
    if let Some(target) = id {
        if target.starts_with("flow:") {
            let canonical = crate::services::flow2api::canonical_flow_id(target);
            if let Some(m) = crate::services::flow2api::flow_video_models().into_iter().find(|m| m.id == canonical) {
                return m;
            }
        }
    }
    crate::services::flow2api::flow_video_models().into_iter().next().expect("No flow video models configured")
}

pub fn max_refs(m: &HfModel) -> usize {
    if m.kind == "image" && m.input.mode == "reference" {
        m.input.max
    } else {
        0
    }
}

pub fn prompt_limit(m: &HfModel) -> usize {
    m.props
        .get("prompt")
        .and_then(|p| p.get("maxLength"))
        .and_then(|l| l.as_u64())
        .map(|l| l as usize)
        .unwrap_or(if m.kind == "video" { 2500 } else { 4000 })
}


fn clip_str(text: &str, limit: usize) -> &str {
    if text.len() <= limit {
        return text;
    }
    // Snap down to a char boundary: slicing mid-codepoint panics on multibyte text.
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let cut = &text[..end];
    if let Some(space) = cut.rfind(' ') {
        if space > (limit * 8 / 10) {
            return cut[..space].trim_end();
        }
    }
    cut.trim_end()
}



fn split_avoid(visual: &str) -> (String, String) {
    let mut lines: Vec<&str> = visual.lines().collect();
    let mut avoid = String::new();
    let mut avoid_idx = None;

    for (i, l) in lines.iter().enumerate() {
        let trimmed = l.trim_start();
        if trimmed.to_lowercase().starts_with("avoid:") {
            avoid = trimmed[6..].trim().trim_end_matches('.').to_string();
            avoid_idx = Some(i);
            break;
        }
    }

    if let Some(idx) = avoid_idx {
        lines.remove(idx);
    }

    (lines.join("\n").trim().to_string(), avoid)
}

fn clip_prompt(
    aspect: &str,
    style: &str,
    visual: &str,
    chars: &[CharacterRow],
    ref_names: &[String],
    limit: usize,
) -> String {
    let (scene, avoid) = split_avoid(visual);
    let wardrobe = scene.to_lowercase().contains("wardrobe:");

    // Style first so a truncated prompt never loses it.
    let mut parts = vec![
        format!("Art style (must match exactly): {style}"),
        format!("Scene:\n{scene}"),
    ];

    if !chars.is_empty() {
        let mut char_text = format!(
            "Recurring characters in this scene. Keep each one's body, face, hair and skin exactly as described{}.",
            if !ref_names.is_empty() { " and as on their reference sheet" } else { "" }
        );
        if wardrobe {
            char_text.push_str(" Their clothing follows the Wardrobe line of the scene: the default outfit only where it says so, otherwise the custom outfit.");
        } else {
            char_text.push_str(" They wear their default outfit.");
        }
        char_text.push_str("\n");
        char_text.push_str(&chars.iter().map(|c| format!("{}: {}", c.name, c.description)).collect::<Vec<_>>().join("\n"));
        parts.push(char_text);
    }

    if !ref_names.is_empty() {
        let mut ref_text = format!("Reference images are character sheets, in this order: {}. Use them for each character's identity, proportions and colors, but render them in the requested art style even if a reference sheet uses a different style. ", ref_names.join(", "));
        if wardrobe {
            ref_text.push_str("Copy the costume from a sheet only for characters whose Wardrobe line says \"default outfit\". ");
        } else {
            ref_text.push_str("Match the costume to the sheet. ");
        }
        ref_text.push_str("Do not copy the sheet layout, poses or plain background.");
        parts.push(ref_text);
    }

    if !avoid.is_empty() {
        parts.push(format!("Must not appear in the image: {avoid}."));
    }
    parts.push(format!("Frame format {aspect}, full-bleed composition."));

    let full = parts.join("\n\n");
    if full.len() <= limit {
        full
    } else {
        clip_str(&full, limit).to_string()
    }
}

fn sheet_prompt(c: &CharacterRow, style: &str, hint: &str) -> String {
    format!(
        "Character reference sheet for \"{}\": {}. \
Draw the character as {}. \
Show the SAME character three times side by side: front view, three-quarter view and side view, \
full body, neutral standing pose, evenly spaced, on a simple uncluttered background that does not override the selected art style. \
Art style of every view: {style}. This style determines the character's visual form and rendering; keep the character's identity and costume details. \
Consistent proportions, costume and colors in all three views.",
        c.name, c.description, hint
    )
}


async fn wait_for_character_sheets(
    db: &DbPool,
    char_ids: &[String],
    token: &tokio_util::sync::CancellationToken,
) {
    let start = std::time::Instant::now();
    while start.elapsed() < std::time::Duration::from_secs(15 * 60) {
        if token.is_cancelled() {
            return;
        }
        let pending = {
            let conn = db.lock().unwrap();
            char_ids.iter().any(|cid| active_job_for(&conn, None, Some(cid), "sheet").is_some())
        };
        if !pending {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
    }
}


// Shared by Flow clip images and sheets, including manual generation.
// Acquire after waiting for references and hold until the asset is saved.
fn image_gate() -> &'static tokio::sync::Semaphore {
    static GATE: OnceLock<tokio::sync::Semaphore> = OnceLock::new();
    GATE.get_or_init(|| tokio::sync::Semaphore::new(1))
}

async fn acquire_image_turn<'a>(
    gate: &'a tokio::sync::Semaphore,
    token: &tokio_util::sync::CancellationToken,
) -> Result<tokio::sync::SemaphorePermit<'a>, String> {
    tokio::select! {
        biased;
        _ = token.cancelled() => Err("Pembuatan gambar dibatalkan".into()),
        permit = gate.acquire() => permit.map_err(|_| "Antrean gambar ditutup".into()),
    }
}

fn save_asset_file(project_id: &str, folder: &str, ext: &str, bytes: &[u8]) -> Result<String, String> {
    let rel = format!("{}/{}{}", folder, Uuid::new_v4(), ext);
    let abs = asset_abs_path(project_id, &rel);
    if let Some(parent) = abs.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&abs, bytes).map_err(|e| e.to_string())?;
    Ok(rel)
}


pub fn generate_clip_image(
    db: DbPool,
    job_mgr: Arc<JobManager>,
    app: AppHandle,
    clip_id: String,
) -> Result<Value, String> {
    let (project_id, clip) = {
        let conn = db.lock().unwrap();
        if let Some(j) = active_job_for(&conn, Some(&clip_id), None, "image") {
            return Ok(j);
        }
        let c = get_clip(&conn, &clip_id).ok_or("Klip tidak ditemukan")?;
        let p = get_project(&conn, &c.project_id).ok_or("Proyek tidak ditemukan")?;
        let _m = get_image_model(p.image_model.as_deref());
        (c.project_id.clone(), c)
    };

    let provider = "flow2api";
    let job = {
        let conn = db.lock().unwrap();
        insert_job(&conn, &project_id, "image", provider, json!({}), Some(&clip_id), None)?
    };

    let j_id = job.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string();
    let _ = app.emit("event:job", &job);

    let db_clone = db.clone();
    let app_clone = app.clone();
    let c_id = clip_id.clone();
    let p_id = project_id.clone();

    job_mgr.spawn_task(&j_id, QueueName::Sheet, move |ctx| async move {
        ctx.progress(0.04, Some("Menyiapkan prompt"));

        let (mode, flow_base, flow_key, project, model, chars) = {
            let conn = db_clone.lock().unwrap();
            let s = get_settings(&conn);
            let mode = s.get("flow2apiMode").and_then(|x| x.as_str()).unwrap_or("bundled").to_string();
            let fb = s.get("flow2apiBaseUrl").and_then(|x| x.as_str()).unwrap_or("http://localhost:8000").to_string();
            let fk = get_secret(&conn, "flow2api");
            let p = get_project(&conn, &p_id).ok_or("Proyek tidak ditemukan")?;
            let m = get_image_model(p.image_model.as_deref());
            let all_chars = characters_of(&conn, &p_id);
            let needed: Vec<CharacterRow> = all_chars
                .into_iter()
                .filter(|c| clip.character_ids.contains(&c.id))
                .collect();
            (mode, fb, fk, p, m, needed)
        };
        let (flow_base, flow_key) = if crate::flow2api_runtime::is_bundled(&mode) {
            match flow2api_manager(&app_clone) {
                Some(manager) => crate::flow2api_runtime::resolve_target("bundled", &flow_base, None, Some(&manager)).await?,
                None => return Err("Flow2API internal belum siap. Buka Pengaturan lalu coba lagi.".into()),
            }
        } else {
            (flow_base, flow_key)
        };

        let limit = max_refs(&model);
        let mut ref_images = Vec::new();
        let mut ref_names = Vec::new();

        if limit > 0 && !chars.is_empty() {
            let c_ids: Vec<String> = chars.iter().map(|c| c.id.clone()).collect();
            wait_for_character_sheets(&db_clone, &c_ids, &ctx.token).await;

            for c in &chars {
                if ref_images.len() >= limit {
                    break;
                }
                let sheet_path_opt = {
                    let conn = db_clone.lock().unwrap();
                    get_character(&conn, &c.id)
                        .and_then(|fresh| fresh.sheet_asset_id)
                        .and_then(|sid| get_asset(&conn, &sid))
                        .map(|a| asset_abs_path(&p_id, &a.local_path))
                };
                if let Some(path) = sheet_path_opt {
                    if let Ok(bytes) = fs::read(&path) {
                        ref_images.push(bytes);
                        ref_names.push(c.name.clone());
                    }
                }
            }
        }

        let prompt = clip_prompt(&project.aspect_ratio, styles::style_prompt(&project.style_id), &clip.visual_prompt, &chars, &ref_names, prompt_limit(&model));
        ctx.progress(0.15, Some("Menunggu antrean gambar Flow"));

        // Strictly one Flow image request at a time: parallel requests are
        // rejected with UNUSUAL_ACTIVITY. Hold the turn through the save so
        // the next image only starts once this one is complete.
        let _turn = acquire_image_turn(image_gate(), &ctx.token).await?;
        ctx.progress(0.16, Some("Mengirim ke Google Flow (Flow2API)"));

        let img_bytes = crate::services::flow2api::generate_image(
            &flow_base,
            flow_key.as_deref(),
            &model.id,
            &prompt,
            &project.aspect_ratio,
            &ref_images,
            &ctx.token,
        ).await?;

        ctx.progress(0.92, Some("Menyimpan hasil gambar"));
        let ext = if img_bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            ".jpg"
        } else if img_bytes.starts_with(b"RIFF") && img_bytes.len() > 12 && &img_bytes[8..12] == b"WEBP" {
            ".webp"
        } else {
            ".png"
        };
        let rel = save_asset_file(&p_id, "images", ext, &img_bytes)?;
        let abs = asset_abs_path(&p_id, &rel);
        let (w, h) = probe_size(&abs).await.unwrap_or((1920, 1080));

        let new_asset = NewAsset {
            project_id: p_id.clone(),
            clip_id: Some(c_id.clone()),
            character_id: None,
            kind: "image".to_string(),
            provider: "flow2api".to_string(),
            model: Some(model.id.clone()),
            prompt: Some(prompt),
            remote_id: None,
            remote_url: None,
            local_path: rel,
            duration_ms: None,
            width: Some(w as i64),
            height: Some(h as i64),
            meta: json!({}),
        };

        let asset_row = {
            let conn = db_clone.lock().unwrap();
            insert_asset(&conn, new_asset)?
        };

        let _ = app_clone.emit("event:asset", &asset_row);

        {
            let conn = db_clone.lock().unwrap();
            let _ = patch_clip(&conn, &c_id, json!({ "imageAssetId": asset_row.id }));
            let _ = app_clone.emit(
                "event:clip",
                json!({
                    "clipId": c_id,
                    "patch": { "imageAssetId": asset_row.id }
                }),
            );
            if let Some(p) = get_project(&conn, &p_id) {
                let first = clips_of(&conn, &p_id).into_iter().next();
                if p.cover_asset_id.is_none() || first.map(|c| c.id) == Some(c_id.to_string()) {
                    let _ = touch_project(&conn, &p_id, None);
                }
            }
            let _ = touch_project(&conn, &p_id, None);
        }

        return Ok(Some(asset_row.id));
    });

    Ok(job)
}
pub fn generate_character_sheet(
    db: DbPool,
    job_mgr: Arc<JobManager>,
    app: AppHandle,
    character_id: String,
) -> Result<Value, String> {
    let (project_id, character) = {
        let conn = db.lock().unwrap();
        if let Some(j) = active_job_for(&conn, None, Some(&character_id), "sheet") {
            return Ok(j);
        }
        let c = get_character(&conn, &character_id).ok_or("Karakter tidak ditemukan")?;
        let p = get_project(&conn, &c.project_id).ok_or("Proyek tidak ditemukan")?;
        let _m = get_image_model(p.image_model.as_deref());
        (c.project_id.clone(), c)
    };

    let provider = "flow2api";
    let job = {
        let conn = db.lock().unwrap();
        insert_job(&conn, &project_id, "sheet", provider, json!({}), None, Some(&character_id))?
    };

    let j_id = job.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string();
    let _ = app.emit("event:job", &job);

    let db_clone = db.clone();
    let app_clone = app.clone();
    let char_id = character_id.clone();
    let p_id = project_id.clone();

    job_mgr.spawn_task(&j_id, QueueName::Sheet, move |ctx| async move {
        ctx.progress(0.04, Some("Menyiapkan lembar karakter"));

        let (mode, flow_base, flow_key, model, style) = {
            let conn = db_clone.lock().unwrap();
            let s = get_settings(&conn);
            let mode = s.get("flow2apiMode").and_then(|x| x.as_str()).unwrap_or("bundled").to_string();
            let fb = s.get("flow2apiBaseUrl").and_then(|x| x.as_str()).unwrap_or("http://localhost:8000").to_string();
            let fk = get_secret(&conn, "flow2api");
            let p = get_project(&conn, &p_id).ok_or("Proyek tidak ditemukan")?;
            let m = get_image_model(p.image_model.as_deref());
            let style = (styles::style_prompt(&p.style_id).to_string(), styles::style_character_hint(&p.style_id).to_string());
            (mode, fb, fk, m, style)
        };
        let (flow_base, flow_key) = if crate::flow2api_runtime::is_bundled(&mode) {
            match flow2api_manager(&app_clone) {
                Some(manager) => crate::flow2api_runtime::resolve_target("bundled", &flow_base, None, Some(&manager)).await?,
                None => return Err("Flow2API internal belum siap. Buka Pengaturan lalu coba lagi.".into()),
            }
        } else {
            (flow_base, flow_key)
        };

        let prompt = sheet_prompt(&character, &style.0, &style.1);
        ctx.progress(0.15, Some("Menunggu antrean gambar Flow"));
        let _turn = acquire_image_turn(image_gate(), &ctx.token).await?;
        ctx.progress(0.16, Some("Mengirim ke Google Flow (Flow2API)"));

        let img_bytes = crate::services::flow2api::generate_image(
            &flow_base,
            flow_key.as_deref(),
            &model.id,
            &prompt,
            "16:9",
            &[],
            &ctx.token,
        ).await?;

        ctx.progress(0.92, Some("Menyimpan lembar karakter"));
        let ext = if img_bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            ".jpg"
        } else if img_bytes.starts_with(b"RIFF") && img_bytes.len() > 12 && &img_bytes[8..12] == b"WEBP" {
            ".webp"
        } else {
            ".png"
        };
        let rel = save_asset_file(&p_id, "images", ext, &img_bytes)?;
        let abs = asset_abs_path(&p_id, &rel);
        let (w, h) = probe_size(&abs).await.unwrap_or((1920, 1080));

        let new_asset = NewAsset {
            project_id: p_id.clone(),
            clip_id: None,
            character_id: Some(char_id.clone()),
            kind: "image".to_string(),
            provider: "flow2api".to_string(),
            model: Some(model.id.clone()),
            prompt: Some(prompt),
            remote_id: None,
            remote_url: None,
            local_path: rel,
            duration_ms: None,
            width: Some(w as i64),
            height: Some(h as i64),
            meta: json!({}),
        };

        let asset_row = {
            let conn = db_clone.lock().unwrap();
            insert_asset(&conn, new_asset)?
        };

        let _ = app_clone.emit("event:asset", &asset_row);

        {
            let conn = db_clone.lock().unwrap();
            let _ = set_character_sheet(&conn, &char_id, &asset_row.id);
            let _ = app_clone.emit(
                "event:character",
                json!({
                    "characterId": char_id,
                    "patch": { "sheetAssetId": asset_row.id }
                }),
            );
            let _ = touch_project(&conn, &p_id, None);
        }

        return Ok(Some(asset_row.id));
    });

    Ok(job)
}
pub fn generate_clip_video(
    db: DbPool,
    job_mgr: Arc<JobManager>,
    app: AppHandle,
    clip_id: String,
) -> Result<Value, String> {
    let (project_id, clip) = {
        let conn = db.lock().unwrap();
        if let Some(j) = active_job_for(&conn, Some(&clip_id), None, "video") {
            return Ok(j);
        }
        let c = get_clip(&conn, &clip_id).ok_or("Klip tidak ditemukan")?;
        if c.image_asset_id.is_none() {
            return Err("Buat gambar klip ini dulu sebelum membuat video AI.".into());
        }
        let p = get_project(&conn, &c.project_id).ok_or("Proyek tidak ditemukan")?;
        let _m = get_video_model(p.video_model.as_deref());
        (c.project_id.clone(), c)
    };

    let provider = "flow2api";
    let job = {
        let conn = db.lock().unwrap();
        insert_job(&conn, &project_id, "video", provider, json!({}), Some(&clip_id), None)?
    };

    let j_id = job.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string();
    let _ = app.emit("event:job", &job);

    let db_clone = db.clone();
    let app_clone = app.clone();
    let c_id = clip_id.clone();
    let p_id = project_id.clone();

    job_mgr.spawn_task(&j_id, QueueName::Sheet, move |ctx| async move {
        ctx.progress(0.05, Some("Menyiapkan gambar awal"));

        let (mode, flow_base, flow_key, project, model, image_asset) = {
            let conn = db_clone.lock().unwrap();
            let s = get_settings(&conn);
            let mode = s.get("flow2apiMode").and_then(|x| x.as_str()).unwrap_or("bundled").to_string();
            let fb = s.get("flow2apiBaseUrl").and_then(|x| x.as_str()).unwrap_or("http://localhost:8000").to_string();
            let fk = get_secret(&conn, "flow2api");
            let p = get_project(&conn, &p_id).ok_or("Proyek tidak ditemukan")?;
            let m = get_video_model(p.video_model.as_deref());
            let img = clip.image_asset_id.as_ref().and_then(|id| get_asset(&conn, id)).ok_or("Gambar klip tidak ditemukan")?;
            (mode, fb, fk, p, m, img)
        };
        let (flow_base, flow_key) = if crate::flow2api_runtime::is_bundled(&mode) {
            match flow2api_manager(&app_clone) {
                Some(manager) => crate::flow2api_runtime::resolve_target("bundled", &flow_base, None, Some(&manager)).await?,
                None => return Err("Flow2API internal belum siap. Buka Pengaturan lalu coba lagi.".into()),
            }
        } else {
            (flow_base, flow_key)
        };

        let frame_path = asset_abs_path(&p_id, &image_asset.local_path);
        let frame_bytes = fs::read(&frame_path)
            .map_err(|e| format!("Gagal membaca frame gambar {}: {e}", frame_path.display()))?;

        let (scene, _avoid) = split_avoid(&clip.visual_prompt);
        let base_prompt = if !clip.video_prompt.trim().is_empty() {
            clip.video_prompt.trim().to_string()
        } else {
            format!("{}. Subtle natural motion of the characters and environment, the camera slowly pushes in.", scene)
        };

        let sec = (clip.duration_ms as f64) / 1000.0;
        ctx.progress(0.12, Some("Mengirim ke Google Flow (Flow2API)"));

        let video_bytes = crate::services::flow2api::generate_video(
            &flow_base,
            flow_key.as_deref(),
            &model.id,
            &base_prompt,
            &frame_bytes,
            sec,
            &project.aspect_ratio,
            &ctx.token,
        ).await?;

        ctx.progress(0.92, Some("Menyimpan hasil video"));
        let ext = if video_bytes.starts_with(b"GIF8") {
            ".gif"
        } else if video_bytes.starts_with(b"\x1a\x45\xdf\xa3") {
            ".webm"
        } else {
            ".mp4"
        };
        let rel = save_asset_file(&p_id, "videos", ext, &video_bytes)?;
        let abs = asset_abs_path(&p_id, &rel);
        let (w, h) = probe_size(&abs).await.unwrap_or((1920, 1080));
        let dur_ms = probe_duration_ms(&abs).await.ok().map(|d| d as i64);

        let new_asset = NewAsset {
            project_id: p_id.clone(),
            clip_id: Some(c_id.clone()),
            character_id: None,
            kind: "video".to_string(),
            provider: "flow2api".to_string(),
            model: Some(model.id.clone()),
            prompt: Some(base_prompt),
            remote_id: None,
            remote_url: None,
            local_path: rel,
            duration_ms: dur_ms,
            width: Some(w as i64),
            height: Some(h as i64),
            meta: json!({}),
        };

        let asset_row = {
            let conn = db_clone.lock().unwrap();
            insert_asset(&conn, new_asset)?
        };

        let _ = app_clone.emit("event:asset", &asset_row);

        {
            let conn = db_clone.lock().unwrap();
            let _ = patch_clip(
                &conn,
                &c_id,
                json!({ "videoAssetId": asset_row.id, "motionType": "video" }),
            );
            let _ = app_clone.emit(
                "event:clip",
                json!({
                    "clipId": c_id,
                    "patch": { "videoAssetId": asset_row.id, "motionType": "video" }
                }),
            );
            let _ = touch_project(&conn, &p_id, None);
        }

        return Ok(Some(asset_row.id));
    });

    Ok(job)
}
pub fn generate_clip_tts(
    db: DbPool,
    job_mgr: Arc<JobManager>,
    app: AppHandle,
    clip_id: String,
) -> Result<Value, String> {
    let (project_id, clip) = {
        let conn = db.lock().unwrap();
        if let Some(j) = active_job_for(&conn, Some(&clip_id), None, "tts") {
            return Ok(j);
        }
        let c = get_clip(&conn, &clip_id).ok_or("Klip tidak ditemukan")?;
        (c.project_id.clone(), c)
    };

    let text = clip.narration.trim().to_string();
    if text.is_empty() {
        return Err("Naskah klip ini masih kosong".into());
    }

    let project = {
        let conn = db.lock().unwrap();
        get_project(&conn, &project_id).ok_or("Proyek tidak ditemukan")?
    };

    let job = {
        let conn = db.lock().unwrap();
        insert_job(&conn, &project_id, "tts", &project.tts_provider, json!({}), Some(&clip_id), None)?
    };

    let j_id = job.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string();
    let _ = app.emit("event:job", &job);

    let db_clone = db.clone();
    let app_clone = app.clone();
    let c_id = clip_id.clone();
    let p_id = project_id.clone();

    job_mgr.spawn_task(&j_id, QueueName::Tts, move |ctx| async move {
        let (provider, voice, lang, model, key) = {
            let conn = db_clone.lock().unwrap();
            let p = get_project(&conn, &p_id).ok_or("Proyek tidak ditemukan")?;
            let s = get_settings(&conn);
            let prov = p.tts_provider.clone();
            let k = get_secret(&conn, &prov).unwrap_or_default();
            let m = if prov == "gemini" {
                // No fallback model here: an empty value is reported to the user
                // by gemini::speak_with_model instead of silently swapping in a
                // different model than the one chosen in Pengaturan.
                s.get("geminiTtsModel").and_then(|x| x.as_str()).unwrap_or("").to_string()
            } else {
                s.get("elevenModel").and_then(|x| x.as_str()).unwrap_or("eleven_multilingual_v2").to_string()
            };
            let v = if !p.tts_voice.is_empty() {
                p.tts_voice.clone()
            } else if prov == "gemini" {
                "Charon".to_string()
            } else {
                String::new()
            };
            (prov, v, p.language.clone(), m, k)
        };

        ctx.progress(0.2, Some("Membuat suara"));

        let (rel, words_opt, estimated) = if provider == "elevenlabs" {
            if voice.is_empty() {
                return Err("Pilih suara ElevenLabs dulu di langkah Ide cerita.".into());
            }
            let (audio, words) = elevenlabs::speak(&key, &voice, &model, &text, &lang).await?;
            let r = save_asset_file(&p_id, "audio", ".mp3", &audio)?;
            (r, Some(words), false)
        } else {
            let wav = gemini::speak_with_model(&key, &model, &text, &voice, &lang).await?;
            let r = save_asset_file(&p_id, "audio", ".wav", &wav)?;
            (r, None, true)
        };

        ctx.progress(0.85, Some("Mengukur durasi"));
        let abs = asset_abs_path(&p_id, &rel);
        let audio_ms = probe_duration_ms(&abs).await.unwrap_or(2000.0) as i64;

        let words = if let Some(w) = words_opt {
            w
        } else {
            estimate_word_timings(&strip_voice_tags(&text), (audio_ms as f64) / 1000.0)
        };

        let meta = json!({
            "words": words,
            "estimatedTimings": estimated,
            "voice": voice
        });

        let new_asset = NewAsset {
            project_id: p_id.clone(),
            clip_id: Some(c_id.clone()),
            character_id: None,
            kind: "audio".to_string(),
            provider: provider.clone(),
            model: Some(model),
            prompt: Some(text),
            remote_id: None,
            remote_url: None,
            local_path: rel,
            duration_ms: Some(audio_ms),
            width: None,
            height: None,
            meta,
        };

        let asset_row = {
            let conn = db_clone.lock().unwrap();
            insert_asset(&conn, new_asset)?
        };

        let _ = app_clone.emit("event:asset", &asset_row);

        let dur = (audio_ms + 400).max(1500);
        {
            let conn = db_clone.lock().unwrap();
            let _ = patch_clip(
                &conn,
                &c_id,
                json!({ "audioAssetId": asset_row.id, "durationMs": dur }),
            );
            let _ = touch_project(&conn, &p_id, None);
        }

        let _ = app_clone.emit(
            "event:clip",
            json!({
                "clipId": c_id,
                "patch": { "audioAssetId": asset_row.id, "durationMs": dur }
            }),
        );

        Ok(Some(asset_row.id))
    });

    Ok(job)
}

pub fn generate_missing_images(
    db: DbPool,
    job_mgr: Arc<JobManager>,
    app: AppHandle,
    project_id: String,
) -> Result<Vec<Value>, String> {
    let mut jobs = Vec::new();
    let (project, clips, chars) = {
        let conn = db.lock().unwrap();
        let p = get_project(&conn, &project_id).ok_or("Proyek tidak ditemukan")?;
        let cl = clips_of(&conn, &project_id);
        let ch = characters_of(&conn, &project_id);
        (p, cl, ch)
    };

    let model = get_image_model(project.image_model.as_deref());
    if max_refs(&model) > 0 {
        let needed_char_ids: std::collections::HashSet<String> = clips.iter().flat_map(|c| c.character_ids.clone()).collect();
        for c in chars {
            if needed_char_ids.contains(&c.id) && c.sheet_asset_id.is_none() {
                if let Ok(j) = generate_character_sheet(db.clone(), job_mgr.clone(), app.clone(), c.id) {
                    jobs.push(j);
                }
            }
        }
    }

    for c in clips {
        if c.image_asset_id.is_none() {
            if let Ok(j) = generate_clip_image(db.clone(), job_mgr.clone(), app.clone(), c.id) {
                jobs.push(j);
            }
        }
    }

    Ok(jobs)
}

pub fn generate_missing_videos(
    db: DbPool,
    job_mgr: Arc<JobManager>,
    app: AppHandle,
    project_id: String,
) -> Result<Vec<Value>, String> {
    let clips = {
        let conn = db.lock().unwrap();
        clips_of(&conn, &project_id)
    };

    if clips.iter().any(|c| c.image_asset_id.is_none()) {
        return Err("Buat gambar semua klip dulu sebelum menjadikannya video.".into());
    }

    let mut jobs = Vec::new();
    for c in clips {
        if c.video_asset_id.is_none() {
            if let Ok(j) = generate_clip_video(db.clone(), job_mgr.clone(), app.clone(), c.id) {
                jobs.push(j);
            }
        }
    }

    Ok(jobs)
}

pub fn generate_missing_tts(
    db: DbPool,
    job_mgr: Arc<JobManager>,
    app: AppHandle,
    project_id: String,
) -> Result<Vec<Value>, String> {
    let clips = {
        let conn = db.lock().unwrap();
        clips_of(&conn, &project_id)
    };

    let mut jobs = Vec::new();
    for c in clips {
        if c.audio_asset_id.is_none() && !c.narration.trim().is_empty() {
            if let Ok(j) = generate_clip_tts(db.clone(), job_mgr.clone(), app.clone(), c.id) {
                jobs.push(j);
            }
        }
    }

    Ok(jobs)
}

pub fn set_active_asset(
    db: &DbPool,
    app: &AppHandle,
    clip_id: &str,
    asset_id: &str,
) -> Result<(), String> {
    let (kind, duration_ms) = {
        let conn = db.lock().unwrap();
        let asset = get_asset(&conn, asset_id).ok_or("Aset tidak ditemukan")?;
        if asset.clip_id.as_deref() != Some(clip_id) {
            return Err("Aset tidak cocok dengan klip ini".into());
        }
        (asset.kind, asset.duration_ms)
    };

    let patch = match kind.as_str() {
        "image" => json!({ "imageAssetId": asset_id }),
        "video" => json!({ "videoAssetId": asset_id }),
        "audio" => {
            let dur = (duration_ms.unwrap_or(0) + 400).max(1500);
            json!({ "audioAssetId": asset_id, "durationMs": dur })
        }
        _ => json!({}),
    };

    {
        let conn = db.lock().unwrap();
        patch_clip(&conn, clip_id, patch.clone())?;
    }

    let _ = app.emit(
        "event:clip",
        json!({
            "clipId": clip_id,
            "patch": patch
        }),
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn image_turn_serializes_until_the_previous_image_finishes() {
        let gate = tokio::sync::Semaphore::new(1);
        let token = tokio_util::sync::CancellationToken::new();
        let first = acquire_image_turn(&gate, &token).await.unwrap();
        let second = acquire_image_turn(&gate, &token);
        tokio::pin!(second);
        assert!(tokio::time::timeout(std::time::Duration::from_millis(30), &mut second).await.is_err());
        drop(first);
        assert!(tokio::time::timeout(std::time::Duration::from_secs(1), &mut second).await.is_ok());
    }

    #[tokio::test]
    async fn canceled_image_turn_does_not_block_next_image() {
        let gate = tokio::sync::Semaphore::new(1);
        let first_token = tokio_util::sync::CancellationToken::new();
        let waiting_token = tokio_util::sync::CancellationToken::new();
        let first = acquire_image_turn(&gate, &first_token).await.unwrap();
        let waiting = acquire_image_turn(&gate, &waiting_token);
        tokio::pin!(waiting);
        assert!(tokio::time::timeout(std::time::Duration::from_millis(30), &mut waiting).await.is_err());
        waiting_token.cancel();
        assert!(waiting.await.is_err());
        drop(first);
        let next = acquire_image_turn(&gate, &first_token).await;
        assert!(next.is_ok());
    }

    #[test]
    fn clip_str_stops_before_a_multibyte_character() {
        assert_eq!(clip_str("ab😀cd", 3), "ab");
        assert_eq!(clip_str("ab😀cd", 2), "ab");
        assert_eq!(clip_str("éx", 1), "");
        assert_eq!(clip_str("abc", 0), "");
    }

    #[test]
    fn clip_prompt_leads_with_the_style() {
        let c = CharacterRow {
            id: "c1".into(),
            project_id: "p1".into(),
            name: "Maya".into(),
            description: "A young nurse. Default outfit: blue scrubs.".into(),
            sheet_asset_id: None,
            sort: 0,
        };
        let prompt = clip_prompt(
            "16:9",
            "High-quality anime film still, clean cel shading. No text.",
            "Setting: a hospital ward at dusk.\nAction: Maya checks a chart.",
            &[c],
            &[],
            4000,
        );
        assert!(prompt.starts_with("Art style (must match exactly): High-quality anime film still"));
        assert!(prompt.contains("Scene:\nSetting: a hospital ward"));
        assert!(prompt.ends_with("Frame format 16:9, full-bleed composition."));
    }

    #[test]
    fn clip_prompt_keeps_style_when_truncated() {
        let long_scene = "x".repeat(5000);
        let prompt = clip_prompt("16:9", "ANIME STYLE.", &long_scene, &[], &[], 400);
        assert!(prompt.starts_with("Art style (must match exactly): ANIME STYLE."));
        assert!(prompt.len() <= 400);
    }

    #[test]
    fn sheet_prompt_carries_the_style() {
        let c = CharacterRow {
            id: "c1".into(),
            project_id: "p1".into(),
            name: "Maya".into(),
            description: "A young nurse.".into(),
            sheet_asset_id: None,
            sort: 0,
        };
        let prompt = sheet_prompt(&c, "High-quality anime film still.", "an anime character with clean cel shading");
        assert!(prompt.contains("an anime character with clean cel shading"));
        assert!(prompt.contains("High-quality anime film still."));
        assert!(prompt.contains("Character reference sheet for \"Maya\""));
    }

    #[test]
    fn test_catalog_models_not_empty() {
        let models = crate::services::flow2api::flow_image_models();
        println!("Loaded {} flow image models", models.len());
        assert!(!models.is_empty(), "Flow image models should not be empty");
        let img = get_image_model(None);
        println!("Default image model: {}", img.id);
        let vid = get_video_model(None);
        println!("Default video model: {}", vid.id);
        let flow_img = get_image_model(Some("flow:gemini-3.1-flash-image"));
        println!("Flow image model: {}", flow_img.id);
        let flow_vid = get_video_model(Some("flow:veo-3.1-fast"));
        println!("Flow video model: {}", flow_vid.id);
        // Ids saved while the app used useapi.net still resolve to a real model.
        let aliased = get_image_model(Some("flow:nano-banana-2"));
        println!("Aliased flow image model: {}", aliased.id);
        assert_eq!(aliased.id, "flow:gemini-3.1-flash-image");
    }
}


pub async fn estimate_credits(
    _db: &DbPool,
    _kind: &str,
    model_id: &str,
    _aspect: &str,
    _seconds: Option<f64>,
    _resolution: Option<&str>,
) -> Option<Value> {
    if model_id.starts_with("flow:") {
        return Some(json!({
            "credits": 0,
            "usd": 0.0
        }));
    }
    None
}

