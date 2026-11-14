import { FLOW_IMAGE_MODELS, FLOW_VIDEO_MODELS, mapLegacyFlowModel } from './flow'
import type { ModelOption } from './types'

/**
 * Model helpers shared by the storyboard UI. Every image and video model now comes from
 * Google Flow (see ./flow.ts); the file keeps its old name so imports stay stable.
 */

interface FieldSchema {
  type?: string
  enum?: (string | number)[]
  default?: unknown
  minimum?: number
  maximum?: number
  maxLength?: number
  maxItems?: number
}

export interface HfModel {
  id: string
  name: string
  family: string
  kind: 'image' | 'video'
  /** Where the clip image (video) or the character sheets (image) go in the request. */
  input: { field: string | null; mode: 'none' | 'first-frame' | 'reference'; max: number }
  required: string[]
  props: Record<string, FieldSchema>
  durations: { values: number[] } | { min: number; max: number } | null
  tags: string[]
}

export const HF_IMAGE_MODELS = FLOW_IMAGE_MODELS
export const HF_VIDEO_MODELS = FLOW_VIDEO_MODELS

export const DEFAULT_IMAGE_MODEL = 'flow:gemini-3.1-flash-image'
export const DEFAULT_VIDEO_MODEL = 'flow:veo-3.1-fast'

export function getImageModel(id: string | null | undefined): HfModel {
  if (id) {
    const target = mapLegacyFlowModel(id)
    const hit = FLOW_IMAGE_MODELS.find((m) => m.id === target)
    if (hit) return hit
  }
  return FLOW_IMAGE_MODELS[0]
}

export function getVideoModel(id: string | null | undefined): HfModel {
  if (id) {
    const target = mapLegacyFlowModel(id)
    const hit = FLOW_VIDEO_MODELS.find((m) => m.id === target)
    if (hit) return hit
  }
  return FLOW_VIDEO_MODELS[0]
}

/** How many character sheets an image model accepts as references. 0 means sheets are not used. */
export function maxRefs(m: HfModel): number {
  return m.kind === 'image' && m.input.mode === 'reference' ? m.input.max : 0
}

/** Longest prompt the model accepts. Endpoints without a documented limit get a conservative one. */
export function promptLimit(m: HfModel): number {
  return m.props.prompt?.maxLength ?? (m.kind === 'video' ? 2500 : 4000)
}

/** The video length to request for a clip: the shortest allowed length that covers it, or the longest. */
export function fitDuration(m: HfModel, seconds: number): number | null {
  const d = m.durations
  if (!d) return null
  const want = Math.max(1, Math.ceil(seconds))
  if ('values' in d) {
    const values = [...d.values].sort((a, b) => a - b)
    return values.find((v) => v >= want) ?? values[values.length - 1]
  }
  return Math.min(d.max, Math.max(d.min, want))
}

export function durationLabel(m: HfModel): string | null {
  const d = m.durations
  if (!d) return null
  return 'values' in d ? `${d.values.join(' / ')} dtk` : `${d.min}–${d.max} dtk`
}

function pick(s: FieldSchema | undefined, prefer: (string | number)[]): string | number | undefined {
  if (!s?.enum?.length) return undefined
  for (const p of prefer) if (s.enum.includes(p)) return p
  return (s.default as string | number | undefined) ?? s.enum[0]
}

/** The resolution sent to the model: the requested one when the model offers it, else the recommended one. */
function resolutionFor(m: HfModel, want?: string | null): string | number | undefined {
  if (want && m.props.resolution?.enum?.some((v) => String(v) === want)) return m.props.resolution.enum.find((v) => String(v) === want)
  return pick(m.props.resolution, m.kind === 'image' ? ['2k', '2K', '1080p', '1k'] : ['1080p', '2K', '2k', '720p'])
}

/** Resolutions a model offers, lowest first, as picker values. */
export function resolutionOptions(m: HfModel): string[] {
  const rank = (v: string): number => {
    const k = /^(\d+(?:\.\d+)?)k$/i.exec(v)
    return k ? Number(k[1]) * 1000 : parseInt(v, 10) || 0
  }
  return (m.props.resolution?.enum ?? []).map(String).sort((a, b) => rank(a) - rank(b))
}

/** The resolution a project's videos actually use with this model. */
export function effectiveResolution(m: HfModel, want?: string | null): string | null {
  const r = resolutionFor(m, want)
  return r === undefined ? null : String(r)
}

/** What the app will actually request from this model, in words. */
export function describeModel(m: HfModel): string {
  const res = resolutionFor(m)?.toString().replace(/k$/, 'K')
  if (m.kind === 'image') {
    const refs = maxRefs(m)
    return [refs ? `pakai lembar karakter (maks ${refs})` : 'tanpa lembar karakter', res && `hasil ${res}`].filter(Boolean).join(' · ')
  }
  // Video resolution is picked per project, next to the model.
  return [durationLabel(m), m.input.mode === 'reference' ? 'gambar klip jadi referensi' : 'mulai dari gambar klip'].filter(Boolean).join(' · ')
}

/** Picker rows for Google Flow models, default first with a "Rekomendasi" tag. */
export function hfModelOptions(kind: 'image' | 'video'): ModelOption[] {
  const list = kind === 'image' ? FLOW_IMAGE_MODELS : FLOW_VIDEO_MODELS
  const recommended = kind === 'image' ? DEFAULT_IMAGE_MODEL : DEFAULT_VIDEO_MODEL
  return [...list.filter((m) => m.id === recommended), ...list.filter((m) => m.id !== recommended)].map((m) => ({
    id: m.id,
    name: m.name,
    family: m.family,
    description: describeModel(m),
    tags: m.id === recommended ? ['Rekomendasi', ...m.tags.filter((t) => t !== 'Rekomendasi')] : m.tags
  }))
}
