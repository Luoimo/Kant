const rawBaseUrl = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')

export const apiBaseUrl = rawBaseUrl

export function withApiBaseUrl(path) {
  if (!apiBaseUrl) return path
  if (/^https?:\/\//i.test(path)) return path
  return `${apiBaseUrl}${path.startsWith('/') ? path : `/${path}`}`
}
