const ACCESS_TOKEN_KEY = 'access_token'
const REFRESH_TOKEN_KEY = 'refresh_token'
const USER_ID_KEY = 'user_id'
const ROLE_KEY = 'role'

const DEFAULT_PREEMPTIVE_SECONDS = 90
const FOCUS_REFRESH_SECONDS = 120
const AUTH_CHANGED_EVENT = 'auth:changed'

let refreshPromise = null

function _emitAuthChanged(reason = 'updated') {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent(AUTH_CHANGED_EVENT, { detail: { reason } }))
}

function _toPathname(url) {
  try {
    return new URL(url, window.location.origin).pathname
  } catch {
    return ''
  }
}

export function isAuthEndpoint(url = '') {
  const path = _toPathname(url)
  return path.startsWith('/auth/')
}

function _decodeJwtPayload(token) {
  if (!token) return null
  const parts = token.split('.')
  if (parts.length < 2) return null
  const b64 = parts[1].replace(/-/g, '+').replace(/_/g, '/')
  const padded = b64 + '='.repeat((4 - (b64.length % 4)) % 4)
  try {
    const json = atob(padded)
    return JSON.parse(json)
  } catch {
    return null
  }
}

function _nowSeconds() {
  return Math.floor(Date.now() / 1000)
}

export function readStoredAuth() {
  return {
    accessToken: localStorage.getItem(ACCESS_TOKEN_KEY) || '',
    refreshToken: localStorage.getItem(REFRESH_TOKEN_KEY) || '',
    userId: localStorage.getItem(USER_ID_KEY) || '',
    role: localStorage.getItem(ROLE_KEY) || '',
  }
}

export function getAccessToken() {
  return localStorage.getItem(ACCESS_TOKEN_KEY) || ''
}

export function getRefreshToken() {
  return localStorage.getItem(REFRESH_TOKEN_KEY) || ''
}

export function setStoredAuth({
  accessToken = '',
  refreshToken = '',
  userId = '',
  role = '',
} = {}) {
  localStorage.setItem(ACCESS_TOKEN_KEY, accessToken || '')
  localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken || '')
  localStorage.setItem(USER_ID_KEY, userId || '')
  localStorage.setItem(ROLE_KEY, role || '')
  _emitAuthChanged('set')
}

export function clearStoredAuth() {
  localStorage.setItem(ACCESS_TOKEN_KEY, '')
  localStorage.setItem(REFRESH_TOKEN_KEY, '')
  localStorage.setItem(USER_ID_KEY, '')
  localStorage.setItem(ROLE_KEY, '')
  _emitAuthChanged('clear')
}

export function isAccessTokenExpiringSoon(thresholdSeconds = DEFAULT_PREEMPTIVE_SECONDS) {
  const token = getAccessToken()
  if (!token) return false
  const payload = _decodeJwtPayload(token)
  const exp = Number(payload?.exp || 0)
  if (!exp) return true
  return exp - _nowSeconds() <= thresholdSeconds
}

function _extractErrorMessage(payload, status) {
  if (!payload) return `HTTP ${status}`
  if (typeof payload.detail === 'string' && payload.detail.trim()) return payload.detail
  if (typeof payload.message === 'string' && payload.message.trim()) return payload.message
  return `HTTP ${status}`
}

export async function refreshAccessToken() {
  if (refreshPromise) return refreshPromise

  const currentRefresh = getRefreshToken()
  if (!currentRefresh) {
    clearStoredAuth()
    throw new Error('missing refresh token')
  }

  refreshPromise = (async () => {
    const response = await fetch('/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: currentRefresh }),
    })

    let payload = null
    try {
      payload = await response.json()
    } catch {
      payload = null
    }

    if (!response.ok) {
      clearStoredAuth()
      throw new Error(_extractErrorMessage(payload, response.status))
    }

    const nextAccess = payload?.access_token || ''
    const nextRefresh = payload?.refresh_token || ''
    if (!nextAccess || !nextRefresh) {
      clearStoredAuth()
      throw new Error('invalid refresh response')
    }

    const current = readStoredAuth()
    setStoredAuth({
      accessToken: nextAccess,
      refreshToken: nextRefresh,
      userId: current.userId,
      role: current.role,
    })
    return nextAccess
  })()

  try {
    return await refreshPromise
  } finally {
    refreshPromise = null
  }
}

export async function ensureFreshAccessToken({
  thresholdSeconds = DEFAULT_PREEMPTIVE_SECONDS,
  force = false,
} = {}) {
  const access = getAccessToken()
  if (!access) return ''

  if (force || isAccessTokenExpiringSoon(thresholdSeconds)) {
    return refreshAccessToken()
  }
  return access
}

export async function fetchWithAuthRefresh(
  url,
  init = {},
  {
    thresholdSeconds = DEFAULT_PREEMPTIVE_SECONDS,
    retryOn401 = true,
    preemptive = true,
  } = {},
) {
  const headers = new Headers(init.headers || {})
  const attachToken = async () => {
    if (isAuthEndpoint(url)) return
    if (preemptive) {
      await ensureFreshAccessToken({ thresholdSeconds })
    }
    const token = getAccessToken()
    if (token) headers.set('Authorization', `Bearer ${token}`)
  }

  await attachToken()
  let response = await fetch(url, { ...init, headers })

  if (response.status === 401 && retryOn401 && !isAuthEndpoint(url)) {
    await refreshAccessToken()
    const token = getAccessToken()
    if (token) headers.set('Authorization', `Bearer ${token}`)
    response = await fetch(url, { ...init, headers })
  }
  return response
}

export function setupFocusRefresh() {
  if (typeof window === 'undefined') return () => {}

  const handler = async () => {
    try {
      const hasAccess = !!getAccessToken()
      const hasRefresh = !!getRefreshToken()
      if (!hasAccess || !hasRefresh) return
      if (document.visibilityState && document.visibilityState !== 'visible') return
      await ensureFreshAccessToken({ thresholdSeconds: FOCUS_REFRESH_SECONDS })
    } catch {
      // Ignore; a failed refresh will clear auth state via token manager.
    }
  }

  window.addEventListener('focus', handler)
  document.addEventListener('visibilitychange', handler)
  return () => {
    window.removeEventListener('focus', handler)
    document.removeEventListener('visibilitychange', handler)
  }
}

export { AUTH_CHANGED_EVENT }
