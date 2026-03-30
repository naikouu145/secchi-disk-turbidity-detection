function resolveApiBaseUrl() {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim()
  if (!configured) {
    return '/api'
  }

  const cleaned = configured.replace(/\/+$/, '')
  const isAbsolute = /^https?:\/\//i.test(cleaned)

  if (!isAbsolute) {
    return cleaned
  }

  try {
    const url = new URL(cleaned)
    const pathname = url.pathname.replace(/\/+$/, '')
    return pathname ? `${url.origin}${pathname}` : `${url.origin}/api`
  } catch {
    return cleaned
  }
}

const API_BASE_URL = resolveApiBaseUrl()

const DEFAULT_TIMEOUT_MS = 15000
const DEFAULT_RETRY_COUNT = 2
const DEFAULT_CACHE_TTL_MS = 30000

let authToken = null
const responseCache = new Map()

export function setAuthToken(token) {
  authToken = token || null
}

export function clearAuthToken() {
  authToken = null
}

class ApiError extends Error {
  constructor(message, { status = null, code = 'UNKNOWN', details = null } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function getCachedResponse(cacheKey) {
  if (!cacheKey || !responseCache.has(cacheKey)) {
    return null
  }

  const cached = responseCache.get(cacheKey)
  if (!cached || Date.now() > cached.expiresAt) {
    responseCache.delete(cacheKey)
    return null
  }

  return cached.value
}

function setCachedResponse(cacheKey, value, ttlMs = DEFAULT_CACHE_TTL_MS) {
  if (!cacheKey) {
    return
  }

  responseCache.set(cacheKey, {
    value,
    expiresAt: Date.now() + ttlMs,
  })
}

function invalidateCache(cacheKeys = []) {
  cacheKeys.forEach((cacheKey) => {
    responseCache.delete(cacheKey)
  })
}

function getUserFriendlyError(error) {
  if (error instanceof ApiError) {
    if (error.code === 'TIMEOUT') {
      return 'Request timed out. Please try again.'
    }
    if (error.code === 'NETWORK') {
      return 'Cannot reach backend service. Check your connection or backend status.'
    }
    if (error.status === 400) {
      return error.message || 'Invalid request. Please check your input and try again.'
    }
    if (error.status === 401) {
      return 'Unauthorized request. Please sign in again.'
    }
    if (error.status === 404) {
      return 'Requested resource was not found.'
    }
    if (error.status >= 500) {
      return 'Backend encountered an error. Please try again shortly.'
    }
    return error.message
  }

  return 'Unexpected error occurred. Please try again.'
}

function buildHeaders(baseHeaders = {}, isFormData = false, method = 'GET') {
  const headers = {
    Accept: 'application/json',
    ...baseHeaders,
  }

  if (authToken) {
    headers.Authorization = `Bearer ${authToken}`
  }

  // For FormData uploads, do not set Content-Type manually.
  // The browser injects multipart/form-data with the proper boundary.
  const normalizedMethod = String(method || 'GET').toUpperCase()
  const shouldSetJsonContentType = !isFormData && !['GET', 'HEAD'].includes(normalizedMethod)

  if (shouldSetJsonContentType && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json'
  }

  return headers
}

async function parseResponse(response) {
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    return response.json()
  }
  return null
}

async function apiRequest(path, options = {}) {
  const {
    timeoutMs = DEFAULT_TIMEOUT_MS,
    retryCount = DEFAULT_RETRY_COUNT,
    retryDelayMs = 400,
    isFormData = false,
    cacheKey = null,
    cacheTtlMs = DEFAULT_CACHE_TTL_MS,
    invalidateCacheKeys = [],
    ...fetchOptions
  } = options

  const method = (fetchOptions.method || 'GET').toUpperCase()
  const headers = buildHeaders(fetchOptions.headers, isFormData, method)

  if (method === 'GET') {
    const cached = getCachedResponse(cacheKey)
    if (cached) {
      return cached
    }
  }

  let attempt = 0
  let lastError = null

  while (attempt <= retryCount) {
    const controller = new AbortController()
    const timeoutHandle = setTimeout(() => controller.abort(), timeoutMs)

    try {
      const response = await fetch(`${API_BASE_URL}${path}`, {
        ...fetchOptions,
        headers,
        signal: controller.signal,
      })

      clearTimeout(timeoutHandle)

      const payload = await parseResponse(response)

      if (!response.ok) {
        const detail = payload?.detail || payload?.message || `Request failed with status ${response.status}`
        const apiError = new ApiError(detail, {
          status: response.status,
          code: response.status >= 500 ? 'SERVER' : 'CLIENT',
          details: payload,
        })

        if (response.status >= 500 && attempt < retryCount) {
          attempt += 1
          await sleep(retryDelayMs * attempt)
          continue
        }

        throw apiError
      }

      if (method === 'GET' && cacheKey) {
        setCachedResponse(cacheKey, payload, cacheTtlMs)
      }

      if (invalidateCacheKeys.length > 0) {
        invalidateCache(invalidateCacheKeys)
      }

      return payload
    } catch (error) {
      clearTimeout(timeoutHandle)

      if (error.name === 'AbortError') {
        lastError = new ApiError('Request timeout', {
          code: 'TIMEOUT',
        })
      } else if (error instanceof ApiError) {
        lastError = error
      } else {
        lastError = new ApiError(error.message || 'Network request failed', {
          code: 'NETWORK',
        })
      }

      if (attempt < retryCount && (lastError.code === 'NETWORK' || lastError.code === 'TIMEOUT')) {
        attempt += 1
        await sleep(retryDelayMs * attempt)
        continue
      }

      break
    }
  }

  const userMessage = getUserFriendlyError(lastError)
  throw new ApiError(userMessage, {
    status: lastError?.status || null,
    code: lastError?.code || 'UNKNOWN',
    details: lastError?.details || null,
  })
}

export async function assessImage(file, config = {}) {
  const formData = new FormData()
  formData.append('file', file)

  const params = new URLSearchParams()
  if (typeof config.adaptiveScoring === 'boolean') {
    params.set('adaptive_scoring', String(config.adaptiveScoring))
  }
  if (config.overrideSource) {
    params.set('override_source', String(config.overrideSource))
  }

  const query = params.toString() ? `?${params.toString()}` : ''
  return apiRequest(`/assess${query}`, {
    method: 'POST',
    body: formData,
    isFormData: true,
    timeoutMs: config.timeoutMs,
    retryCount: config.retryCount,
  })
}

export async function assessBatch(files, config = {}) {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))

  const params = new URLSearchParams()
  if (typeof config.adaptiveScoring === 'boolean') {
    params.set('adaptive_scoring', String(config.adaptiveScoring))
  }

  const query = params.toString() ? `?${params.toString()}` : ''
  return apiRequest(`/assess/batch${query}`, {
    method: 'POST',
    body: formData,
    isFormData: true,
    timeoutMs: config.timeoutMs,
    retryCount: config.retryCount,
  })
}

export async function getConfig() {
  return apiRequest('/config', {
    cacheKey: 'config',
    cacheTtlMs: 30000,
  })
}

export async function updateConfig(configPayload) {
  return apiRequest('/config', {
    method: 'POST',
    body: JSON.stringify(configPayload),
    invalidateCacheKeys: ['config', 'health'],
  })
}

export async function checkHealth() {
  return apiRequest('/health', {
    cacheKey: 'health',
    cacheTtlMs: 10000,
  })
}

export { API_BASE_URL, ApiError }
