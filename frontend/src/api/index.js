import axios from 'axios'
import {
  clearStoredAuth,
  ensureFreshAccessToken,
  getAccessToken,
  isAuthEndpoint,
  refreshAccessToken,
} from '@/auth/tokenManager'
import { apiBaseUrl } from './baseUrl'

const api = axios.create({
  baseURL: apiBaseUrl || '/',
  timeout: 30000,
})

api.interceptors.request.use(async (config) => {
  if (!isAuthEndpoint(config.url || '')) {
    await ensureFreshAccessToken()
  }

  const token = getAccessToken()
  if (token && !isAuthEndpoint(config.url || '')) {
    config.headers = config.headers || {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const response = error?.response
    const original = error?.config || {}
    const url = original.url || ''

    if (!response || response.status !== 401 || isAuthEndpoint(url) || original._retry) {
      throw error
    }

    original._retry = true
    try {
      await refreshAccessToken()
      const token = getAccessToken()
      original.headers = original.headers || {}
      if (token) original.headers.Authorization = `Bearer ${token}`
      return api(original)
    } catch {
      clearStoredAuth()
      throw error
    }
  },
)

export const booksApi = {
  list: () => api.get('/books'),
  upload: (file, onProgress) => {
    const form = new FormData()
    form.append('file', file)
    return api.post('/books/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 180000,
      onUploadProgress: onProgress,
    })
  },
  remove: (bookId) => api.delete(`/books/${bookId}`, { timeout: 60000 }),
}

export const chatApi = {
  send: (payload) => api.post('/api/user/chat', payload, { timeout: 60000 }),
  history: (conversationId) => api.get('/chat/history', { params: { conversation_id: conversationId } }),
  clearHistory: (conversationId) => api.delete('/chat/history', { params: { conversation_id: conversationId } }),
}

export const authApi = {
  register: (payload) => api.post('/auth/register', payload),
  login: (payload) => api.post('/auth/login', payload),
  refresh: (payload) => api.post('/auth/refresh', payload),
  logout: (payload) => api.post('/auth/logout', payload),
  logoutAll: (payload) => api.post('/auth/logout-all', payload),
}

export const conversationsApi = {
  create: (bookId, payload = { title: '' }) => api.post(`/api/user/books/${bookId}/conversations`, payload),
  list: (bookId) => api.get(`/api/user/books/${bookId}/conversations`),
}

export const adminApi = {
  users: () => api.get('/api/admin/users'),
  userBooks: (userId) => api.get(`/api/admin/users/${userId}/books`),
  userBookConversations: (userId, bookId) => api.get(`/api/admin/users/${userId}/books/${bookId}/conversations`),
}

export const notesApi = {
  books: () => api.get('/notes/books'),
  timeline: (bookId) =>
    api.get('/notes/timeline', { params: bookId ? { book_id: bookId } : {} }),
  content: (bookId) => api.get(`/notes/${bookId}`),
  append: (bookId, content) =>
    api.post('/notes/append', { book_id: bookId, content }),
}

export const readerApi = {
  init: (bookId, readingGoal = '') =>
    api.post(`/reader/${bookId}/init`, { reading_goal: readingGoal }),
  plan: (bookId) => api.get(`/reader/${bookId}/plan`),
  progress: (bookId, chapter) =>
    api.post(`/reader/${bookId}/progress`, { chapter }),
}
