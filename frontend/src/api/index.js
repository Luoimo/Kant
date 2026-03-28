import axios from 'axios'

const api = axios.create({
  baseURL: '/',
  timeout: 30000,
})

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
}

export const chatApi = {
  send: (payload) => api.post('/chat', payload, { timeout: 60000 }),
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
