import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { booksApi } from '@/api'

// Deterministic pastel gradient from title string
const GRADIENTS = [
  'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
  'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
  'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
  'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
  'linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)',
  'linear-gradient(135deg, #c47c3e 0%, #e8a060 100%)',
  'linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)',
]

export function bookGradient(title = '') {
  let hash = 0
  for (let i = 0; i < title.length; i++) {
    hash = title.charCodeAt(i) + ((hash << 5) - hash)
  }
  return GRADIENTS[Math.abs(hash) % GRADIENTS.length]
}

export const useBooksStore = defineStore('books', () => {
  const books = ref([])
  const loading = ref(false)
  const error = ref(null)

  const readingBooks = computed(() => books.value.filter((b) => b.status === 'reading'))
  const doneBooks = computed(() => books.value.filter((b) => b.status === 'done'))
  const currentBook = computed(() => readingBooks.value[0] ?? null)

  async function fetchBooks() {
    loading.value = true
    error.value = null
    try {
      const { data } = await booksApi.list()
      books.value = data
    } catch (e) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function uploadBook(file, onProgress) {
    const { data } = await booksApi.upload(file, onProgress)
    await fetchBooks()
    return data
  }

  async function deleteBook(bookId) {
    const { data } = await booksApi.remove(bookId)
    books.value = books.value.filter((b) => b.id !== bookId)
    return data
  }

  function getCoverUrl(book) {
    if (!book?.cover_path) return null
    const filename = book.cover_path.split(/[/\\]/).pop()
    return `/covers/${filename}`
  }

  return {
    books,
    loading,
    error,
    readingBooks,
    doneBooks,
    currentBook,
    fetchBooks,
    uploadBook,
    deleteBook,
    getCoverUrl,
  }
})
