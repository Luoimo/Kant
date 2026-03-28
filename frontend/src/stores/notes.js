import { defineStore } from 'pinia'
import { ref } from 'vue'
import { notesApi } from '@/api'

export const useNotesStore = defineStore('notes', () => {
  const noteBooks = ref([])
  const timeline = ref([])
  const currentContent = ref(null)
  const loading = ref(false)
  const selectedBookId = ref(null)

  async function fetchNoteBooks() {
    const { data } = await notesApi.books()
    noteBooks.value = data
  }

  async function fetchTimeline(bookId = null) {
    loading.value = true
    try {
      const { data } = await notesApi.timeline(bookId)
      timeline.value = data.entries ?? []
    } finally {
      loading.value = false
    }
  }

  async function fetchContent(bookId) {
    loading.value = true
    try {
      const { data } = await notesApi.content(bookId)
      currentContent.value = data
    } finally {
      loading.value = false
    }
  }

  async function appendNote(bookId, content) {
    const { data } = await notesApi.append(bookId, content)
    await fetchContent(bookId)
    return data
  }

  return {
    noteBooks,
    timeline,
    currentContent,
    loading,
    selectedBookId,
    fetchNoteBooks,
    fetchTimeline,
    fetchContent,
    appendNote,
  }
})
