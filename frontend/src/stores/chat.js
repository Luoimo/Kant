import { defineStore } from 'pinia'
import { ref } from 'vue'
import { fetchSSEStream } from '@/composables/useSSEStream'

export const useChatStore = defineStore('chat', () => {
  const messages = ref([])
  const loading = ref(false)
  const selectedBookId = ref(null)
  const threadId = ref('default')

  async function sendMessageStream(query) {
    const userMsg = { role: 'user', content: query, id: `u-${Date.now()}` }
    messages.value = [...messages.value, userMsg]
    loading.value = true

    const aiMsgId = `a-${Date.now()}`
    messages.value = [
      ...messages.value,
      { role: 'ai', content: '', citations: [], intent: null, id: aiMsgId, streaming: true },
    ]

    const updateAiMsg = (patch) => {
      messages.value = messages.value.map((m) =>
        m.id === aiMsgId ? { ...m, ...patch } : m,
      )
    }

    try {
      await fetchSSEStream(
        '/chat/stream',
        { query, book_id: selectedBookId.value || null, thread_id: threadId.value, user_id: 'default' },
        {
          onThinking: () => {
            updateAiMsg({ content: '正在思考…', isStatus: true })
            loading.value = false
          },
          onStatus: (text) => updateAiMsg({ content: text, isStatus: true }),
          onToken: (text) => {
            const current = messages.value.find((m) => m.id === aiMsgId)
            const base = current?.isStatus ? '' : (current?.content ?? '')
            updateAiMsg({ content: base + text, isStatus: false })
            if (loading.value) loading.value = false
          },
          onDone: (evt) => updateAiMsg({ citations: evt.citations ?? [], followups: evt.followups ?? [], streaming: false }),
          onError: (msg) => updateAiMsg({ content: `请求失败：${msg}`, isError: true, isStatus: false, streaming: false }),
        },
      )
    } catch (e) {
      updateAiMsg({ content: `请求失败：${e.message}`, isError: true, streaming: false })
      throw e
    } finally {
      loading.value = false
      // Ensure streaming flag is cleared even on unexpected exit
      const msg = messages.value.find((m) => m.id === aiMsgId)
      if (msg?.streaming) updateAiMsg({ streaming: false })
    }
  }

  function clearMessages() {
    messages.value = []
  }

  return { messages, loading, selectedBookId, threadId, sendMessageStream, clearMessages }
})
