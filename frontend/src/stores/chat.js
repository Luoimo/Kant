import { defineStore } from 'pinia'
import { ref } from 'vue'
import { fetchSSEStream } from '@/composables/useSSEStream'
import { chatApi, conversationsApi } from '@/api'

export const useChatStore = defineStore('chat', () => {
  const messages = ref([])
  const loading = ref(false)
  const selectedBookId = ref(null)
  const conversationId = ref(null)

  async function ensureConversation() {
    if (!selectedBookId.value) return null
    if (conversationId.value) return conversationId.value
    const { data } = await conversationsApi.create(selectedBookId.value, { title: '' })
    conversationId.value = data.conversation_id
    return conversationId.value
  }

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
      const cid = await ensureConversation()
      await fetchSSEStream(
        '/api/user/chat/stream',
        { query, book_id: selectedBookId.value || null, conversation_id: cid },
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

  async function clearMessages() {
    messages.value = []
    try {
      if (conversationId.value) {
        await chatApi.clearHistory(conversationId.value)
      }
    } catch (e) {
      console.error('Failed to clear chat history on server:', e)
    }
  }

  return { messages, loading, selectedBookId, conversationId, sendMessageStream, clearMessages, ensureConversation }
})
