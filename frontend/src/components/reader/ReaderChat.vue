<script setup>
import { ref, watch, nextTick } from 'vue'
import { NInput } from 'naive-ui'
import MarkdownIt from 'markdown-it'
import { fetchSSEStream } from '@/composables/useSSEStream'

const props = defineProps({
  bookId:         { type: String, default: null },
  bookTitle:      { type: String, default: '' },
  currentChapter: { type: String, default: '' },
  selectedText:   { type: String, default: null },
})

const emit = defineEmits(['clear-selection'])

const md = new MarkdownIt({ breaks: true, linkify: false })

const messages   = ref([])
const inputText  = ref('')
const loading    = ref(false)
const messagesEl = ref(null)

// When new selectedText arrives, scroll to bottom and focus the quote bar
watch(() => props.selectedText, async (val) => {
  if (val) {
    await nextTick()
    scrollToBottom()
  }
})

watch(
  () => messages.value.length,
  async () => {
    await nextTick()
    scrollToBottom()
  }
)

function scrollToBottom() {
  if (messagesEl.value) {
    messagesEl.value.scrollTop = messagesEl.value.scrollHeight
  }
}

async function send() {
  const q = inputText.value.trim()
  if (!q || loading.value) return

  const selectedText = props.selectedText || null
  inputText.value = ''
  emit('clear-selection')

  const displayContent = selectedText ? `> ${selectedText}\n\n${q}` : q
  messages.value = [
    ...messages.value,
    { role: 'user', content: displayContent, id: `u-${Date.now()}` },
  ]
  loading.value = true

  const aiMsgId = `a-${Date.now()}`
  messages.value = [
    ...messages.value,
    { role: 'ai', content: '', citations: [], id: aiMsgId, streaming: true },
  ]

  const updateAiMsg = (patch) => {
    messages.value = messages.value.map((m) =>
      m.id === aiMsgId ? { ...m, ...patch } : m,
    )
  }

  try {
    await fetchSSEStream(
      '/chat/stream',
      {
        query: q,
        book_id: props.bookId || null,
        thread_id: 'reader-default',
        user_id: 'default',
        selected_text: selectedText,
        current_chapter: props.currentChapter || null,
      },
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
        onDone: (evt) => updateAiMsg({ citations: evt.citations ?? [], streaming: false }),
        onError: (msg) => updateAiMsg({ content: `请求失败：${msg}`, isError: true, isStatus: false, streaming: false }),
      },
    )
  } catch (e) {
    updateAiMsg({ content: `请求失败：${e.message}`, isError: true, streaming: false })
  } finally {
    loading.value = false
    const msg = messages.value.find((m) => m.id === aiMsgId)
    if (msg?.streaming) updateAiMsg({ streaming: false })
  }
}

function onKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
  }
}

function clearMessages() {
  messages.value = []
}
</script>

<template>
  <div class="reader-chat">
    <!-- Header -->
    <div class="chat-header">
      <span class="chat-title">AI 问答</span>
      <button v-if="messages.length" class="clear-btn" @click="clearMessages">清空</button>
    </div>

    <!-- Messages -->
    <div ref="messagesEl" class="messages">
      <!-- Empty hint -->
      <div v-if="messages.length === 0" class="empty-hint">
        <div class="hint-icon">💬</div>
        <p class="hint-text">选取左侧文字，或直接输入问题</p>
        <p class="hint-sub">AI 将结合书籍内容有根据地回答</p>
      </div>

      <TransitionGroup name="msg" tag="div" class="msg-list">
        <div
          v-for="msg in messages"
          :key="msg.id"
          class="message-row"
          :class="msg.role"
        >
          <div class="avatar" :class="msg.role === 'ai' ? 'ai-avatar' : 'user-avatar'">
            {{ msg.role === 'ai' ? '🤖' : '你' }}
          </div>
          <div class="bubble-wrap">
            <div class="bubble" :class="{ error: msg.isError }">
              <div v-if="msg.role === 'ai'" class="md-content">
                <span v-if="msg.isStatus" class="status-text">{{ msg.content }}</span>
                <span v-else v-html="md.render(msg.content || '')" />
                <span v-if="msg.streaming" class="stream-cursor" />
              </div>
              <div v-else class="user-content">
                <!-- Quote block if message contains quoted text -->
                <template v-if="msg.content.startsWith('> ')">
                  <blockquote class="quoted-text">{{ msg.content.split('\n\n')[0].slice(2) }}</blockquote>
                  <span>{{ msg.content.split('\n\n').slice(1).join('\n\n') }}</span>
                </template>
                <span v-else>{{ msg.content }}</span>
              </div>
            </div>
            <!-- Citations -->
            <div v-if="msg.citations && msg.citations.length" class="citations">
              <span
                v-for="(c, i) in msg.citations"
                :key="i"
                class="cite-chip"
                :title="c.snippet || ''"
              >
                {{ c.source || c.book_title }}
              </span>
            </div>
          </div>
        </div>
      </TransitionGroup>

      <!-- Typing indicator -->
      <div v-if="loading" class="message-row ai">
        <div class="avatar ai-avatar">🤖</div>
        <div class="bubble typing">
          <span class="dot" /><span class="dot" /><span class="dot" />
        </div>
      </div>
    </div>

    <!-- Input area -->
    <div class="input-area">
      <!-- Selected text quote bar -->
      <div v-if="selectedText" class="quote-bar">
        <span class="quote-icon">❝</span>
        <span class="quote-preview">{{ selectedText.length > 60 ? selectedText.slice(0, 60) + '…' : selectedText }}</span>
        <button class="quote-close" @click="emit('clear-selection')">✕</button>
      </div>

      <div class="input-row">
        <NInput
          v-model:value="inputText"
          type="textarea"
          :autosize="{ minRows: 1, maxRows: 4 }"
          :placeholder="selectedText ? '针对划选文字提问…' : '输入问题，Enter 发送…'"
          :disabled="loading"
          @keydown="onKeydown"
          class="chat-input"
        />
        <button
          class="send-btn"
          :disabled="!inputText.trim() || loading"
          @click="send"
        >
          ➤
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.reader-chat {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--bg);
}

/* ── Header ── */
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px 8px;
  border-bottom: 1px solid var(--border);
  background: white;
  flex-shrink: 0;
}

.chat-title {
  font-size: 12.5px;
  font-weight: 600;
  color: var(--text);
  font-family: var(--font-ui);
  letter-spacing: 0.3px;
}

.clear-btn {
  background: none;
  border: none;
  font-size: 11px;
  color: var(--text-muted);
  cursor: pointer;
  font-family: var(--font-ui);
  padding: 2px 6px;
  border-radius: 4px;
  transition: color 0.15s;
}
.clear-btn:hover { color: var(--accent); }

/* ── Messages ── */
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px 12px 8px;
  display: flex;
  flex-direction: column;
  gap: 0;
}

.msg-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* ── Empty hint ── */
.empty-hint {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 32px 16px;
  gap: 6px;
}
.hint-icon { font-size: 32px; margin-bottom: 4px; }
.hint-text { font-size: 13px; font-weight: 500; color: var(--text); }
.hint-sub  { font-size: 11.5px; color: var(--text-muted); }

/* ── Message rows ── */
.message-row {
  display: flex;
  gap: 8px;
  align-items: flex-start;
}

.message-row.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
}

.ai-avatar   { background: linear-gradient(135deg, var(--accent), #e8a060); }
.user-avatar { background: var(--sidebar-bg); color: #f0ebe0; font-size: 10px; font-weight: 600; }

.bubble-wrap {
  display: flex;
  flex-direction: column;
  gap: 5px;
  max-width: calc(100% - 40px);
}

.bubble {
  padding: 9px 12px;
  border-radius: 12px;
  font-size: 12.5px;
  line-height: 1.65;
  word-break: break-word;
}

.message-row.ai .bubble {
  background: white;
  border: 1px solid var(--border);
  border-top-left-radius: 3px;
  color: var(--text);
}

.message-row.user .bubble {
  background: var(--sidebar-bg);
  color: #f0ebe0;
  border-bottom-right-radius: 3px;
}

.bubble.error { background: #fff0f0; border-color: #fcc; color: #c00; }

/* ── User quote ── */
.user-content { display: flex; flex-direction: column; gap: 6px; }

blockquote.quoted-text {
  margin: 0;
  padding: 4px 8px;
  border-left: 2px solid rgba(255,255,255,0.4);
  font-size: 11px;
  color: rgba(240,235,224,0.7);
  font-style: italic;
  line-height: 1.5;
}

/* ── Markdown ── */
.md-content :deep(p)            { margin-bottom: 7px; }
.md-content :deep(p:last-child) { margin-bottom: 0; }
.md-content :deep(ul),
.md-content :deep(ol)           { padding-left: 18px; margin-bottom: 7px; }
.md-content :deep(li)           { margin-bottom: 3px; }
.md-content :deep(strong)       { color: var(--accent); }
.md-content :deep(code) {
  background: var(--accent-dim);
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 11.5px;
}

/* ── Citations ── */
.citations { display: flex; flex-wrap: wrap; gap: 4px; }
.cite-chip {
  background: var(--accent-dim);
  color: var(--accent);
  font-size: 10px;
  padding: 2px 7px;
  border-radius: 4px;
  font-weight: 500;
}

/* ── Typing indicator ── */
.typing {
  display: flex;
  gap: 4px;
  align-items: center;
  padding: 12px 14px;
}
.dot {
  width: 6px; height: 6px;
  background: var(--text-muted);
  border-radius: 50%;
  display: inline-block;
  animation: bounce 1.2s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
  0%, 60%, 100% { transform: translateY(0); }
  30%           { transform: translateY(-5px); }
}

/* ── Input area ── */
.input-area {
  flex-shrink: 0;
  border-top: 1px solid var(--border);
  background: white;
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

/* ── Quote bar ── */
.quote-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--accent-dim);
  border-radius: 6px;
  padding: 5px 8px;
  border-left: 3px solid var(--accent);
}
.quote-icon  { font-size: 12px; color: var(--accent); flex-shrink: 0; }
.quote-preview {
  flex: 1;
  font-size: 11px;
  color: var(--accent);
  font-style: italic;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.quote-close {
  background: none; border: none; color: var(--accent);
  font-size: 11px; cursor: pointer; padding: 0 2px; flex-shrink: 0;
  opacity: 0.7; transition: opacity 0.15s;
}
.quote-close:hover { opacity: 1; }

/* ── Input row ── */
.input-row {
  display: flex;
  gap: 8px;
  align-items: flex-end;
}

.chat-input { flex: 1; }
.chat-input :deep(.n-input__border)       { display: none; }
.chat-input :deep(.n-input__state-border) { display: none; }
.chat-input :deep(.n-input-wrapper)       { padding: 0; }
.chat-input :deep(.n-input__textarea-el)  { font-family: var(--font-ui); font-size: 12.5px; }

.send-btn {
  background: var(--accent);
  border: none;
  width: 34px; height: 34px;
  border-radius: 8px;
  color: white;
  font-size: 14px;
  cursor: pointer;
  transition: opacity 0.15s;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.send-btn:hover:not(:disabled) { opacity: 0.85; }
.send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

/* ── Status / streaming ── */
.status-text { color: var(--text-muted); font-style: italic; font-size: 12px; }
.stream-cursor {
  display: inline-block;
  width: 2px; height: 1em;
  background: var(--accent);
  margin-left: 2px;
  vertical-align: text-bottom;
  animation: blink 0.8s step-end infinite;
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0; }
}

/* ── Transition ── */
.msg-enter-active { animation: msgIn 0.2s ease; }
@keyframes msgIn {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
</style>
