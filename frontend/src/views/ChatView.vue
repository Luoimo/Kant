<script setup>
import { ref, computed, onMounted, nextTick, watch } from 'vue'
import { useRoute } from 'vue-router'
import { NSelect, NInput, NButton, NSpin } from 'naive-ui'
import { useChatStore } from '@/stores/chat'
import { useBooksStore } from '@/stores/books'
import MarkdownIt from 'markdown-it'

const route = useRoute()
const chatStore = useChatStore()
const booksStore = useBooksStore()
const md = new MarkdownIt({ breaks: true, linkify: false })

const inputText = ref('')
const messagesEl = ref(null)

onMounted(async () => {
  await booksStore.fetchBooks()
  if (route.query.bookId) {
    chatStore.selectedBookId = route.query.bookId
  }
})

const scrollToBottom = async () => {
  await nextTick()
  if (messagesEl.value) {
    messagesEl.value.scrollTop = messagesEl.value.scrollHeight
  }
}

// Scroll when messages are added or when streaming content grows
watch(
  () => {
    const streaming = chatStore.messages.find((m) => m.streaming)
    return [chatStore.messages.length, streaming?.content?.length ?? 0]
  },
  scrollToBottom,
)

const bookOptions = computed(() => [
  { label: '全部书籍', value: null },
  ...booksStore.books.map((b) => ({ label: b.title, value: b.id })),
])

async function send() {
  const q = inputText.value.trim()
  if (!q || chatStore.loading) return
  inputText.value = ''
  await chatStore.sendMessageStream(q)
}

function onKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
  }
}

const SUGGESTIONS = [
  '这本书的核心论点是什么？',
  '帮我总结第一章',
  '这个概念如何理解？',
  '作者的写作风格有何特点？',
]
</script>

<template>
  <div class="chat-view">
    <!-- ── Topbar ── -->
    <div class="topbar">
      <h1 class="page-title">AI 对话</h1>
      <div class="book-selector">
        <NSelect
          v-model:value="chatStore.selectedBookId"
          :options="bookOptions"
          placeholder="选择书籍范围"
          size="small"
          style="width: 200px;"
          clearable
        />
      </div>
      <button class="clear-btn" @click="chatStore.clearMessages()">清空对话</button>
    </div>

    <div class="chat-container">
      <!-- Empty / Suggestions -->
      <div v-if="chatStore.messages.length === 0" class="empty-state">
        <div class="empty-logo">📖</div>
        <h2 class="empty-title">有什么想深入探讨的？</h2>
        <p class="empty-sub">选择一本书，开始与 Kant AI 对话</p>
        <div class="suggestions">
          <button
            v-for="s in SUGGESTIONS"
            :key="s"
            class="suggestion-chip"
            @click="inputText = s"
          >
            {{ s }}
          </button>
        </div>
      </div>

      <!-- Messages -->
      <div v-else ref="messagesEl" class="messages">
        <TransitionGroup name="msg">
          <div
            v-for="msg in chatStore.messages"
            :key="msg.id"
            class="message-row"
            :class="msg.role"
          >
            <div class="avatar" :class="msg.role === 'ai' ? 'ai-avatar' : 'user-avatar'">
              {{ msg.role === 'ai' ? '🤖' : '你' }}
            </div>
            <div class="bubble-wrap">
              <div class="bubble" :class="{ error: msg.isError }">
                <!-- AI: render markdown -->
                <div v-if="msg.role === 'ai'" class="md-content">
                  <span v-if="msg.isStatus" class="status-text">{{ msg.content }}</span>
                  <span v-else v-html="md.render(msg.content || '')" />
                  <span v-if="msg.streaming" class="stream-cursor" />
                </div>
                <!-- User: plain text -->
                <span v-else>{{ msg.content }}</span>
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

        <!-- Typing indicator: only before first token arrives -->
        <div v-if="chatStore.loading" class="message-row ai">
          <div class="avatar ai-avatar">🤖</div>
          <div class="bubble typing">
            <span class="dot" /><span class="dot" /><span class="dot" />
          </div>
        </div>
      </div>

      <!-- Input area -->
      <div class="input-area">
        <NInput
          v-model:value="inputText"
          type="textarea"
          :autosize="{ minRows: 1, maxRows: 4 }"
          placeholder="输入问题，Shift+Enter 换行，Enter 发送…"
          :disabled="chatStore.loading"
          @keydown="onKeydown"
          class="chat-input"
        />
        <button
          class="send-btn"
          :disabled="!inputText.trim() || chatStore.loading"
          @click="send"
        >
          ➤
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.topbar {
  padding: 24px 28px 0;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}

.page-title {
  font-family: var(--font-serif);
  font-size: 22px;
  font-weight: 600;
  color: var(--text);
}

.book-selector { margin-left: auto; }

.clear-btn {
  background: none;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 5px 12px;
  font-size: 12px;
  color: var(--text-muted);
  cursor: pointer;
  font-family: var(--font-ui);
  transition: all 0.15s;
}
.clear-btn:hover { border-color: var(--accent); color: var(--accent); }

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 20px 28px 20px;
  gap: 16px;
}

/* ── Empty state ── */
.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding-bottom: 40px;
}

.empty-logo { font-size: 48px; margin-bottom: 4px; }
.empty-title { font-family: var(--font-serif); font-size: 20px; font-weight: 600; color: var(--text); }
.empty-sub { font-size: 13px; color: var(--text-muted); }

.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin-top: 8px;
}

.suggestion-chip {
  background: white;
  border: 1px solid var(--border);
  border-radius: 99px;
  padding: 7px 14px;
  font-size: 12.5px;
  color: var(--text-muted);
  cursor: pointer;
  font-family: var(--font-ui);
  transition: all 0.15s;
}
.suggestion-chip:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }

/* ── Messages ── */
.messages {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding-right: 4px;
}

.message-row {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  max-width: 88%;
}

.message-row.user {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
}

.ai-avatar { background: linear-gradient(135deg, var(--accent), #e8a060); }
.user-avatar { background: var(--sidebar-bg); color: #f0ebe0; font-size: 11px; font-weight: 600; }

.bubble-wrap {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.bubble {
  padding: 11px 15px;
  border-radius: 14px;
  font-size: 13.5px;
  line-height: 1.65;
}

.message-row.ai .bubble {
  background: white;
  border: 1px solid var(--border);
  border-top-left-radius: 4px;
  color: var(--text);
}

.message-row.user .bubble {
  background: var(--sidebar-bg);
  color: #f0ebe0;
  border-bottom-right-radius: 4px;
}

.bubble.error { background: #fff0f0; border-color: #fcc; color: #c00; }

/* ── Markdown content ── */
.md-content :deep(p) { margin-bottom: 8px; }
.md-content :deep(p:last-child) { margin-bottom: 0; }
.md-content :deep(ul), .md-content :deep(ol) { padding-left: 20px; margin-bottom: 8px; }
.md-content :deep(li) { margin-bottom: 4px; }
.md-content :deep(strong) { color: var(--accent); }
.md-content :deep(code) {
  background: var(--accent-dim);
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 12px;
}

/* ── Citations ── */
.citations {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.cite-chip {
  background: var(--accent-dim);
  color: var(--accent);
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 500;
  cursor: default;
}

/* ── Typing indicator ── */
.typing {
  display: flex;
  gap: 5px;
  align-items: center;
  padding: 14px 16px;
}

.dot {
  width: 7px;
  height: 7px;
  background: var(--text-muted);
  border-radius: 50%;
  display: inline-block;
  animation: bounce 1.2s infinite;
}

.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
  0%, 60%, 100% { transform: translateY(0); }
  30% { transform: translateY(-6px); }
}

/* ── Input area ── */
.input-area {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  background: white;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 12px;
  flex-shrink: 0;
}

.chat-input { flex: 1; }
.chat-input :deep(.n-input__border) { display: none; }
.chat-input :deep(.n-input__state-border) { display: none; }
.chat-input :deep(.n-input-wrapper) { padding: 0; }
.chat-input :deep(.n-input__textarea-el) { font-family: var(--font-ui); font-size: 13.5px; }

.send-btn {
  background: var(--accent);
  border: none;
  width: 38px;
  height: 38px;
  border-radius: 9px;
  color: white;
  font-size: 16px;
  cursor: pointer;
  transition: opacity 0.15s;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.send-btn:hover:not(:disabled) { opacity: 0.85; }
.send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

/* ── Status text (thinking / tool call) ── */
.status-text {
  color: var(--text-muted);
  font-style: italic;
  font-size: 13px;
}

/* ── Streaming cursor ── */
.stream-cursor {
  display: inline-block;
  width: 2px;
  height: 1em;
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
.msg-enter-active { animation: msgIn 0.25s ease; }
@keyframes msgIn {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
</style>
