<script setup>
import { computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { NSpin, NEmpty, NTag, NProgress } from 'naive-ui'
import { useBooksStore, bookGradient } from '@/stores/books'
import { useChatStore } from '@/stores/chat'

const router = useRouter()
const booksStore = useBooksStore()

onMounted(() => booksStore.fetchBooks())

const currentBook = computed(() => booksStore.currentBook)

const STATUS_LABEL = { unread: '未读', reading: '在读', done: '已完成' }
const STATUS_TYPE = { unread: 'default', reading: 'warning', done: 'success' }

function formatDate(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}

function pct(book) {
  return Math.round((book.progress ?? 0) * 100)
}

function openReader(book) {
  router.push({ name: 'reader', params: { bookId: book.id } })
}

function openChat(book) {
  const chatStore = useChatStore()
  chatStore.selectedBookId = book.id
  router.push({ name: 'chat', query: { bookId: book.id } })
}
</script>

<template>
  <div class="library-view">
    <!-- ── Topbar ── -->
    <div class="topbar">
      <h1 class="page-title">我的书库</h1>
      <div class="topbar-count" v-if="!booksStore.loading">
        {{ booksStore.books.length }} 本书
      </div>
    </div>

    <div class="content">
      <NSpin :show="booksStore.loading" description="加载中…">

        <!-- Currently Reading Hero -->
        <template v-if="currentBook">
          <div class="section-header">
            <span class="section-title">当前在读</span>
          </div>
          <div class="hero-card">
            <div
              class="hero-cover"
              :style="booksStore.getCoverUrl(currentBook)
                ? { backgroundImage: `url(${booksStore.getCoverUrl(currentBook)})` }
                : { background: bookGradient(currentBook.title) }"
            >
              <span v-if="!booksStore.getCoverUrl(currentBook)" class="cover-title-text">
                {{ currentBook.title }}
              </span>
            </div>
            <div class="hero-info">
              <NTag type="warning" size="small" :bordered="false" class="reading-tag">
                正在阅读
              </NTag>
              <h2 class="hero-title">{{ currentBook.title }}</h2>
              <p class="hero-author">{{ currentBook.author }}</p>
              <div class="hero-progress">
                <NProgress
                  type="line"
                  :percentage="pct(currentBook)"
                  :indicator-placement="'inside'"
                  :color="'#c47c3e'"
                  :rail-color="'#e4ddd2'"
                  :height="6"
                />
                <span class="progress-label">{{ pct(currentBook) }}% 完成</span>
              </div>
            </div>
            <div class="hero-actions">
              <button class="btn-primary" @click="openReader(currentBook)">继续阅读</button>
              <button class="btn-ghost" @click="openChat(currentBook)">向 AI 提问</button>
            </div>
          </div>
        </template>

        <!-- All Books Grid -->
        <div v-if="booksStore.books.length > 0">
          <div class="section-header" style="margin-top: 28px;">
            <span class="section-title">全部书籍</span>
          </div>
          <div class="book-grid">
            <div
              v-for="book in booksStore.books"
              :key="book.id"
              class="book-card"
              @click="openReader(book)"
            >
              <div
                class="book-cover"
                :style="booksStore.getCoverUrl(book)
                  ? { backgroundImage: `url(${booksStore.getCoverUrl(book)})`, backgroundSize: 'cover', backgroundPosition: 'center' }
                  : { background: bookGradient(book.title) }"
              >
                <span v-if="!booksStore.getCoverUrl(book)" class="cover-title-text small">
                  {{ book.title }}
                </span>
                <!-- Circular progress ring -->
                <svg v-if="book.status !== 'unread'" class="cover-ring" viewBox="0 0 36 36">
                  <circle class="ring-bg" cx="18" cy="18" r="15.9155" />
                  <circle
                    class="ring-fill"
                    cx="18" cy="18" r="15.9155"
                    :stroke-dasharray="`${pct(book)} 100`"
                  />
                </svg>
              </div>
              <div class="book-body">
                <p class="book-title">{{ book.title }}</p>
                <p class="book-author">{{ book.author }}</p>
                <div class="book-footer">
                  <NTag :type="STATUS_TYPE[book.status]" size="small" :bordered="false">
                    {{ STATUS_LABEL[book.status] }}
                  </NTag>
                  <span class="book-date">{{ formatDate(book.added_at) }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Empty state -->
        <NEmpty
          v-else-if="!booksStore.loading"
          description="书库空空如也，点击左侧「导入书籍」开始阅读"
          style="margin-top: 80px;"
        />
      </NSpin>
    </div>
  </div>
</template>

<style scoped>
.library-view {
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

.topbar-count {
  font-size: 13px;
  color: var(--text-muted);
  background: var(--accent-dim);
  color: var(--accent);
  padding: 2px 10px;
  border-radius: 99px;
  font-weight: 500;
}

.content {
  flex: 1;
  overflow-y: auto;
  padding: 20px 28px 28px;
}

.section-header { margin-bottom: 14px; }
.section-title { font-size: 14px; font-weight: 600; color: var(--text-muted); letter-spacing: 0.5px; }

/* ── Hero card ── */
.hero-card {
  background: var(--bg-card);
  border-radius: var(--radius);
  padding: 20px 24px;
  display: flex;
  align-items: center;
  gap: 24px;
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  margin-bottom: 8px;
  transition: box-shadow 0.2s;
}

.hero-cover {
  width: 80px;
  height: 110px;
  border-radius: 6px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background-size: cover;
  background-position: center;
  box-shadow: 3px 3px 12px rgba(0,0,0,0.25);
  overflow: hidden;
}

.hero-info { flex: 1; }
.reading-tag { margin-bottom: 8px; }
.hero-title {
  font-family: var(--font-serif);
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 4px;
  color: var(--text);
}
.hero-author { font-size: 13px; color: var(--text-muted); margin-bottom: 12px; }
.hero-progress { display: flex; align-items: center; gap: 10px; }
.progress-label { font-size: 12px; color: var(--text-muted); white-space: nowrap; }
.hero-actions { display: flex; flex-direction: column; gap: 8px; align-items: flex-end; }

/* ── Book grid ── */
.book-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 18px;
}

.book-card {
  background: var(--bg-card);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  cursor: pointer;
  transition: transform 0.18s, box-shadow 0.18s;
}

.book-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 32px rgba(26,26,46,0.14);
}

.book-cover {
  height: 150px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

.cover-title-text {
  font-family: var(--font-serif);
  font-size: 12px;
  color: rgba(255,255,255,0.9);
  text-align: center;
  padding: 12px;
  font-weight: 600;
  line-height: 1.5;
  text-shadow: 0 1px 4px rgba(0,0,0,0.4);
}

.cover-title-text.small { font-size: 11px; }

.cover-ring {
  position: absolute;
  bottom: 6px;
  right: 6px;
  width: 30px;
  height: 30px;
  transform: rotate(-90deg);
}

.ring-bg {
  fill: none;
  stroke: rgba(255,255,255,0.25);
  stroke-width: 3;
}

.ring-fill {
  fill: none;
  stroke: rgba(255,255,255,0.9);
  stroke-width: 3;
  stroke-linecap: round;
  transition: stroke-dasharray 0.5s;
}

.book-body { padding: 12px; }
.book-title { font-family: var(--font-serif); font-size: 13px; font-weight: 600; margin-bottom: 3px; line-height: 1.4; color: var(--text); }
.book-author { font-size: 11px; color: var(--text-muted); margin-bottom: 8px; }
.book-footer { display: flex; align-items: center; justify-content: space-between; }
.book-date { font-size: 10px; color: var(--text-muted); }

/* ── Buttons ── */
.btn-primary {
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 9px;
  padding: 10px 20px;
  font-size: 13px;
  font-weight: 500;
  font-family: var(--font-ui);
  cursor: pointer;
  transition: opacity 0.15s;
  white-space: nowrap;
}

.btn-primary:hover { opacity: 0.88; }

.btn-ghost {
  background: none;
  border: 1px solid var(--border);
  color: var(--text-muted);
  border-radius: 9px;
  padding: 9px 16px;
  font-size: 12px;
  font-family: var(--font-ui);
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
}

.btn-ghost:hover { border-color: var(--accent); color: var(--accent); }
</style>
