<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NProgress } from 'naive-ui'
import EpubReader from '@/components/EpubReader.vue'
import ReaderChat from '@/components/reader/ReaderChat.vue'
import { useBooksStore, bookGradient } from '@/stores/books'
import { useReaderStore } from '@/stores/reader'

const route   = useRoute()
const router  = useRouter()
const booksStore  = useBooksStore()
const readerStore = useReaderStore()

const bookId = computed(() => route.params.bookId)
const book   = computed(() => booksStore.books.find((b) => b.id === bookId.value))

const epubUrl = computed(() => {
  if (!book.value?.source) return null
  const filename = book.value.source.split(/[/\\]/).pop()
  return `/ebooks/${encodeURIComponent(filename)}`
})

const savedCfi       = computed(() => readerStore.loadCfi(bookId.value))
const currentChapter = ref('')
const selectedText   = ref(null)   // { text, cfi } | null
const epubReader     = ref(null)

function onChapterChange(chapter) {
  currentChapter.value = chapter
}

function onCfiChange(cfi) {
  if (bookId.value) readerStore.saveCfi(bookId.value, cfi)
}

function onTextSelected(sel) {
  selectedText.value = sel?.text ?? null
}

function navigateCitation(citation) {
  epubReader.value?.goToCitation(citation)
}

onMounted(async () => {
  await booksStore.fetchBooks()
})
</script>

<template>
  <div class="reader-view">
    <!-- ── Topbar ── -->
    <div class="topbar">
      <button class="back-btn" @click="router.push({ name: 'library' })">← 书库</button>

      <div class="book-meta" v-if="book">
        <div
          class="mini-cover"
          :style="booksStore.getCoverUrl(book)
            ? { backgroundImage: `url(${booksStore.getCoverUrl(book)})`, backgroundSize: 'cover', backgroundPosition: 'center' }
            : { background: bookGradient(book.title) }"
        />
        <div class="meta-text">
          <h1 class="book-title-top">{{ book.title }}</h1>
          <p class="book-author-top">{{ book.author }}</p>
        </div>
      </div>

      <div class="chapter-pill" v-if="currentChapter">{{ currentChapter }}</div>

      <div class="progress-area" v-if="book && book.progress > 0">
        <NProgress
          type="line"
          :percentage="Math.round(book.progress * 100)"
          :color="'#c47c3e'"
          :rail-color="'#e4ddd2'"
          :height="4"
          :show-indicator="false"
          style="width: 80px;"
        />
        <span class="progress-text">{{ Math.round(book.progress * 100) }}%</span>
      </div>
    </div>

    <!-- ── Split body ── -->
    <div class="split-body">

      <!-- Left: EPUB reader -->
      <div class="epub-panel">
        <EpubReader
          v-if="epubUrl"
          ref="epubReader"
          :url="epubUrl"
          :initial-cfi="savedCfi"
          @chapter-change="onChapterChange"
          @cfi-change="onCfiChange"
          @text-selected="onTextSelected"
        />
        <div v-else class="epub-placeholder">书籍文件不可用</div>
      </div>

      <!-- Right: AI chat -->
      <div class="side-panel">
        <ReaderChat
          :book-id="bookId"
          :book-title="book?.title || ''"
          :current-chapter="currentChapter"
          :selected-text="selectedText"
          :navigate-citation="navigateCitation"
          @clear-selection="selectedText = null"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.reader-view {
  display: flex; flex-direction: column; height: 100%; overflow: hidden;
}

/* ── Topbar ── */
.topbar {
  padding: 10px 16px; display: flex; align-items: center; gap: 12px;
  flex-shrink: 0; border-bottom: 1px solid var(--border); background: white;
  min-height: 52px;
}

.back-btn {
  background: none; border: none; color: var(--text-muted); font-size: 13px;
  cursor: pointer; font-family: var(--font-ui); padding: 4px 0;
  white-space: nowrap; transition: color 0.15s; flex-shrink: 0;
}
.back-btn:hover { color: var(--accent); }

.book-meta { display: flex; align-items: center; gap: 9px; min-width: 0; }

.mini-cover {
  width: 26px; height: 36px; border-radius: 3px; flex-shrink: 0;
  box-shadow: 1px 2px 6px rgba(0,0,0,0.2);
}

.meta-text { min-width: 0; }

.book-title-top {
  font-family: var(--font-serif); font-size: 13.5px; font-weight: 700;
  color: var(--text); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; max-width: 180px;
}

.book-author-top { font-size: 11px; color: var(--text-muted); }

.chapter-pill {
  font-size: 11px; color: var(--text-muted); background: var(--bg);
  border: 1px solid var(--border); border-radius: 99px; padding: 3px 9px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 160px;
  flex-shrink: 1;
}

.progress-area {
  margin-left: auto; display: flex; align-items: center; gap: 7px; flex-shrink: 0;
}
.progress-text { font-size: 11px; color: var(--text-muted); }

/* ── Split body ── */
.split-body { flex: 1; display: flex; overflow: hidden; }

/* ── EPUB panel ── */
.epub-panel { flex: 1; min-width: 0; overflow: hidden; }

.epub-placeholder {
  height: 100%; display: flex; align-items: center; justify-content: center;
  font-size: 13px; color: var(--text-muted); background: #fafaf8;
}

/* ── Side panel ── */
.side-panel {
  width: 300px; flex-shrink: 0; border-left: 1px solid var(--border);
  display: flex; flex-direction: column; overflow: hidden;
}
</style>
