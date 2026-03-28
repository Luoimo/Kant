<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { NSelect, NSpin, NEmpty, NInput, NButton, useMessage } from 'naive-ui'
import { useNotesStore } from '@/stores/notes'
import { useBooksStore } from '@/stores/books'
import MarkdownIt from 'markdown-it'

const notesStore = useNotesStore()
const booksStore = useBooksStore()
const message = useMessage()
const md = new MarkdownIt({ breaks: true })

const selectedBook = ref(null)
const showNotePanel = ref(false)
const appendText = ref('')
const appending = ref(false)

onMounted(async () => {
  await Promise.all([
    booksStore.fetchBooks(),
    notesStore.fetchNoteBooks(),
    notesStore.fetchTimeline(),
  ])
})

watch(selectedBook, async (id) => {
  if (id) {
    await notesStore.fetchContent(id)
    showNotePanel.value = true
  } else {
    showNotePanel.value = false
    notesStore.currentContent = null
    await notesStore.fetchTimeline()
  }
})

const bookOptions = computed(() => [
  { label: '全部', value: null },
  ...notesStore.noteBooks.map((b) => ({ label: b.book_title, value: b.book_id })),
])

async function appendNote() {
  if (!appendText.value.trim() || !selectedBook.value) return
  appending.value = true
  try {
    await notesStore.appendNote(selectedBook.value, appendText.value.trim())
    appendText.value = ''
    message.success('笔记已添加')
  } catch (e) {
    message.error(`添加失败：${e.message}`)
  } finally {
    appending.value = false
  }
}

function formatDate(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleString('zh-CN', {
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}
</script>

<template>
  <div class="notes-view">
    <!-- ── Topbar ── -->
    <div class="topbar">
      <h1 class="page-title">我的笔记</h1>
      <div class="book-filter">
        <NSelect
          v-model:value="selectedBook"
          :options="bookOptions"
          placeholder="筛选书籍"
          size="small"
          style="width: 200px;"
          clearable
        />
      </div>
    </div>

    <div class="content">
      <NSpin :show="notesStore.loading" description="加载中…">

        <!-- Split: timeline + note content -->
        <div class="split-layout">

          <!-- Left: Timeline -->
          <div class="timeline-panel">
            <NEmpty
              v-if="notesStore.timeline.length === 0 && !notesStore.loading"
              description="暂无笔记记录"
              style="margin-top: 48px;"
            />

            <div v-else class="timeline">
              <div
                v-for="(entry, i) in notesStore.timeline"
                :key="i"
                class="timeline-item"
              >
                <div class="timeline-dot" :class="entry.entry_type" />
                <div class="timeline-body">
                  <div class="timeline-meta">
                    <span class="timeline-book">{{ entry.book_title }}</span>
                    <span class="timeline-date">{{ formatDate(entry.date) }}</span>
                    <span class="entry-type-tag" :class="entry.entry_type">
                      {{ entry.entry_type === 'auto' ? '自动' : '手记' }}
                    </span>
                  </div>
                  <p class="timeline-summary">{{ entry.summary }}</p>
                  <div v-if="entry.concepts?.length" class="concepts">
                    <span
                      v-for="c in entry.concepts.slice(0, 4)"
                      :key="c"
                      class="concept-tag"
                    >
                      {{ c }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Right: Note content + append -->
          <div v-if="showNotePanel && notesStore.currentContent" class="note-panel">
            <div class="note-panel-header">
              <h3 class="note-panel-title">{{ notesStore.currentContent.book_title }}</h3>
              <span class="note-panel-sub">完整笔记文档</span>
            </div>
            <div class="note-content">
              <div
                class="md-content"
                v-html="md.render(notesStore.currentContent.content || '（暂无内容）')"
              />
            </div>
            <!-- Append note form -->
            <div class="append-form">
              <NInput
                v-model:value="appendText"
                type="textarea"
                :autosize="{ minRows: 2, maxRows: 5 }"
                placeholder="追加手记…"
              />
              <NButton
                :loading="appending"
                type="primary"
                :color="'#c47c3e'"
                @click="appendNote"
                :disabled="!appendText.trim()"
                size="small"
                style="margin-top: 8px;"
              >
                添加笔记
              </NButton>
            </div>
          </div>
        </div>
      </NSpin>
    </div>
  </div>
</template>

<style scoped>
.notes-view {
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

.book-filter { margin-left: auto; }

.content {
  flex: 1;
  overflow: hidden;
  padding: 20px 28px 20px;
}

.split-layout {
  display: flex;
  gap: 20px;
  height: 100%;
  overflow: hidden;
}

/* ── Timeline ── */
.timeline-panel {
  flex: 1;
  overflow-y: auto;
  min-width: 0;
}

.timeline {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.timeline-item {
  display: flex;
  gap: 14px;
  padding: 14px 0;
  border-bottom: 1px solid var(--border);
  position: relative;
}

.timeline-item:last-child { border-bottom: none; }

.timeline-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
  margin-top: 6px;
  border: 2px solid;
}

.timeline-dot.auto { background: var(--accent); border-color: var(--accent); }
.timeline-dot.manual { background: #4a7bbf; border-color: #4a7bbf; }

.timeline-body { flex: 1; min-width: 0; }

.timeline-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 4px;
}

.timeline-book {
  font-family: var(--font-serif);
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}

.timeline-date { font-size: 11px; color: var(--text-muted); }

.entry-type-tag {
  font-size: 10px;
  padding: 1px 7px;
  border-radius: 99px;
  font-weight: 500;
}

.entry-type-tag.auto { background: var(--accent-dim); color: var(--accent); }
.entry-type-tag.manual { background: rgba(74,123,191,0.12); color: #4a7bbf; }

.timeline-summary {
  font-size: 13px;
  color: var(--text);
  line-height: 1.5;
  margin-bottom: 6px;
}

.concepts { display: flex; flex-wrap: wrap; gap: 5px; }

.concept-tag {
  background: var(--border);
  color: var(--text-muted);
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 99px;
}

/* ── Note panel ── */
.note-panel {
  width: 380px;
  flex-shrink: 0;
  background: white;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.note-panel-header {
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
}

.note-panel-title {
  font-family: var(--font-serif);
  font-size: 16px;
  font-weight: 700;
  color: var(--text);
}

.note-panel-sub { font-size: 12px; color: var(--text-muted); margin-top: 2px; display: block; }

.note-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
}

.md-content {
  font-size: 13px;
  line-height: 1.75;
  color: var(--text);
}

.md-content :deep(h2) {
  font-family: var(--font-serif);
  font-size: 14px;
  font-weight: 700;
  color: var(--accent);
  margin: 16px 0 6px;
}

.md-content :deep(p) { margin-bottom: 8px; }

.append-form {
  padding: 12px 16px;
  border-top: 1px solid var(--border);
  background: #fafaf8;
}
</style>
