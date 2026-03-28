<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NModal, NUpload, NUploadDragger, NButton, NSpin, useMessage } from 'naive-ui'
import { useBooksStore } from '@/stores/books'

const route = useRoute()
const router = useRouter()
const booksStore = useBooksStore()
const message = useMessage()

const showUpload = ref(false)
const uploading = ref(false)
const uploadPercent = ref(0)

onMounted(() => booksStore.fetchBooks())

const navItems = [
  { key: 'library', label: '我的书库', icon: '📚' },
  { key: 'chat', label: 'AI 对话', icon: '💬' },
  { key: 'notes', label: '我的笔记', icon: '📝' },
]

const activeNav = computed(() => route.name)

function go(name) {
  router.push({ name })
}

async function handleUpload({ file }) {
  uploading.value = true
  uploadPercent.value = 0
  try {
    await booksStore.uploadBook(file.file, (e) => {
      if (e.total) uploadPercent.value = Math.round((e.loaded / e.total) * 100)
    })
    message.success('书籍导入成功！')
    showUpload.value = false
  } catch (e) {
    message.error(`导入失败：${e.response?.data?.detail ?? e.message}`)
  } finally {
    uploading.value = false
    uploadPercent.value = 0
  }
  return false // prevent naive-ui's default upload
}
</script>

<template>
  <div class="app-layout">
    <!-- ── Sidebar ── -->
    <aside class="sidebar">
      <div class="sidebar-logo">
        <span class="logo-name">Kant</span>
        <span class="logo-sub">读书 AI 助手</span>
      </div>

      <nav class="sidebar-nav">
        <div class="nav-section-label">主菜单</div>
        <button
          v-for="item in navItems"
          :key="item.key"
          class="nav-item"
          :class="{ active: activeNav === item.key }"
          @click="go(item.key)"
        >
          <span class="nav-icon">{{ item.icon }}</span>
          <span class="nav-label">{{ item.label }}</span>
        </button>
      </nav>

      <div class="sidebar-stats">
        <div class="stat-item">
          <div class="stat-num">{{ booksStore.books.length }}</div>
          <div class="stat-label">书库</div>
        </div>
        <div class="stat-item">
          <div class="stat-num">{{ booksStore.readingBooks.length }}</div>
          <div class="stat-label">在读</div>
        </div>
        <div class="stat-item">
          <div class="stat-num">{{ booksStore.doneBooks.length }}</div>
          <div class="stat-label">已读</div>
        </div>
      </div>

      <button class="upload-btn" @click="showUpload = true">＋ 导入书籍</button>
    </aside>

    <!-- ── Main ── -->
    <main class="main-content">
      <RouterView />
    </main>

    <!-- ── Upload Modal ── -->
    <NModal v-model:show="showUpload" preset="card" title="导入 EPUB" style="width: 480px;">
      <NSpin :show="uploading">
        <NUpload
          accept=".epub"
          :max="1"
          :custom-request="handleUpload"
          :show-file-list="false"
        >
          <NUploadDragger>
            <div class="upload-dragger-body">
              <div class="upload-icon">📖</div>
              <p class="upload-hint">点击或拖拽 EPUB 文件到此区域</p>
              <p class="upload-sub">仅支持 .epub 格式</p>
              <div v-if="uploading" class="upload-progress">
                <div class="progress-bar" :style="{ width: uploadPercent + '%' }"></div>
              </div>
            </div>
          </NUploadDragger>
        </NUpload>
      </NSpin>
    </NModal>
  </div>
</template>

<style scoped>
.app-layout {
  display: flex;
  height: 100vh;
  overflow: hidden;
  background: var(--bg);
}

/* ── Sidebar ── */
.sidebar {
  width: var(--sidebar-w);
  background: var(--sidebar-bg);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  z-index: 10;
}

.sidebar-logo {
  padding: 28px 20px 24px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}

.logo-name {
  display: block;
  font-family: var(--font-serif);
  font-size: 22px;
  color: #f0ebe0;
  font-weight: 700;
  letter-spacing: 1px;
}

.logo-sub {
  display: block;
  font-size: 11px;
  color: rgba(240,235,224,0.4);
  margin-top: 3px;
  letter-spacing: 0.5px;
}

.sidebar-nav {
  padding: 16px 12px;
  flex: 1;
}

.nav-section-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: rgba(255,255,255,0.25);
  padding: 0 8px;
  margin-bottom: 8px;
  margin-top: 4px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 9px 12px;
  border-radius: 9px;
  cursor: pointer;
  color: rgba(240,235,224,0.6);
  font-size: 13.5px;
  font-weight: 400;
  font-family: var(--font-ui);
  background: none;
  border: none;
  text-align: left;
  transition: all 0.15s;
  user-select: none;
}

.nav-item:hover { background: var(--sidebar-hover); color: #f0ebe0; }
.nav-item.active { background: var(--sidebar-active); color: #f0ebe0; font-weight: 500; }
.nav-icon { font-size: 16px; width: 20px; text-align: center; }

.sidebar-stats {
  padding: 12px;
  border-top: 1px solid rgba(255,255,255,0.06);
  display: flex;
  gap: 6px;
}

.stat-item {
  flex: 1;
  background: rgba(255,255,255,0.05);
  border-radius: 10px;
  padding: 10px 4px;
  text-align: center;
}

.stat-num {
  font-size: 18px;
  font-weight: 600;
  color: #f0ebe0;
  font-family: var(--font-ui);
}

.stat-label {
  font-size: 10px;
  color: rgba(255,255,255,0.3);
  margin-top: 2px;
}

.upload-btn {
  margin: 12px;
  padding: 10px;
  background: var(--accent);
  border: none;
  border-radius: 10px;
  color: white;
  font-size: 13px;
  font-weight: 500;
  font-family: var(--font-ui);
  cursor: pointer;
  transition: opacity 0.15s;
}

.upload-btn:hover { opacity: 0.88; }

/* ── Main ── */
.main-content {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* ── Upload dragger ── */
.upload-dragger-body {
  padding: 32px 16px;
  text-align: center;
}

.upload-icon { font-size: 40px; margin-bottom: 12px; }

.upload-hint {
  font-size: 14px;
  color: var(--text);
  margin-bottom: 4px;
}

.upload-sub {
  font-size: 12px;
  color: var(--text-muted);
}

.upload-progress {
  margin-top: 16px;
  height: 4px;
  background: var(--border);
  border-radius: 99px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: var(--accent);
  border-radius: 99px;
  transition: width 0.3s;
}
</style>
