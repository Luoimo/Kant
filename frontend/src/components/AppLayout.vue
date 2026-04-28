<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { NModal, NUpload, NUploadDragger, NButton, NSpin, useMessage } from 'naive-ui'
import { useBooksStore } from '@/stores/books'
import { SUPPORTED_LOCALES, persistLocale } from '@/i18n'

const route = useRoute()
const router = useRouter()
const booksStore = useBooksStore()
const message = useMessage()
const { t, locale } = useI18n()

const showUpload = ref(false)
const uploading = ref(false)
const uploadPercent = ref(0)

onMounted(() => booksStore.fetchBooks())

const navItems = computed(() => [
  { key: 'library', label: t('sidebar.library'), icon: '📚' },
])

const activeNav = computed(() => route.name)

function go(name) {
  router.push({ name })
}

function switchLocale(value) {
  locale.value = value
  persistLocale(value)
}

async function handleUpload({ file }) {
  uploading.value = true
  uploadPercent.value = 0
  try {
    await booksStore.uploadBook(file.file, (e) => {
      if (e.total) uploadPercent.value = Math.round((e.loaded / e.total) * 100)
    })
    message.success(t('upload.success'))
    showUpload.value = false
  } catch (e) {
    message.error(t('upload.failed', { msg: e.response?.data?.detail ?? e.message }))
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
        <span class="logo-sub">{{ t('sidebar.logoSub') }}</span>
      </div>

      <nav class="sidebar-nav">
        <div class="nav-section-label">{{ t('sidebar.mainMenu') }}</div>
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
          <div class="stat-label">{{ t('sidebar.statsLibrary') }}</div>
        </div>
        <div class="stat-item">
          <div class="stat-num">{{ booksStore.readingBooks.length }}</div>
          <div class="stat-label">{{ t('sidebar.statsReading') }}</div>
        </div>
        <div class="stat-item">
          <div class="stat-num">{{ booksStore.doneBooks.length }}</div>
          <div class="stat-label">{{ t('sidebar.statsDone') }}</div>
        </div>
      </div>

      <div class="lang-switch" role="group" :aria-label="t('common.language')">
        <button
          v-for="opt in SUPPORTED_LOCALES"
          :key="opt.value"
          class="lang-btn"
          :class="{ active: locale === opt.value }"
          @click="switchLocale(opt.value)"
        >{{ opt.label }}</button>
      </div>

      <button class="upload-btn" @click="showUpload = true">{{ t('sidebar.uploadBook') }}</button>
    </aside>

    <!-- ── Main ── -->
    <main class="main-content">
      <RouterView />
    </main>

    <!-- ── Upload Modal ── -->
    <NModal v-model:show="showUpload" preset="card" :title="t('upload.title')" style="width: 480px;">
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
              <p class="upload-hint">{{ t('upload.hint') }}</p>
              <p class="upload-sub">{{ t('upload.sub') }}</p>
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

/* ── Language switch ── */
.lang-switch {
  margin: 10px 12px 0;
  display: flex;
  gap: 4px;
  padding: 3px;
  background: rgba(255,255,255,0.05);
  border-radius: 8px;
}
.lang-btn {
  flex: 1;
  background: none;
  border: none;
  color: rgba(240,235,224,0.55);
  font-size: 11px;
  font-family: var(--font-ui);
  font-weight: 600;
  letter-spacing: 0.5px;
  padding: 5px 0;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s;
}
.lang-btn:hover { color: #f0ebe0; }
.lang-btn.active {
  background: var(--accent);
  color: white;
}

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
