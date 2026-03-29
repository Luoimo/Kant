<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'

const props = defineProps({
  url:        { type: String,  required: true },
  initialCfi: { type: String,  default: null },
})

const emit = defineEmits(['chapter-change', 'cfi-change', 'text-selected', 'ready', 'error'])

const containerRef = ref(null)
const loading      = ref(true)
const errorMsg     = ref(null)
const currentChapter = ref('')
const toc          = ref([])
const showToc      = ref(false)
const atStart      = ref(true)
const atEnd        = ref(false)

let book = null
let rendition = null
let resizeObserver = null
let timeoutId = null
let pendingTocHref = null    // set by goToHref, consumed once by resolveChapter

async function init() {
  if (!containerRef.value) return

  // cleanup
  clearTimeout(timeoutId)
  resizeObserver?.disconnect(); resizeObserver = null
  try { rendition?.destroy() } catch (_) {}; rendition = null
  try { book?.destroy()      } catch (_) {}; book = null

  loading.value = true
  errorMsg.value = null
  toc.value = []
  currentChapter.value = ''
  atStart.value = true
  atEnd.value = false
  pendingTocHref = null

  let ePub
  try {
    const mod = await import('epubjs')
    ePub = mod.default ?? mod.ePub ?? mod
  } catch (e) {
    errorMsg.value = `无法加载 epubjs：${e.message}`
    loading.value = false
    return
  }

  try {
    const el = containerRef.value
    const w = el.clientWidth  || el.offsetWidth  || 640
    const h = el.clientHeight || el.offsetHeight || 520

    book = ePub(props.url, { openAs: 'epub' })

    rendition = book.renderTo(el, {
      width: w, height: h,
      spread: 'none', flow: 'paginated',
      allowScriptedContent: false,
    })

    rendition.themes.register('kant', {
      'body': {
        'font-family': "'Noto Serif SC', Georgia, serif !important",
        'font-size':   '15px !important',
        'line-height': '1.9 !important',
        'color':       '#1a1a2e !important',
        'padding':     '0 20px !important',
        'background':  '#fffdf8 !important',
      },
      'p': {
        'margin-bottom': '1em !important',
        'text-align':    'justify !important',
      },
    })
    rendition.themes.select('kant')

    // TOC
    book.loaded.navigation
      .then((nav) => {
        toc.value = flattenToc(nav.toc ?? [])
        emit('ready', { toc: toc.value })
      })
      .catch(() => {})

    // Relocated → CFI + chapter
    rendition.on('relocated', (loc) => {
      atStart.value = !!loc.atStart
      atEnd.value   = !!loc.atEnd
      resolveChapter(loc)   // async, fire-and-forget
      if (loc.start?.cfi) emit('cfi-change', loc.start.cfi)
    })

    // Text selection inside EPUB iframe
    rendition.on('selected', (cfiRange, contents) => {
      const text = contents?.window?.getSelection()?.toString()?.trim()
      if (text) emit('text-selected', { text, cfi: cfiRange })
    })

    // First render → remove spinner
    rendition.once('rendered', () => {
      loading.value = false
      clearTimeout(timeoutId)
    })

    // Timeout fallback
    timeoutId = setTimeout(() => {
      if (loading.value) {
        errorMsg.value = '加载超时，请确认书籍文件可访问'
        loading.value = false
      }
    }, 10000)

    // Display from saved position or beginning
    await rendition.display(props.initialCfi ?? undefined).catch((e) => {
      errorMsg.value = `渲染失败：${e.message}`
      loading.value = false
    })

    // ResizeObserver
    resizeObserver = new ResizeObserver(() => {
      const nw = el.clientWidth, nh = el.clientHeight
      if (nw > 0 && nh > 0) rendition?.resize(nw, nh)
    })
    resizeObserver.observe(el)

  } catch (e) {
    console.error('[EpubReader]', e)
    errorMsg.value = `初始化失败：${e.message}`
    loading.value = false
    emit('error', e)
  }
}

function flattenToc(items, depth = 0) {
  const out = []
  for (const item of items) {
    out.push({ label: (item.label ?? '').trim(), href: item.href ?? '', depth })
    if (item.subitems?.length) out.push(...flattenToc(item.subitems, depth + 1))
  }
  return out
}

async function resolveChapter(loc) {
  if (!toc.value.length) return

  const explicitHref = pendingTocHref
  pendingTocHref = null

  const hrefToMatch = explicitHref ?? loc?.start?.href
  if (!hrefToMatch) return

  const [matchBase, matchAnchor] = splitHref(hrefToMatch)

  // 1. TOC click → exact anchor match
  if (matchAnchor) {
    const idx = toc.value.findIndex((t) => {
      const [tb, ta] = splitHref(t.href)
      return ta === matchAnchor && baseMatch(matchBase, tb)
    })
    if (idx !== -1) return setChapter(buildBreadcrumb(idx))
  }

  // TOC entries for this file
  const fileEntries = toc.value
    .map((t, i) => ({ ...t, idx: i }))
    .filter(({ href }) => baseMatch(matchBase, splitHref(href)[0]))

  if (!fileEntries.length) return
  if (fileEntries.length === 1) return setChapter(buildBreadcrumb(fileEntries[0].idx))

  // 2. Multiple sections in same file:
  //    Get the DOM node at loc.start.cfi, then find the last anchor
  //    that precedes it in document order via compareDocumentPosition.
  const doc = rendition?.getContents?.()?.[0]?.document
  if (!doc) return setChapter(buildBreadcrumb(fileEntries[0].idx))

  let refNode = null
  try {
    if (loc.start?.cfi && book) {
      const range = await book.getRange(loc.start.cfi)
      refNode = range?.startContainer ?? null
    }
  } catch (_) {}

  // Fallback refNode: element at the visible top-left of the iframe
  if (!refNode) {
    const win = doc.defaultView
    refNode = doc.elementFromPoint(4, (win?.innerHeight ?? 200) / 2)
  }

  let bestIdx = fileEntries[0].idx
  for (const entry of fileEntries) {
    const [, anchor] = splitHref(entry.href)
    if (!anchor) { bestIdx = entry.idx; continue }
    const el = doc.getElementById(anchor) ?? doc.querySelector(`[name="${anchor}"]`)
    if (!el || !refNode) continue
    // DOCUMENT_POSITION_PRECEDING (2): refNode precedes el → el is still ahead, skip
    if (!(el.compareDocumentPosition(refNode) & 2)) bestIdx = entry.idx
  }

  setChapter(buildBreadcrumb(bestIdx))
}

// Walk backwards from matchedIndex to collect one ancestor per depth level,
// then join them into a breadcrumb string: "Chapter 1 › Section 1.2"
function buildBreadcrumb(index) {
  const item = toc.value[index]
  if (item.depth === 0) return item.label

  const ancestors = []
  let need = item.depth - 1
  for (let i = index - 1; i >= 0 && need >= 0; i--) {
    if (toc.value[i].depth === need) {
      ancestors.unshift(toc.value[i].label)
      need--
    }
  }
  return [...ancestors, item.label].join(' › ')
}

function splitHref(href) {
  const idx = href.indexOf('#')
  return idx === -1 ? [href, null] : [href.slice(0, idx), href.slice(idx + 1)]
}

function baseMatch(a, b) {
  return a === b || a.endsWith('/' + b) || b.endsWith('/' + a)
}

function setChapter(label) {
  if (currentChapter.value !== label) {
    currentChapter.value = label
    emit('chapter-change', label)
  }
}

function prev()           { rendition?.prev() }
function next()           { rendition?.next() }
function goToHref(href) {
  pendingTocHref = href   // resolveChapter will use this for exact anchor matching
  rendition?.display(href)
  showToc.value = false
}

async function goToCitation(citation) {
  if (!rendition || !book) return

  let href = null

  // 1. Use spine index (most reliable — exact position from RAG metadata)
  const indices = citation.section_indices
  if (Array.isArray(indices) && indices.length > 0) {
    const item = book.spine.get(indices[0])
    if (item?.href) href = item.href
  }

  // 2. Fallback: fuzzy-match chapter/section title against TOC
  if (!href) {
    const target = citation.section_title || citation.chapter_title
    if (target && toc.value.length) {
      const entry = toc.value.find(t => t.label === target)
        ?? toc.value.find(t => t.label.includes(target) || target.includes(t.label))
      if (entry) href = entry.href
    }
  }

  if (!href) return

  pendingTocHref = href
  await rendition.display(href)

  if (citation.snippet) {
    // Allow iframe to fully settle before searching
    await new Promise(r => setTimeout(r, 250))
    _highlightSnippet(citation.snippet)
  }
}

function _highlightSnippet(snippet) {
  const contents = rendition?.getContents?.()
  if (!contents?.length) return
  const doc = contents[0]?.document
  const win = doc?.defaultView
  if (!doc || !win) return

  // Inject highlight styles once
  if (!doc.getElementById('__kant_hl_style')) {
    const s = doc.createElement('style')
    s.id = '__kant_hl_style'
    s.textContent = `
      .kant-hl { background: rgba(255,200,0,0.55) !important; border-radius:2px; }
      .kant-hl.kant-hl-fade { background: transparent !important; transition: background 1.5s ease-out; }
    `
    doc.head.appendChild(s)
  }

  // Remove any previous highlights
  doc.querySelectorAll('.kant-hl').forEach(el => {
    const p = el.parentNode
    if (p) { el.replaceWith(...el.childNodes); p.normalize() }
  })

  const searchStr = snippet.replace(/…$/, '').slice(0, 60).trim()
  if (!win.find?.(searchStr)) return

  const sel = win.getSelection()
  if (!sel?.rangeCount) return

  try {
    const range = sel.getRangeAt(0)
    const span = doc.createElement('span')
    span.className = 'kant-hl'
    try {
      range.surroundContents(span)
    } catch {
      const frag = range.extractContents()
      span.appendChild(frag)
      range.insertNode(span)
    }
    sel.removeAllRanges()
    span.scrollIntoView({ behavior: 'smooth', block: 'center' })

    setTimeout(() => {
      span.classList.add('kant-hl-fade')
      setTimeout(() => {
        const p = span.parentNode
        if (p) { span.replaceWith(...span.childNodes); p.normalize() }
      }, 1500)
    }, 2500)
  } catch (e) {
    console.warn('[EpubReader] highlight failed:', e)
  }
}

onMounted(() => nextTick(() => init()))
onUnmounted(() => {
  clearTimeout(timeoutId)
  resizeObserver?.disconnect()
  try { rendition?.destroy() } catch (_) {}
  try { book?.destroy()      } catch (_) {}
})
watch(() => props.url, () => nextTick(() => init()))

defineExpose({ prev, next, goToHref, goToCitation, currentChapter, toc })
</script>

<template>
  <div class="epub-reader">
    <!-- Header -->
    <div class="reader-bar">
      <button class="toc-btn" @click="showToc = !showToc" title="目录">☰</button>
      <span class="chapter-label" :title="currentChapter">
        {{ loading ? '加载中…' : (currentChapter || '　') }}
      </span>
    </div>

    <!-- TOC drawer -->
    <Transition name="toc-slide">
      <div v-if="showToc" class="toc-panel">
        <div class="toc-header">
          <span>目录</span>
          <button class="toc-close" @click="showToc = false">✕</button>
        </div>
        <div class="toc-list">
          <button
            v-for="item in toc"
            :key="item.href + item.label"
            class="toc-item"
            :style="{ paddingLeft: `${12 + item.depth * 14}px` }"
            @click="goToHref(item.href)"
          >{{ item.label }}</button>
          <div v-if="toc.length === 0" class="toc-empty">暂无目录</div>
        </div>
      </div>
    </Transition>

    <!-- Render area -->
    <div class="render-wrap">
      <div v-if="loading && !errorMsg" class="epub-loading">
        <div class="spinner"></div>
        <p class="loading-tip">正在加载书籍内容…</p>
      </div>
      <div v-if="errorMsg" class="epub-error">
        <p>⚠️</p>
        <p class="error-msg">{{ errorMsg }}</p>
        <button class="retry-btn" @click="init">重新加载</button>
      </div>
      <div
        ref="containerRef"
        class="epub-container"
        :style="{ visibility: (loading || errorMsg) ? 'hidden' : 'visible' }"
      />
    </div>

    <!-- Nav controls -->
    <div class="nav-controls">
      <button class="nav-btn" :disabled="atStart || loading" @click="prev">‹ 上页</button>
      <span class="nav-hint">← → 键翻页</span>
      <button class="nav-btn" :disabled="atEnd   || loading" @click="next">下页 ›</button>
    </div>
  </div>
</template>

<style scoped>
.epub-reader {
  display: flex; flex-direction: column; height: 100%;
  background: #fffdf8; border-right: 1px solid var(--border);
  position: relative; overflow: hidden;
}
.reader-bar {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 14px; background: white;
  border-bottom: 1px solid var(--border); flex-shrink: 0;
}
.toc-btn {
  background: none; border: none; font-size: 18px; cursor: pointer;
  color: var(--text-muted); padding: 2px 6px; border-radius: 6px; line-height: 1;
  transition: all 0.15s;
}
.toc-btn:hover { background: var(--accent-dim); color: var(--accent); }
.chapter-label {
  font-family: var(--font-serif); font-size: 13px; color: var(--text-muted);
  flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
/* TOC */
.toc-panel {
  position: absolute; top: 40px; left: 0; width: 260px;
  max-height: calc(100% - 92px); background: white;
  border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);
  border-bottom-right-radius: 12px; z-index: 20;
  display: flex; flex-direction: column;
  box-shadow: 4px 4px 16px rgba(26,26,46,0.1);
}
.toc-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 14px; border-bottom: 1px solid var(--border);
  font-size: 13px; font-weight: 600; color: var(--text); flex-shrink: 0;
}
.toc-close { background: none; border: none; cursor: pointer; color: var(--text-muted); font-size: 14px; }
.toc-list { overflow-y: auto; flex: 1; padding: 6px 0; }
.toc-item {
  display: block; width: 100%; text-align: left; background: none; border: none;
  padding: 8px 12px; font-size: 13px; font-family: var(--font-serif);
  color: var(--text); cursor: pointer; transition: all 0.12s; line-height: 1.4;
}
.toc-item:hover { background: var(--accent-dim); color: var(--accent); }
.toc-empty { padding: 16px; font-size: 13px; color: var(--text-muted); text-align: center; }
/* Render */
.render-wrap { flex: 1; overflow: hidden; position: relative; }
.epub-container { width: 100%; height: 100%; }
.epub-loading {
  position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 14px; background: #fffdf8; z-index: 5;
}
.spinner {
  width: 36px; height: 36px; border: 3px solid var(--border);
  border-top-color: var(--accent); border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-tip { font-size: 13px; color: var(--text-muted); font-family: var(--font-ui); }
.epub-error {
  position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 8px;
  background: #fffdf8; z-index: 5; padding: 24px;
  font-size: 13px; color: var(--text-muted); text-align: center;
}
.error-msg { max-width: 300px; }
.retry-btn {
  margin-top: 6px; background: var(--accent); border: none; border-radius: 8px;
  padding: 8px 20px; color: white; font-size: 13px; cursor: pointer; font-family: var(--font-ui);
}
/* Nav */
.nav-controls {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 20px; background: white; border-top: 1px solid var(--border); flex-shrink: 0;
}
.nav-btn {
  background: none; border: 1px solid var(--border); border-radius: 8px;
  padding: 5px 16px; font-size: 13px; font-family: var(--font-ui);
  color: var(--text); cursor: pointer; transition: all 0.15s;
}
.nav-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }
.nav-btn:disabled { opacity: 0.3; cursor: not-allowed; }
.nav-hint { font-size: 11px; color: var(--text-muted); }
/* Transition */
.toc-slide-enter-active, .toc-slide-leave-active { transition: opacity 0.16s ease, transform 0.16s ease; }
.toc-slide-enter-from, .toc-slide-leave-to { opacity: 0; transform: translateX(-10px); }
</style>
