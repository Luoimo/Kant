import { defineStore } from 'pinia'

function lsGet(key, fallback = null) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback }
  catch { return fallback }
}
function lsSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)) } catch (_) {}
}

export const useReaderStore = defineStore('reader', () => {
  function saveCfi(bookId, cfi) { lsSet(`kant_cfi_${bookId}`, cfi) }
  function loadCfi(bookId)      { return lsGet(`kant_cfi_${bookId}`, null) }

  return { saveCfi, loadCfi }
})
