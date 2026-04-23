import { createRouter, createWebHistory } from 'vue-router'
import AppLayout from '@/components/AppLayout.vue'
import LibraryView from '@/views/LibraryView.vue'
import ReaderView from '@/views/ReaderView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      component: AppLayout,
      children: [
        { path: '', redirect: '/library' },
        { path: 'library', name: 'library', component: LibraryView },
        { path: 'reader/:bookId', name: 'reader', component: ReaderView },
      ],
    },
  ],
})

export default router
