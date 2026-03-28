import { createRouter, createWebHistory } from 'vue-router'
import AppLayout from '@/components/AppLayout.vue'
import LibraryView from '@/views/LibraryView.vue'
import ChatView from '@/views/ChatView.vue'
import ReaderView from '@/views/ReaderView.vue'
import NotesView from '@/views/NotesView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      component: AppLayout,
      children: [
        { path: '', redirect: '/library' },
        { path: 'library', name: 'library', component: LibraryView },
        { path: 'chat', name: 'chat', component: ChatView },
        { path: 'reader/:bookId', name: 'reader', component: ReaderView },
        { path: 'notes', name: 'notes', component: NotesView },
      ],
    },
  ],
})

export default router
