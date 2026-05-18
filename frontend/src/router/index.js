import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import AppLayout from '@/components/AppLayout.vue'
import LibraryView from '@/views/LibraryView.vue'
import ReaderView from '@/views/ReaderView.vue'
import AdminView from '@/views/AdminView.vue'
import LoginView from '@/views/LoginView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', name: 'login', component: LoginView },
    {
      path: '/',
      component: AppLayout,
      meta: { requiresAuth: true },
      children: [
        { path: '', redirect: '/library' },
        { path: 'library', name: 'library', component: LibraryView },
        { path: 'reader/:bookId', name: 'reader', component: ReaderView },
        { path: 'admin', name: 'admin', component: AdminView, meta: { requiresAdmin: true } },
      ],
    },
  ],
})

router.beforeEach((to, _from, next) => {
  const auth = useAuthStore()
  if (to.meta.requiresAuth && !auth.isAuthed) {
    next({ name: 'login' })
    return
  }
  if (to.meta.requiresAdmin && !auth.isAdmin) {
    next({ name: 'library' })
    return
  }
  if (to.name === 'login' && auth.isAuthed) {
    next(auth.isAdmin ? { name: 'admin' } : { name: 'library' })
    return
  }
  next()
})

export default router
