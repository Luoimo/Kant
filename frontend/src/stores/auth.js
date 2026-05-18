import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { authApi } from '@/api'
import {
  AUTH_CHANGED_EVENT,
  clearStoredAuth,
  readStoredAuth,
  setStoredAuth,
} from '@/auth/tokenManager'

export const useAuthStore = defineStore('auth', () => {
  const initial = readStoredAuth()
  const accessToken = ref(initial.accessToken)
  const refreshToken = ref(initial.refreshToken)
  const userId = ref(initial.userId)
  const role = ref(initial.role)

  const isAuthed = computed(() => !!accessToken.value)
  const isAdmin = computed(() => role.value === 'admin')

  function _syncFromStorage() {
    const v = readStoredAuth()
    accessToken.value = v.accessToken
    refreshToken.value = v.refreshToken
    userId.value = v.userId
    role.value = v.role
  }

  function _persist() {
    setStoredAuth({
      accessToken: accessToken.value,
      refreshToken: refreshToken.value,
      userId: userId.value,
      role: role.value,
    })
  }

  async function login(email, password) {
    const { data } = await authApi.login({ email, password })
    accessToken.value = data.access_token || ''
    refreshToken.value = data.refresh_token || ''
    userId.value = data.user_id || ''
    role.value = data.role || 'member'
    _persist()
    return data
  }

  async function register(email, password) {
    const { data } = await authApi.register({ email, password })
    return data
  }

  async function logout() {
    try {
      if (refreshToken.value) {
        await authApi.logout({ refresh_token: refreshToken.value })
      }
    } finally {
      clearStoredAuth()
    }
  }

  if (typeof window !== 'undefined') {
    window.addEventListener(AUTH_CHANGED_EVENT, _syncFromStorage)
  }

  return {
    accessToken,
    refreshToken,
    userId,
    role,
    isAuthed,
    isAdmin,
    login,
    register,
    logout,
  }
})
