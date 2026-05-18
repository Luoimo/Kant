<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const authStore = useAuthStore()

const mode = ref('login') // 'login' | 'register'
const email = ref('')
const password = ref('')
const confirmPassword = ref('')
const loading = ref(false)
const error = ref('')
const success = ref('')

const isRegister = computed(() => mode.value === 'register')

function resetMessage() {
  error.value = ''
  success.value = ''
}

function switchMode(nextMode) {
  mode.value = nextMode
  password.value = ''
  confirmPassword.value = ''
  resetMessage()
}

async function submit() {
  if (!email.value || !password.value) return
  resetMessage()

  if (isRegister.value) {
    if (password.value.length < 8) {
      error.value = '密码至少 8 位'
      return
    }
    if (password.value !== confirmPassword.value) {
      error.value = '两次输入的密码不一致'
      return
    }
  }

  loading.value = true
  try {
    if (isRegister.value) {
      await authStore.register(email.value, password.value)
      success.value = '注册成功，正在自动登录...'
    }
    const data = await authStore.login(email.value, password.value)
    if (data.role === 'admin') router.replace({ name: 'admin' })
    else router.replace({ name: 'library' })
  } catch (e) {
    error.value = e?.response?.data?.detail || e.message
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <div class="card">
      <h1>{{ isRegister ? 'Kant Register' : 'Kant Login' }}</h1>
      <p class="sub">{{ isRegister ? '创建平台账号并自动登录' : '使用平台账号登录' }}</p>
      <input v-model.trim="email" type="email" placeholder="Email" />
      <input v-model.trim="password" type="password" placeholder="Password" @keydown.enter="submit" />
      <input
        v-if="isRegister"
        v-model.trim="confirmPassword"
        type="password"
        placeholder="Confirm Password"
        @keydown.enter="submit"
      />
      <button :disabled="loading" @click="submit">
        {{ loading ? '请稍候...' : (isRegister ? '注册并登录' : '登录') }}
      </button>
      <p v-if="error" class="error">{{ error }}</p>
      <p v-if="success" class="success">{{ success }}</p>
      <button class="link-btn" :disabled="loading" @click="switchMode(isRegister ? 'login' : 'register')">
        {{ isRegister ? '已有账号？去登录' : '没有账号？去注册' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.login-page {
  min-height: 100vh;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, #f6f2eb, #ede3d2);
}
.card {
  width: min(420px, 92vw);
  background: #fff;
  border: 1px solid #e7dbc8;
  border-radius: 14px;
  padding: 22px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
}
h1 {
  margin: 0;
  font-size: 22px;
}
.sub {
  margin: 6px 0 14px;
  color: #666;
  font-size: 13px;
}
input {
  width: 100%;
  margin-bottom: 10px;
  border: 1px solid #d8c9b1;
  border-radius: 10px;
  padding: 10px 12px;
  font-size: 14px;
}
button {
  width: 100%;
  border: none;
  border-radius: 10px;
  padding: 10px 12px;
  background: #c47c3e;
  color: #fff;
  font-weight: 600;
  cursor: pointer;
}
button:disabled {
  opacity: 0.7;
  cursor: default;
}
.error {
  margin: 10px 0 0;
  color: #b00020;
  font-size: 13px;
}
.success {
  margin: 10px 0 0;
  color: #1e7a35;
  font-size: 13px;
}
.link-btn {
  width: 100%;
  margin-top: 10px;
  border: 1px solid #d8c9b1;
  border-radius: 10px;
  padding: 10px 12px;
  background: #fff;
  color: #5d4a2a;
  font-weight: 600;
  cursor: pointer;
}
.link-btn:disabled {
  opacity: 0.7;
  cursor: default;
}
</style>
