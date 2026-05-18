<script setup>
import { onMounted, ref } from 'vue'
import { adminApi } from '@/api'

const users = ref([])
const loading = ref(false)
const error = ref('')

async function loadUsers() {
  loading.value = true
  error.value = ''
  try {
    const { data } = await adminApi.users()
    users.value = data
  } catch (e) {
    error.value = e?.response?.data?.detail || e.message
  } finally {
    loading.value = false
  }
}

onMounted(loadUsers)
</script>

<template>
  <div class="admin-view">
    <h1>Admin Console</h1>
    <p class="muted">只读用户视图</p>

    <div v-if="loading">Loading...</div>
    <div v-else-if="error" class="error">{{ error }}</div>
    <table v-else class="table">
      <thead>
        <tr>
          <th>User ID</th>
          <th>Email</th>
          <th>Role</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="u in users" :key="u.user_id">
          <td>{{ u.user_id }}</td>
          <td>{{ u.email }}</td>
          <td>{{ u.role }}</td>
          <td>{{ u.status }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.admin-view {
  padding: 20px;
}
.muted {
  color: #666;
}
.error {
  color: #b00020;
}
.table {
  width: 100%;
  border-collapse: collapse;
}
.table th,
.table td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left;
}
</style>
