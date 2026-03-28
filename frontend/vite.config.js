import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

export default defineConfig({
  plugins: [vue()],
  optimizeDeps: {
    include: ['epubjs'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/books':      { target: 'http://localhost:8000', changeOrigin: true },
      '/notes':      { target: 'http://localhost:8000', changeOrigin: true },
      '/reader':     { target: 'http://localhost:8000', changeOrigin: true },
      '/covers':     { target: 'http://localhost:8000', changeOrigin: true },
      '/ebooks':     { target: 'http://localhost:8000', changeOrigin: true },
      '/companions': { target: 'http://localhost:8000', changeOrigin: true },
      // SSE requires explicit config — disable compression so chunks are not held
      '/chat': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.setHeader('Accept-Encoding', 'identity')
          })
        },
      },
    },
  },
})
