import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

const htmlBypass = (req) => {
  if (req.headers.accept && req.headers.accept.includes('text/html')) {
    return '/index.html'
  }
}

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
      '/books':      { target: 'http://localhost:8000', changeOrigin: true, bypass: htmlBypass },
      '/notes':      { target: 'http://localhost:8000', changeOrigin: true, bypass: htmlBypass },
      '/reader':     { target: 'http://localhost:8000', changeOrigin: true, bypass: htmlBypass },
      '/covers':     { target: 'http://localhost:8000', changeOrigin: true },
      '/ebooks':     { target: 'http://localhost:8000', changeOrigin: true },
      '/companions': { target: 'http://localhost:8000', changeOrigin: true },
      // SSE requires explicit config — disable compression so chunks are not held
      '/chat': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass: htmlBypass,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.setHeader('Accept-Encoding', 'identity')
          })
        },
      },
    },
  },
})