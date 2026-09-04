import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    // Dev-time proxy so the browser never needs CORS or an absolute API URL.
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: {
        // Plotly is most of the bundle and never changes between deploys —
        // give it its own long-lived chunk so app edits don't re-download it.
        manualChunks: {
          plotly: ['plotly.js-dist-min'],
          react: ['react', 'react-dom'],
        },
      },
    },
  },
})
