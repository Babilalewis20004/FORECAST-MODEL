import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Binds 0.0.0.0 instead of just localhost, so the dev server is reachable
    // from outside its container (docker-compose maps 5173 to the host).
    host: true,
    proxy: {
      // In Docker, docker-compose points this at the `backend` service
      // (its own container's localhost isn't the backend's); native `npm
      // run dev` keeps the localhost default.
      '/api': process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:8000',
    },
  },
})
