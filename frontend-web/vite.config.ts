import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Docker container içinde çalışırken dışarıdan (host) erişilebilir
    // olması için 0.0.0.0'da dinlemesi gerekiyor.
    host: true,
    watch: {
      // Windows host -> Docker bind mount senaryosunda dosya sistemi
      // event'leri (inotify) çoğu zaman container'a ulaşmıyor; bu yüzden
      // hot-reload'ın çalışması için polling'e geçiyoruz.
      usePolling: true,
    },
  },
})
