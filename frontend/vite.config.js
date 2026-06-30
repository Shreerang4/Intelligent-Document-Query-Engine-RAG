import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';


export default defineConfig({
  plugins: [react()],   // keep whatever plugins are already there
  server: {
    proxy: {
      '/hackrx': 'http://localhost:8000',
      '/history': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})