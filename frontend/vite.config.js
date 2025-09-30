import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const REPO_BASE = '/rssTrends/'

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  base: command === 'serve' ? '/' : REPO_BASE,
  plugins: [react()],
}))
