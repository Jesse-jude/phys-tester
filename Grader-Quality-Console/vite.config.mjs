import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

const here = path.dirname(fileURLToPath(import.meta.url));
const consoleRoot = path.resolve(here, 'Grader-Quality-Console/artifacts/grader-quality');

export default defineConfig({
  root: consoleRoot,
  base: '/',
  cacheDir: path.resolve(here, 'node_modules/.vite'),
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(consoleRoot, 'src'),
    },
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    allowedHosts: true,
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
  },
  build: {
    outDir: path.resolve(consoleRoot, 'dist/public'),
    emptyOutDir: true,
  },
});
