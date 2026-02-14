import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import sirv from 'sirv';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-public-under-public',
      configureServer(server) {
        server.middlewares.use('/public', sirv('public', { dev: true }));
      }
    },
    viteStaticCopy({
      targets: [
        {
          src: 'public/models/*',
          dest: 'public/models'
        }
      ]
    })
  ],
  publicDir: false,
  server: {
    port: 5173,
    strictPort: true
  }
});
