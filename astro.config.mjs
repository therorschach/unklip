import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
  site: 'https://unklip.pages.dev',
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'es', 'pt', 'de', 'ja', 'hi', 'zh'],
    routing: {
      prefixDefaultLocale: false
    }
  },
  integrations: [sitemap()],
  vite: {
    plugins: [tailwindcss()],
    worker: {
      format: 'es',
      rollupOptions: {
        external: ['/vendor/transformers.web.min.js'],
      },
    },
    optimizeDeps: {
      exclude: ['@huggingface/transformers'],
    },
  },
});
