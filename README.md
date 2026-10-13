# Unklip — 100% Free, Private, Full-Resolution Background Remover

> **A high-performance, viral, open-source alternative to remove.bg built with Astro, Transformers.js, and WebGPU.**

With `remove.bg` transitioning to paid Canva integrations and limiting free cutouts to 0.25 MP previews, **Unklip** delivers state-of-the-art neural background removal running **100% locally in the user's browser** with **$0 in server inference costs**.

---

## ✨ Key Features

- **⚡ Hardware Accelerated (WebGPU):** Processes images in ~0.3–0.8s on the device's GPU (Apple Silicon, NVIDIA RTX, AMD, Intel) with automatic fallback to multi-threaded WebAssembly (WASM).
- **🎯 100% Native Resolution Retained:** Unlike remove.bg (which caps free downloads to 500×500 px), Unklip applies the neural alpha matte directly to the original high-resolution pixels (Full HD, 4K, 8K).
- **🔒 100% Private (On-Device):** Photos never leave the user's browser. Zero cloud uploads, zero tracking, zero logs. Safe for confidential corporate, medical, and personal images.
- **🎨 Interactive Studio Suite:**
  - **Before / After Split Slider** for real-time edge comparison.
  - **Studio Backgrounds:** Instant pure white (`#FFFFFF`) for Amazon/Shopify/eBay, solid black, studio gray, and custom color picker.
  - **Background Blur:** Keep subject crisp while blurring the original background.
  - **1-Click Copy to Clipboard:** Copy transparent PNGs directly into Figma, Photoshop, Canva, or Slack.
- **🖼️ Instant One-Click Samples:** Pre-bundled samples (Portrait with fine hair, Product sneaker, Pet with fluffy fur, Sports athlete).
- **📋 Zero Sign-Up or Subscriptions:** No account required, no credits, no daily quotas.

---

## 🛠️ Tech Stack & Architecture

- **Framework:** [Astro 7](https://astro.build/) (Static Site Generation, sub-second load times)
- **Styling:** [Tailwind CSS v4](https://tailwindcss.com/)
- **AI Engine:** [@huggingface/transformers](https://github.com/huggingface/transformers.js) (`briaai/RMBG-1.4`)
- **Execution:** Dedicated Web Worker + WebGPU (`device: 'webgpu'`) with graceful WASM fallback
- **Cross-Origin Isolation:** COOP/COEP headers configured for Cloudflare Pages (`public/_headers`) and Vercel (`vercel.json`)

---

## 🚀 Getting Started

### Prerequisites

- Node.js `20.x` or `22.x`
- npm (or pnpm / yarn)

### Installation

```bash
cd /home/sakib/Documents/Projects/unklip
npm install
```

### Run Locally

```bash
npm run dev
```

Open [http://localhost:4321](http://localhost:4321) in your browser.

### Production Build

```bash
npm run build
npm run preview
```

Output is generated into `dist/`.

---

## 🚢 Deploying to Production (Zero Server Bills)

Because Unklip runs client-side via WebGPU and Transformers.js, hosting costs **$0/month**.

### Option 1: Cloudflare Pages
1. Push this repository to GitHub or GitLab.
2. In the Cloudflare Dashboard, go to **Workers & Pages** → **Create application** → **Pages**.
3. Connect your repository:
   - **Framework preset:** `Astro`
   - **Build command:** `npm run build`
   - **Build output directory:** `dist`
4. The included `public/_headers` file will automatically configure `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy: credentialless`.

### Option 2: Vercel
1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` in the project directory, or import via the Vercel dashboard.
3. The included `vercel.json` applies the required headers automatically.

---

## ⚖️ Unklip vs remove.bg

| Feature | Unklip | remove.bg (Canva) |
| :--- | :--- | :--- |
| **Price** | **100% Free Forever ($0)** | Up to $0.20 – $1.99 / image |
| **Free Resolution** | **Full Native 4K / 8K (100%)** | Downscaled to 0.25 MP |
| **Privacy** | **100% On-Device (Browser)** | Uploaded to remote servers |
| **Daily Limits** | **Unlimited** | 1 free preview credit |
| **Account Required** | **None** | Required for HD/credits |
| **Studio Tools** | **White, Blur, Color, Copy** | Locked behind Canva Pro |
