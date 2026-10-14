import { AutoModel, RawImage, Tensor, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3';

// Configure environment for in-browser execution
env.allowLocalModels = false;

const MODEL_ID = 'briaai/RMBG-1.4';

let modelPromise: Promise<any> | null = null;
let model: any = null;
let activeDevice: 'webgpu' | 'wasm' = 'wasm';

async function initModel() {
  if (model) return model;
  if (modelPromise) return modelPromise;

  modelPromise = (async () => {
    let device: 'webgpu' | 'wasm' = 'wasm';

    // Check WebGPU availability
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const adapter = await (navigator as any).gpu?.requestAdapter();
        if (adapter) {
          device = 'webgpu';
        }
      } catch (e) {
        console.warn('WebGPU request failed, falling back to WASM:', e);
        device = 'wasm';
      }
    }

    activeDevice = device;
    self.postMessage({ type: 'device_info', device: activeDevice });

    const progressCallback = (progressInfo: any) => {
      self.postMessage({
        type: 'progress',
        data: progressInfo,
      });
    };

    try {
      model = await AutoModel.from_pretrained(MODEL_ID, {
        device,
        dtype: 'fp32',
        progress_callback: progressCallback,
      });
      return model;
    } catch (gpuErr) {
      if (device === 'webgpu') {
        console.warn('WebGPU model load failed, retrying with WASM fallback:', gpuErr);
        activeDevice = 'wasm';
        self.postMessage({ type: 'device_info', device: activeDevice });

        model = await AutoModel.from_pretrained(MODEL_ID, {
          device: 'wasm',
          dtype: 'fp32',
          progress_callback: progressCallback,
        });
        return model;
      }
      throw gpuErr;
    }
  })();

  return modelPromise;
}

self.addEventListener('message', async (event: MessageEvent) => {
  const { action, id, imageBlob, threshold = 0.5 } = event.data;

  if (action === 'get_device') {
    let hasGpu = false;
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const adapter = await (navigator as any).gpu?.requestAdapter();
        hasGpu = !!adapter;
      } catch {}
    }
    self.postMessage({ type: 'device_info', device: hasGpu ? 'webgpu' : 'wasm' });
    return;
  }

  if (action === 'warmup') {
    try {
      await initModel();
      self.postMessage({ type: 'warmup_complete', id, device: activeDevice });
    } catch (error: any) {
      self.postMessage({ type: 'warmup_error', id, error: error?.message || String(error) });
    }
    return;
  }

  if (action === 'remove_bg') {
    const startTime = performance.now();
    try {
      self.postMessage({
        type: 'step',
        id,
        stage: 'init',
        percent: 12,
        title: 'Initializing AI Engine',
        detail: 'Loading RMBG weights into browser memory...',
      });
      const currentModel = await initModel();

      self.postMessage({
        type: 'step',
        id,
        stage: 'decoding',
        percent: 28,
        title: 'Decoding Image',
        detail: 'Reading pixel channels and geometry...',
      });

      // Load original image
      const rawImg = await RawImage.fromBlob(imageBlob);
      const originalWidth = rawImg.width;
      const originalHeight = rawImg.height;

      // 1. Resize to 1024x1024
      self.postMessage({
        type: 'step',
        id,
        stage: 'tensor',
        percent: 42,
        title: 'Preparing Neural Tensor',
        detail: 'Rescaling to 1024×1024 RGB input...',
      });

      const resized = await rawImg.clone().resize(1024, 1024);
      const rgb = resized.rgb();
      const { data } = rgb;

      // 2. Official RMBG-1.4 normalization: (pixel / 255.0) - 0.5, CHW format
      const floatData = new Float32Array(3 * 1024 * 1024);
      const channelSize = 1024 * 1024;
      for (let i = 0; i < channelSize; i++) {
        floatData[i] = (data[i * 3] / 255.0) - 0.5;
        floatData[channelSize + i] = (data[i * 3 + 1] / 255.0) - 0.5;
        floatData[2 * channelSize + i] = (data[i * 3 + 2] / 255.0) - 0.5;
      }

      const inputTensor = new Tensor('float32', floatData, [1, 3, 1024, 1024]);

      // 3. Run inference on RMBG-1.4
      self.postMessage({
        type: 'step',
        id,
        stage: 'inference',
        percent: 65,
        title: 'Neural Edge Detection',
        detail: 'RMBG-1.4 segmenting subject and transparency...',
      });

      const outputs = await currentModel({ input: inputTensor });
      const outputName = Object.keys(outputs)[0];
      const tensor = outputs[outputName];
      const outData = tensor.data;

      // 4. Smoothstep thresholding to cleanly cut away background ropes/nets/noise
      // while keeping fine edge transitions smooth
      self.postMessage({
        type: 'step',
        id,
        stage: 'matting',
        percent: 86,
        title: 'Refining Alpha Matting',
        detail: 'Cleaning background clutter and smoothing fine edges...',
      });

      // Min-max normalization as in official RMBG-1.4 pipeline
      let minVal = Infinity;
      let maxVal = -Infinity;
      for (let i = 0; i < outData.length; i++) {
        const val = outData[i];
        if (!Number.isNaN(val)) {
          if (val < minVal) minVal = val;
          if (val > maxVal) maxVal = val;
        }
      }
      if (minVal === Infinity) minVal = 0;
      if (maxVal === -Infinity) maxVal = 1;
      const range = (maxVal - minVal) > 0.0001 ? (maxVal - minVal) : 1;

      const thresh = Math.max(0.1, Math.min(0.9, threshold));
      const edge0 = Math.max(0.01, thresh - 0.18);
      const edge1 = Math.min(0.99, thresh + 0.18);
      const span = edge1 - edge0 || 0.01;

      const rawMask = new Uint8Array(1024 * 1024);
      for (let i = 0; i < outData.length; i++) {
        let raw = outData[i];
        if (Number.isNaN(raw)) {
          raw = i > 0 ? (rawMask[i - 1] / 255) : 0;
        }
        const x = Math.max(0, Math.min(1, (raw - minVal) / range));
        if (x <= edge0) {
          rawMask[i] = 0;
        } else if (x >= edge1) {
          rawMask[i] = 255;
        } else {
          const t = (x - edge0) / span;
          rawMask[i] = Math.round(t * t * (3 - 2 * t) * 255);
        }
      }

      // Morphological salt-and-pepper / pinhole cleanup pass:
      // Fills isolated blank holes inside dark subjects (e.g. navy hoodies)
      // and eliminates stray noise specks in the background while preserving natural edges.
      const cleanedMask = new Uint8Array(1024 * 1024);
      const w = 1024;
      let currentMask = rawMask;
      for (let pass = 0; pass < 2; pass++) {
        const nextMask = (pass === 0) ? cleanedMask : rawMask;
        for (let y = 0; y < w; y++) {
          const row = y * w;
          for (let x = 0; x < w; x++) {
            const idx = row + x;
            const v = currentMask[idx];
            if (y === 0 || y === w - 1 || x === 0 || x === w - 1) {
              nextMask[idx] = v;
              continue;
            }
            let solidCount = 0;
            let emptyCount = 0;
            const n1 = currentMask[idx - w - 1];
            const n2 = currentMask[idx - w];
            const n3 = currentMask[idx - w + 1];
            const n4 = currentMask[idx - 1];
            const n5 = currentMask[idx + 1];
            const n6 = currentMask[idx + w - 1];
            const n7 = currentMask[idx + w];
            const n8 = currentMask[idx + w + 1];

            if (n1 > 200) solidCount++; else if (n1 < 50) emptyCount++;
            if (n2 > 200) solidCount++; else if (n2 < 50) emptyCount++;
            if (n3 > 200) solidCount++; else if (n3 < 50) emptyCount++;
            if (n4 > 200) solidCount++; else if (n4 < 50) emptyCount++;
            if (n5 > 200) solidCount++; else if (n5 < 50) emptyCount++;
            if (n6 > 200) solidCount++; else if (n6 < 50) emptyCount++;
            if (n7 > 200) solidCount++; else if (n7 < 50) emptyCount++;
            if (n8 > 200) solidCount++; else if (n8 < 50) emptyCount++;

            // If a pixel is transparent/low alpha (< 180) inside a solid subject (>= 6 solid neighbors), heal it
            if (v < 180 && solidCount >= 6) {
              nextMask[idx] = 255;
            }
            // If a pixel is opaque/semi-opaque (> 50) inside an empty background (>= 6 empty neighbors), clean it
            else if (v > 50 && emptyCount >= 6) {
              nextMask[idx] = 0;
            } else {
              nextMask[idx] = v;
            }
          }
        }
        currentMask = nextMask;
      }

      // 5. Convert to high-resolution mask and composite with original full-size pixels
      self.postMessage({
        type: 'step',
        id,
        stage: 'composite',
        percent: 95,
        title: 'Generating Native Cutout',
        detail: `Preserving full ${originalWidth} × ${originalHeight}px native resolution...`,
      });

      const mask1024 = new RawImage(currentMask, 1024, 1024, 1);
      const maskRaw = await mask1024.resize(originalWidth, originalHeight);
      const cutoutRaw = rawImg.clone().putAlpha(maskRaw);

      // Convert to Blobs
      const [cutoutBlob, maskBlob] = await Promise.all([
        cutoutRaw.toBlob(),
        maskRaw.toBlob(),
      ]);

      const elapsedMs = Math.round(performance.now() - startTime);

      self.postMessage({
        type: 'step',
        id,
        stage: 'complete',
        percent: 100,
        title: 'Background Removed!',
        detail: '100% Complete',
      });

      self.postMessage({
        type: 'success',
        id,
        cutoutBlob,
        maskBlob,
        maskWidth: maskRaw.width,
        maskHeight: maskRaw.height,
        originalWidth,
        originalHeight,
        elapsedMs,
        device: activeDevice,
      });
    } catch (error: any) {
      console.error('Background removal error in worker:', error);
      self.postMessage({
        type: 'error',
        id,
        error: error?.message || 'Failed to remove background. Please try another image.',
      });
    }
  }
});
