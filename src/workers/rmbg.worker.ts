import { pipeline, env, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3';

// Configure environment for browser execution
env.allowLocalModels = false;

// We use the battle-tested state-of-the-art BRIA RMBG-1.4 model
const MODEL_ID = 'briaai/RMBG-1.4';

let segmenterPromise: Promise<any> | null = null;
let activeDevice: 'webgpu' | 'wasm' = 'wasm';

async function initSegmenter() {
  if (segmenterPromise) return segmenterPromise;

  segmenterPromise = (async () => {
    let device: 'webgpu' | 'wasm' = 'wasm';

    // Check for WebGPU support in worker context
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const adapter = await (navigator as any).gpu?.requestAdapter();
        if (adapter) {
          device = 'webgpu';
        }
      } catch (e) {
        console.warn('WebGPU not available in worker context, falling back to WASM:', e);
        device = 'wasm';
      }
    }

    activeDevice = device;
    self.postMessage({ type: 'device_info', device: activeDevice });

    try {
      const seg = await pipeline('image-segmentation', MODEL_ID, {
        device,
        progress_callback: (progressInfo: any) => {
          self.postMessage({
            type: 'progress',
            data: progressInfo,
          });
        },
      });
      return seg;
    } catch (gpuError) {
      if (device === 'webgpu') {
        console.warn('WebGPU pipeline failed, retrying with WASM CPU fallback:', gpuError);
        activeDevice = 'wasm';
        self.postMessage({ type: 'device_info', device: activeDevice });

        const seg = await pipeline('image-segmentation', MODEL_ID, {
          device: 'wasm',
          progress_callback: (progressInfo: any) => {
            self.postMessage({
              type: 'progress',
              data: progressInfo,
            });
          },
        });
        return seg;
      }
      throw gpuError;
    }
  })();

  return segmenterPromise;
}

self.addEventListener('message', async (event: MessageEvent) => {
  const { action, id, imageBlob } = event.data;

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
      await initSegmenter();
      self.postMessage({ type: 'warmup_complete', id, device: activeDevice });
    } catch (error: any) {
      self.postMessage({ type: 'warmup_error', id, error: error?.message || String(error) });
    }
    return;
  }

  if (action === 'remove_bg') {
    const startTime = performance.now();
    try {
      self.postMessage({ type: 'status', id, status: 'loading_model' });
      const segmenter = await initSegmenter();

      self.postMessage({ type: 'status', id, status: 'processing' });

      // Load raw image from the uploaded blob
      const rawImg = await RawImage.fromBlob(imageBlob);
      const originalWidth = rawImg.width;
      const originalHeight = rawImg.height;

      // Run AI segmentation inference
      const output = await segmenter(rawImg);

      // Extract mask
      const maskRaw: RawImage = Array.isArray(output) && output[0]?.mask ? output[0].mask : output;

      let cutoutBlob: Blob | null = null;
      let maskBlob: Blob | null = null;

      try {
        // High-precision cutout: apply mask to original full-size image pixels
        const cutoutRaw = rawImg.clone().putAlpha(maskRaw);
        cutoutBlob = await cutoutRaw.toBlob();
      } catch (err) {
        console.warn('Direct cutout conversion failed, falling back to mask export:', err);
      }

      try {
        maskBlob = await maskRaw.toBlob();
      } catch (err) {
        console.warn('Mask conversion failed:', err);
      }

      const elapsedMs = Math.round(performance.now() - startTime);

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
