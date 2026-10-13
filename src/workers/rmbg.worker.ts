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
      self.postMessage({ type: 'status', id, status: 'loading_model' });
      const currentModel = await initModel();

      self.postMessage({ type: 'status', id, status: 'processing' });

      // Load original image
      const rawImg = await RawImage.fromBlob(imageBlob);
      const originalWidth = rawImg.width;
      const originalHeight = rawImg.height;

      // 1. Resize to 1024x1024
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
      const outputs = await currentModel({ input: inputTensor });
      const outputName = Object.keys(outputs)[0];
      const tensor = outputs[outputName];
      const outData = tensor.data;

      // 4. Smoothstep thresholding to cleanly cut away background ropes/nets/noise
      // while keeping fine edge transitions smooth
      const thresh = Math.max(0.1, Math.min(0.9, threshold));
      const edge0 = Math.max(0.02, thresh - 0.16);
      const edge1 = Math.min(0.98, thresh + 0.16);
      const span = edge1 - edge0 || 0.01;

      const maskUint8 = new Uint8Array(1024 * 1024);
      for (let i = 0; i < outData.length; i++) {
        const x = outData[i];
        if (x <= edge0) {
          maskUint8[i] = 0;
        } else if (x >= edge1) {
          maskUint8[i] = 255;
        } else {
          const t = (x - edge0) / span;
          maskUint8[i] = Math.round(t * t * (3 - 2 * t) * 255);
        }
      }

      // 5. Convert to high-resolution mask and composite with original full-size pixels
      const mask1024 = new RawImage(maskUint8, 1024, 1024, 1);
      const maskRaw = await mask1024.resize(originalWidth, originalHeight);
      const cutoutRaw = rawImg.clone().putAlpha(maskRaw);

      // Convert to Blobs
      const [cutoutBlob, maskBlob] = await Promise.all([
        cutoutRaw.toBlob(),
        maskRaw.toBlob(),
      ]);

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
