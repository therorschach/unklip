import { AutoModel, AutoProcessor, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3';

// Configure environment for in-browser execution
env.allowLocalModels = false;

const MODEL_ID = 'briaai/RMBG-1.4';

let modelPromise: Promise<any> | null = null;
let model: any = null;
let processor: any = null;
let activeDevice: 'webgpu' | 'wasm' = 'wasm';

async function initModel() {
  if (model && processor) return { model, processor };
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
      processor = await AutoProcessor.from_pretrained(MODEL_ID, {
        progress_callback: progressCallback,
      });

      model = await AutoModel.from_pretrained(MODEL_ID, {
        device,
        progress_callback: progressCallback,
      });

      return { model, processor };
    } catch (gpuErr) {
      if (device === 'webgpu') {
        console.warn('WebGPU model load failed, retrying with WASM fallback:', gpuErr);
        activeDevice = 'wasm';
        self.postMessage({ type: 'device_info', device: activeDevice });

        model = await AutoModel.from_pretrained(MODEL_ID, {
          device: 'wasm',
          progress_callback: progressCallback,
        });

        return { model, processor };
      }
      throw gpuErr;
    }
  })();

  return modelPromise;
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
      const instances = await initModel();
      const currentModel = instances.model;
      const currentProcessor = instances.processor;

      self.postMessage({ type: 'status', id, status: 'processing' });

      // Load original image
      const rawImg = await RawImage.fromBlob(imageBlob);
      const originalWidth = rawImg.width;
      const originalHeight = rawImg.height;

      // Preprocess image
      const { pixel_values } = await currentProcessor(rawImg);

      // Run inference on RMBG-1.4
      const outputs = await currentModel({ input: pixel_values });
      const outputName = Object.keys(outputs)[0];
      const tensor = outputs[outputName];

      // Convert output tensor to high-resolution alpha mask
      const maskTensor = tensor[0].mul(255).to('uint8');
      const maskRaw = await RawImage.fromTensor(maskTensor).resize(originalWidth, originalHeight);

      // Composite original image pixels with high-res mask
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
