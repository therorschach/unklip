import { AutoModel, RawImage, Tensor, env } from '/vendor/transformers.min.js';

// Configure environment for in-browser execution
env.allowLocalModels = false;
env.useBrowserCache = typeof caches !== 'undefined';

const MODEL_ID = 'briaai/RMBG-1.4';

// Maximum output dimension on mobile to prevent OOM and speed up PNG encoding.
// Phone cameras shoot 12MP+ (4032×3024) — capping mobile output to 1600px
// keeps retina quality crisp while cutting compositing and PNG encode times in half.
const MOBILE_MAX_DIM = 1600;

type DeviceType = 'webgpu' | 'wasm';
type DtypeType = 'fp16' | 'q8' | 'fp32';

let modelPromise: Promise<any> | null = null;
let model: any = null;
let activeDevice: DeviceType = 'wasm';
let activeDtype: DtypeType = 'q8';

// Precomputed normalization lookup table for RMBG-1.4: (pixel / 255.0) - 0.5
const NORM_LUT = new Float32Array(256);
for (let b = 0; b < 256; b++) {
  NORM_LUT[b] = (b / 255.0) - 0.5;
}

function isMobile(): boolean {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent || '';
  const isMobileUA = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua);
  const isMobileData = Boolean((navigator as any).userAgentData?.mobile);
  return isMobileUA || isMobileData;
}

interface ModelCandidate {
  device: DeviceType;
  dtype: DtypeType;
}

function getCandidateConfigs(gpuAdapter: any | null): ModelCandidate[] {
  if (gpuAdapter) {
    const supportsF16 = Boolean(gpuAdapter.features?.has?.('shader-f16'));

    // Hardware WebGPU fp16 is ultra-fast (sub-second 0.3s-0.6s) on GPUs with shader-f16.
    // Without shader-f16, WebGPU fp32 takes 300+ seconds on software/integrated GPUs,
    // whereas CPU WASM q8 takes only 4-7 seconds with native SIMD int8 vectorization.
    if (supportsF16) {
      return [
        { device: 'webgpu', dtype: 'fp16' },
        { device: 'wasm', dtype: 'q8' },
        { device: 'webgpu', dtype: 'fp32' },
      ];
    } else {
      return [
        { device: 'wasm', dtype: 'q8' },
        { device: 'webgpu', dtype: 'fp32' },
      ];
    }
  }

  // Pure CPU WASM environment: q8 uses int8 SIMD instructions (4-7s)
  return [
    { device: 'wasm', dtype: 'q8' },
    { device: 'wasm', dtype: 'fp32' },
  ];
}

async function initModel() {
  if (model) return model;
  if (modelPromise) return modelPromise;

  modelPromise = (async () => {
    let adapter: any = null;
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        adapter = await (navigator as any).gpu?.requestAdapter();
      } catch (e) {
        console.warn('WebGPU requestAdapter failed, falling back to WASM:', e);
      }
    }

    const candidates = getCandidateConfigs(adapter);
    const progressCallback = (progressInfo: any) => {
      self.postMessage({
        type: 'progress',
        data: progressInfo,
      });
    };

    let lastError: any = null;
    for (const candidate of candidates) {
      try {
        activeDevice = candidate.device;
        activeDtype = candidate.dtype;
        self.postMessage({
          type: 'device_info',
          device: activeDevice,
          dtype: activeDtype,
        });

        model = await AutoModel.from_pretrained(MODEL_ID, {
          device: candidate.device,
          dtype: candidate.dtype,
          progress_callback: progressCallback,
        });
        return model;
      } catch (err) {
        console.warn(`Failed loading RMBG-1.4 with [${candidate.device}, ${candidate.dtype}]:`, err);
        lastError = err;
      }
    }

    throw lastError || new Error('Failed to load RMBG model with all available configurations');
  })();

  return modelPromise;
}

self.addEventListener('message', async (event: MessageEvent) => {
  const { action, id, imageBlob, threshold = 0.5 } = event.data;

  if (action === 'get_device') {
    if (model) {
      self.postMessage({ type: 'device_info', device: activeDevice, dtype: activeDtype });
      return;
    }

    let adapter: any = null;
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        adapter = await (navigator as any).gpu?.requestAdapter();
      } catch {}
    }

    const candidates = getCandidateConfigs(adapter);
    const primary = candidates[0];
    self.postMessage({
      type: 'device_info',
      device: primary.device,
      dtype: primary.dtype,
    });
    return;
  }

  if (action === 'warmup') {
    try {
      await initModel();
      self.postMessage({ type: 'warmup_complete', id, device: activeDevice, dtype: activeDtype });
    } catch (error: any) {
      self.postMessage({ type: 'warmup_error', id, error: error?.message || String(error) });
    }
    return;
  }

  if (action === 'remove_bg') {
    const startTime = performance.now();
    let origBitmap: ImageBitmap | null = null;
    try {
      self.postMessage({
        type: 'step',
        id,
        stage: 'init',
        percent: 20,
        title: 'Initializing AI Engine',
        detail: 'Loading RMBG weights into browser memory...',
      });
      const currentModel = await initModel();

      self.postMessage({
        type: 'step',
        id,
        stage: 'decoding',
        percent: 35,
        title: 'Decoding Image',
        detail: 'Reading pixel channels and geometry...',
      });

      let originalWidth = 0;
      let originalHeight = 0;
      let floatData: Float32Array | null = null;
      let rawImgFallback: any = null;

      // 1. Hardware-accelerated native decode & downscale to 1024x1024 via createImageBitmap & OffscreenCanvas
      const canUseOffscreen = typeof createImageBitmap === 'function' && typeof OffscreenCanvas !== 'undefined';

      if (canUseOffscreen) {
        try {
          origBitmap = await createImageBitmap(imageBlob);
          originalWidth = origBitmap.width;
          originalHeight = origBitmap.height;

          self.postMessage({
            type: 'step',
            id,
            stage: 'tensor',
            percent: 50,
            title: 'Preparing Neural Tensor',
            detail: 'Rescaling to 1024×1024 RGB input...',
          });

          const canvas = new OffscreenCanvas(1024, 1024);
          const ctx = canvas.getContext('2d', { willReadFrequently: true });
          if (!ctx) throw new Error('Could not obtain OffscreenCanvas 2D context');

          ctx.drawImage(origBitmap, 0, 0, 1024, 1024);

          const imageData = ctx.getImageData(0, 0, 1024, 1024);
          const { data } = imageData; // RGBA Uint8ClampedArray

          // RMBG-1.4 normalization via precomputed LUT: (pixel / 255.0) - 0.5, CHW format
          floatData = new Float32Array(3 * 1024 * 1024);
          const channelSize = 1024 * 1024;
          for (let i = 0; i < channelSize; i++) {
            const offset = i * 4;
            floatData[i] = NORM_LUT[data[offset]];
            floatData[channelSize + i] = NORM_LUT[data[offset + 1]];
            floatData[2 * channelSize + i] = NORM_LUT[data[offset + 2]];
          }
        } catch (downscaleErr) {
          console.warn('Hardware downscaling failed, falling back to RawImage:', downscaleErr);
          floatData = null;
          if (origBitmap && typeof origBitmap.close === 'function') {
            try { origBitmap.close(); } catch (_) {}
            origBitmap = null;
          }
        }
      }

      // Fallback: CPU JavaScript array resizing via RawImage
      if (!floatData) {
        rawImgFallback = await RawImage.fromBlob(imageBlob);
        originalWidth = rawImgFallback.width;
        originalHeight = rawImgFallback.height;

        self.postMessage({
          type: 'step',
          id,
          stage: 'tensor',
          percent: 50,
          title: 'Preparing Neural Tensor',
          detail: 'Rescaling to 1024×1024 RGB input...',
        });

        const resized = await rawImgFallback.clone().resize(1024, 1024);
        const rgb = resized.rgb();
        const { data } = rgb;

        floatData = new Float32Array(3 * 1024 * 1024);
        const channelSize = 1024 * 1024;
        for (let i = 0; i < channelSize; i++) {
          floatData[i] = NORM_LUT[data[i * 3]];
          floatData[channelSize + i] = NORM_LUT[data[i * 3 + 1]];
          floatData[2 * channelSize + i] = NORM_LUT[data[i * 3 + 2]];
        }
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
        percent: 90,
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

      // Fast single-pass morphological pinhole filter:
      // Fills isolated pinholes inside dark solid subjects and eliminates stray specks in the background.
      // Skips uniform foreground/background regions where all cardinal neighbors match.
      const cleanedMask = new Uint8Array(1024 * 1024);
      const w = 1024;
      for (let y = 1; y < w - 1; y++) {
        const row = y * w;
        for (let x = 1; x < w - 1; x++) {
          const idx = row + x;
          const v = rawMask[idx];
          const top = rawMask[idx - w];
          const bottom = rawMask[idx + w];
          const left = rawMask[idx - 1];
          const right = rawMask[idx + 1];

          // Fast path: if pixel is 0 and all 4 cardinal neighbors are < 50, it is pure background
          if (v < 50 && top < 50 && bottom < 50 && left < 50 && right < 50) {
            cleanedMask[idx] = 0;
            continue;
          }
          // Fast path: if pixel is 255 and all 4 cardinal neighbors are > 200, it is pure foreground
          if (v > 200 && top > 200 && bottom > 200 && left > 200 && right > 200) {
            cleanedMask[idx] = 255;
            continue;
          }

          // Boundary or suspect pixel: check all 8 neighbors
          let solidCount = (top > 200 ? 1 : 0) + (bottom > 200 ? 1 : 0) + (left > 200 ? 1 : 0) + (right > 200 ? 1 : 0);
          let emptyCount = (top < 50 ? 1 : 0) + (bottom < 50 ? 1 : 0) + (left < 50 ? 1 : 0) + (right < 50 ? 1 : 0);
          const tl = rawMask[idx - w - 1];
          const tr = rawMask[idx - w + 1];
          const bl = rawMask[idx + w - 1];
          const br = rawMask[idx + w + 1];
          if (tl > 200) solidCount++; else if (tl < 50) emptyCount++;
          if (tr > 200) solidCount++; else if (tr < 50) emptyCount++;
          if (bl > 200) solidCount++; else if (bl < 50) emptyCount++;
          if (br > 200) solidCount++; else if (br < 50) emptyCount++;

          if (v < 180 && solidCount >= 6) {
            cleanedMask[idx] = 255;
          } else if (v > 50 && emptyCount >= 6) {
            cleanedMask[idx] = 0;
          } else {
            cleanedMask[idx] = v;
          }
        }
      }
      const currentMask = cleanedMask;

      // 5. Convert to high-resolution mask and composite with original full-size pixels
      // On mobile devices, cap output resolution to prevent memory exhaustion
      const isMobileDevice = isMobile();
      let outputWidth = originalWidth;
      let outputHeight = originalHeight;
      if (isMobileDevice && (originalWidth > MOBILE_MAX_DIM || originalHeight > MOBILE_MAX_DIM)) {
        const scale = MOBILE_MAX_DIM / Math.max(originalWidth, originalHeight);
        outputWidth = Math.round(originalWidth * scale);
        outputHeight = Math.round(originalHeight * scale);
      }

      self.postMessage({
        type: 'step',
        id,
        stage: 'composite',
        percent: 96,
        title: 'Generating Native Cutout',
        detail: `Compositing at ${outputWidth} × ${outputHeight}px...`,
      });

      // Use OffscreenCanvas compositing path (reliable on mobile) when available,
      // falling back to RawImage-based compositing only when necessary.
      const canUseCanvas = typeof OffscreenCanvas !== 'undefined' &&
                           typeof createImageBitmap === 'function' &&
                           typeof (OffscreenCanvas.prototype as any)?.convertToBlob === 'function';

      let cutoutBlob: Blob;
      let maskBlob: Blob | null = null;

      if (canUseCanvas) {
        // --- Fast hardware-accelerated Canvas compositing ---
        if (!origBitmap) {
          origBitmap = await createImageBitmap(imageBlob);
        }

        // Fast 32-bit write of 1024×1024 grayscale mask
        const mask1024Canvas = new OffscreenCanvas(1024, 1024);
        const mask1024Ctx = mask1024Canvas.getContext('2d');
        if (!mask1024Ctx) throw new Error('Could not create mask1024 canvas context');
        const maskImageData = mask1024Ctx.createImageData(1024, 1024);
        const data32 = new Uint32Array(maskImageData.data.buffer);
        for (let i = 0; i < 1048576; i++) {
          data32[i] = (currentMask[i] << 24) | 0x00FFFFFF;
        }
        mask1024Ctx.putImageData(maskImageData, 0, 0);

        // Composite: draw original, then apply mask directly via destination-in
        // Avoids allocating a second full-size intermediate mask canvas
        const compCanvas = new OffscreenCanvas(outputWidth, outputHeight);
        const compCtx = compCanvas.getContext('2d');
        if (!compCtx) throw new Error('Could not create composite canvas context');

        compCtx.drawImage(origBitmap, 0, 0, outputWidth, outputHeight);
        compCtx.globalCompositeOperation = 'destination-in';
        compCtx.drawImage(mask1024Canvas, 0, 0, outputWidth, outputHeight);

        // Single PNG encode (cutout only, skipping redundant mask PNG encode)
        cutoutBlob = await compCanvas.convertToBlob({ type: 'image/png' });

        if (typeof origBitmap.close === 'function') origBitmap.close();
        origBitmap = null;
      } else {
        // --- RawImage fallback (non-mobile / environments without OffscreenCanvas) ---
        const mask1024 = new RawImage(currentMask, 1024, 1024, 1);
        const maskRaw = await mask1024.resize(outputWidth, outputHeight);
        const rawImg = rawImgFallback || (await RawImage.fromBlob(imageBlob));

        // If output was downscaled, resize the source image too
        let srcImg = rawImg;
        if (outputWidth !== originalWidth || outputHeight !== originalHeight) {
          srcImg = await rawImg.resize(outputWidth, outputHeight);
        }

        const cutoutRaw = srcImg.clone().putAlpha(maskRaw);

        [cutoutBlob, maskBlob] = await Promise.all([
          cutoutRaw.toBlob(),
          maskRaw.toBlob(),
        ]);
      }

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
        maskWidth: outputWidth,
        maskHeight: outputHeight,
        originalWidth: outputWidth,
        originalHeight: outputHeight,
        elapsedMs,
        device: activeDevice,
        dtype: activeDtype,
      });
    } catch (error: any) {
      if (origBitmap && typeof (origBitmap as any).close === 'function') {
        try { (origBitmap as any).close(); } catch (_) {}
        origBitmap = null;
      }
      console.error('Background removal error in worker:', error);
      self.postMessage({
        type: 'error',
        id,
        error: error?.message || 'Failed to remove background. Please try another image.',
      });
    }
  }
});
