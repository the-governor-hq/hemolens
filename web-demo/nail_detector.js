/* ================================================================
   HemoLens — Nail Detector (YOLOv8-nano ONNX)
   ================================================================
   Detects fingernail bounding boxes in a full-hand photo.
   Used as a preprocessing stage before the Hb regression model.

   Pipeline:
     1. Resize input image to 320×320 (letterbox, preserve aspect)
     2. Run YOLOv8-nano ONNX inference
     3. Post-process: decode boxes, NMS, filter by confidence
     4. Return nail bounding boxes in original image coordinates

   Usage:
     const detector = new NailDetector();
     await detector.load("model/nail_detector.onnx");
     const detections = await detector.detect(imageData, origW, origH);
     // detections = [{ x, y, w, h, confidence, className }, ...]
   ================================================================ */

class NailDetector {
  constructor(options = {}) {
    this.session = null;
    this.inputSize = options.inputSize || 320;
    this.confThreshold = options.confThreshold || 0.35;
    this.iouThreshold = options.iouThreshold || 0.45;
    this.maxDetections = options.maxDetections || 10;
    this.classNames = ["nail", "skin"];
  }

  async load(modelPath) {
    const response = await fetch(modelPath);
    if (!response.ok) throw new Error(`Nail detector fetch failed: HTTP ${response.status}`);
    const buffer = await response.arrayBuffer();
    this.session = await ort.InferenceSession.create(buffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    console.log("[NailDetector] Loaded. Inputs:", this.session.inputNames, "Outputs:", this.session.outputNames);
  }

  /**
   * Preprocess: letterbox resize to inputSize × inputSize.
   * Returns { tensor, scale, padX, padY } for coordinate mapping.
   */
  _preprocess(imageData, origW, origH) {
    const size = this.inputSize;

    // Create a canvas for the letterboxed image
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");

    // Grey letterbox background (like YOLO training)
    ctx.fillStyle = "#808080";
    ctx.fillRect(0, 0, size, size);

    // Compute scale and padding
    const scale = Math.min(size / origW, size / origH);
    const newW = Math.round(origW * scale);
    const newH = Math.round(origH * scale);
    const padX = Math.round((size - newW) / 2);
    const padY = Math.round((size - newH) / 2);

    // We need a temporary canvas from the source ImageData
    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = origW;
    srcCanvas.height = origH;
    const srcCtx = srcCanvas.getContext("2d");
    srcCtx.putImageData(imageData, 0, 0);

    // Draw scaled image onto letterbox
    ctx.drawImage(srcCanvas, padX, padY, newW, newH);

    // Extract pixel data and build CHW float32 tensor [1, 3, H, W]
    const pixels = ctx.getImageData(0, 0, size, size).data;
    const float32 = new Float32Array(3 * size * size);
    const pixelCount = size * size;

    for (let i = 0; i < pixelCount; i++) {
      float32[0 * pixelCount + i] = pixels[i * 4 + 0] / 255.0; // R
      float32[1 * pixelCount + i] = pixels[i * 4 + 1] / 255.0; // G
      float32[2 * pixelCount + i] = pixels[i * 4 + 2] / 255.0; // B
    }

    return {
      tensor: new ort.Tensor("float32", float32, [1, 3, size, size]),
      scale,
      padX,
      padY,
    };
  }

  /**
   * Post-process YOLOv8 raw output.
   * YOLOv8 output shape: [1, (4 + nc), N] where N = number of anchors.
   * Columns: cx, cy, w, h, class_score_0, class_score_1, ...
   */
  _postprocess(output, scale, padX, padY, origW, origH) {
    const data = output.data;
    const [batch, channels, numBoxes] = output.dims;
    const nc = channels - 4; // number of classes

    const boxes = [];

    for (let i = 0; i < numBoxes; i++) {
      // Find best class
      let bestScore = 0;
      let bestClass = 0;
      for (let c = 0; c < nc; c++) {
        const score = data[(4 + c) * numBoxes + i];
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }

      if (bestScore < this.confThreshold) continue;

      // Decode box (in letterboxed coordinates)
      const cx = data[0 * numBoxes + i];
      const cy = data[1 * numBoxes + i];
      const bw = data[2 * numBoxes + i];
      const bh = data[3 * numBoxes + i];

      // Convert to original image coordinates
      const x1 = (cx - bw / 2 - padX) / scale;
      const y1 = (cy - bh / 2 - padY) / scale;
      const x2 = (cx + bw / 2 - padX) / scale;
      const y2 = (cy + bh / 2 - padY) / scale;

      // Clamp to image bounds
      const fx = Math.max(0, Math.min(origW, x1));
      const fy = Math.max(0, Math.min(origH, y1));
      const fw = Math.max(0, Math.min(origW, x2)) - fx;
      const fh = Math.max(0, Math.min(origH, y2)) - fy;

      if (fw < 5 || fh < 5) continue; // too small

      boxes.push({
        x: fx,
        y: fy,
        w: fw,
        h: fh,
        confidence: bestScore,
        classId: bestClass,
        className: this.classNames[bestClass] || `class_${bestClass}`,
      });
    }

    // Non-maximum suppression per class
    return this._nms(boxes);
  }

  /**
   * Greedy NMS per class.
   */
  _nms(boxes) {
    // Group by class
    const byClass = {};
    for (const b of boxes) {
      if (!byClass[b.classId]) byClass[b.classId] = [];
      byClass[b.classId].push(b);
    }

    const result = [];
    for (const classId in byClass) {
      let classBoxes = byClass[classId].sort((a, b) => b.confidence - a.confidence);
      const kept = [];

      while (classBoxes.length > 0) {
        const best = classBoxes.shift();
        kept.push(best);
        classBoxes = classBoxes.filter(b => this._iou(best, b) < this.iouThreshold);
      }

      result.push(...kept);
    }

    // Sort by confidence descending, limit
    result.sort((a, b) => b.confidence - a.confidence);
    return result.slice(0, this.maxDetections);
  }

  _iou(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w);
    const y2 = Math.min(a.y + a.h, b.y + b.h);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = a.w * a.h;
    const areaB = b.w * b.h;
    return inter / (areaA + areaB - inter + 1e-6);
  }

  /**
   * Run detection on an ImageData object.
   * @param {ImageData} imageData - the full image (e.g. from canvas)
   * @param {number} origW - original image width
   * @param {number} origH - original image height
   * @returns {Array} detections with {x, y, w, h, confidence, className}
   */
  async detect(imageData, origW, origH) {
    if (!this.session) throw new Error("NailDetector not loaded");

    const { tensor, scale, padX, padY } = this._preprocess(imageData, origW, origH);
    const inputName = this.session.inputNames[0];
    const results = await this.session.run({ [inputName]: tensor });
    const outputName = this.session.outputNames[0];
    const output = results[outputName];

    return this._postprocess(output, scale, padX, padY, origW, origH);
  }

  /**
   * Convenience: detect from a <video> or <canvas> element.
   */
  async detectFromElement(element) {
    const canvas = document.createElement("canvas");
    const w = element.videoWidth || element.width;
    const h = element.videoHeight || element.height;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(element, 0, 0, w, h);
    const imageData = ctx.getImageData(0, 0, w, h);
    return this.detect(imageData, w, h);
  }

  /**
   * Get only nail detections (filter out skin).
   */
  async detectNails(imageData, origW, origH) {
    const all = await this.detect(imageData, origW, origH);
    return all.filter(d => d.className === "nail");
  }

  /**
   * Get only nail detections from a video/canvas element.
   */
  async detectNailsFromElement(element) {
    const all = await this.detectFromElement(element);
    return all.filter(d => d.className === "nail");
  }

  get isLoaded() {
    return this.session !== null;
  }
}

// Export for use in app.js
// (loaded via <script src="nail_detector.js"></script> before app.js)
window.NailDetector = NailDetector;
