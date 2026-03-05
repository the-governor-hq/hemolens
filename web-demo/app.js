/* ================================================================
   HemoLens Web Demo - Application Logic
   ================================================================
   Pipeline:
     1. Load ONNX model via ONNX Runtime Web (WASM backend)
     2. Stream camera via getUserMedia (prefer rear camera)
     3. On capture: run multi-frame inference (30 frames)
     4. Per frame: crop nail ROI, resize 224×224, ImageNet normalize
     5. Drop outliers via IQR, take median
     6. Display result with severity classification + confidence stats
   ================================================================ */

// ─── Configuration ───

const MODEL_PATH = "model/hemolens_hybrid_web.onnx";
const INPUT_SIZE = 224;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];
const MULTI_FRAME_COUNT = 30;
const FRAME_INTERVAL_MS = 80;  // ~12.5 fps capture rate

// Frame validity gates (to avoid "normal" results on blocked/black frames)
// These thresholds are intentionally conservative and should be tuned with real captures.
const MIN_VALID_FRAMES = 10;
const FRAME_QUALITY = {
  minMeanLuma: 0.08,     // too dark if below (~20/255)
  maxMeanLuma: 0.95,     // too bright if above
  minStdLuma:  0.02,     // too uniform / no texture
  maxBlackFrac: 0.85,    // mostly black pixels
  maxWhiteFrac: 0.85,    // mostly white pixels
};

// WHO anemia severity thresholds (g/dL)
const SEVERITY = [
  { label: "Severe",   min: 0,  max: 8,  cls: "severe"   },
  { label: "Moderate", min: 8,  max: 11, cls: "moderate" },
  { label: "Mild",     min: 11, max: 13, cls: "mild"     },
  { label: "Normal",   min: 13, max: 25, cls: "normal"   },
];


// ─── DOM References ───

const $splash       = document.getElementById("splash");
const $splashStatus = document.getElementById("splashStatus");
const $loaderBar    = document.querySelector(".loader-bar");
const $app          = document.getElementById("app");
const $video        = document.getElementById("video");
const $captureCanvas= document.getElementById("captureCanvas");
const $guideOverlay = document.getElementById("guideOverlay");
const $scanLine     = document.getElementById("scanLine");
const $captureBtn   = document.getElementById("captureBtn");
const $switchBtn    = document.getElementById("switchBtn");
const $flashBtn     = document.getElementById("flashBtn");
const $uploadBtn    = document.getElementById("uploadBtn");
const $fileInput    = document.getElementById("fileInput");
const $resultSection= document.getElementById("resultSection");
const $cameraSection= document.getElementById("cameraSection");
const $previewCanvas= document.getElementById("previewCanvas");
const $hbValue      = document.getElementById("hbValue");
const $gaugeMarker  = document.getElementById("gaugeMarker");
const $severityBadge= document.getElementById("severityBadge");
const $closeResult  = document.getElementById("closeResult");
const $retakeBtn    = document.getElementById("retakeBtn");

const $instructionText = document.getElementById("instructionText");

// Multi-frame scanning overlay
const $scanOverlay      = document.getElementById("scanOverlay");
const $scanRingProgress = document.getElementById("scanRingProgress");
const $scanFrameNum     = document.getElementById("scanFrameNum");
const $scanDots         = document.getElementById("scanDots");
const $scanCancelBtn    = document.getElementById("scanCancelBtn");

// Multi-frame stats
const $multiFrameStats  = document.getElementById("multiFrameStats");
const $statFramesUsed   = document.getElementById("statFramesUsed");
const $statStdDev       = document.getElementById("statStdDev");
const $statRange        = document.getElementById("statRange");


// ─── State ───

let session = null;          // ONNX Runtime inference session
let currentStream = null;    // MediaStream
let facingMode = "environment";  // prefer rear camera
let isProcessing = false;
let flashOn = false;         // torch state
let scanAborted = false;     // cancel flag for multi-frame


// ─── Instruction Text ───

const INSTRUCTIONS = {
  camera: "Place one fingernail inside the frame — fill the box as much as possible",
  upload: "Uploaded image mode — analyzing one photo. Use a sharp, well-lit close-up of one fingernail.",
};

function setInstructionMode(mode) {
  if (!$instructionText) return;
  const txt = INSTRUCTIONS[mode] || INSTRUCTIONS.camera;
  $instructionText.textContent = txt;
}


// ─── Frame Quality / Validity ───

function computeLumaStats(imageData) {
  const d = imageData.data;
  const n = imageData.width * imageData.height;
  if (!n) {
    return { mean: 0, std: 0, blackFrac: 1, whiteFrac: 0 };
  }

  let sum = 0;
  let sumSq = 0;
  let black = 0;
  let white = 0;

  // Luma in [0,1]
  for (let i = 0; i < n; i++) {
    const r = d[i * 4];
    const g = d[i * 4 + 1];
    const b = d[i * 4 + 2];
    const y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
    sum += y;
    sumSq += y * y;
    if (y < 0.03) black++;
    if (y > 0.97) white++;
  }

  const mean = sum / n;
  const variance = Math.max(0, sumSq / n - mean * mean);
  const std = Math.sqrt(variance);

  return {
    mean,
    std,
    blackFrac: black / n,
    whiteFrac: white / n,
  };
}

function isFrameValid(imageData) {
  const s = computeLumaStats(imageData);
  const ok = (
    s.mean >= FRAME_QUALITY.minMeanLuma &&
    s.mean <= FRAME_QUALITY.maxMeanLuma &&
    s.std >= FRAME_QUALITY.minStdLuma &&
    s.blackFrac <= FRAME_QUALITY.maxBlackFrac &&
    s.whiteFrac <= FRAME_QUALITY.maxWhiteFrac
  );
  return { ok, stats: s };
}


// ─── UI Reset Helpers ───

function resetForNewCapture() {
  // Flush any previous result immediately
  hideResult();

  // Reset scan overlay state
  scanAborted = false;
  $scanDots.innerHTML = "";
  $scanFrameNum.textContent = "0";
  $scanRingProgress.style.strokeDashoffset = 2 * Math.PI * 52;
  $scanOverlay.classList.add("hidden");
  $guideOverlay.style.opacity = "";
}


// ─── Model Loading ───

async function loadModel() {
  try {
    updateSplash("Initializing runtime...", 10);

    // Point WASM binaries to CDN explicitly
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";

    // Single-threaded to avoid SharedArrayBuffer / COOP/COEP requirements
    ort.env.wasm.numThreads = 1;

    updateSplash("Downloading model (~10 MB)...", 25);

    // Fetch model as ArrayBuffer for reliable loading
    const response = await fetch(MODEL_PATH);
    if (!response.ok) {
      throw new Error("Model fetch failed: HTTP " + response.status);
    }
    const modelBuffer = await response.arrayBuffer();
    console.log("Model downloaded:", modelBuffer.byteLength, "bytes");

    updateSplash("Loading inference engine...", 60);

    session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    console.log("Session created. Inputs:", session.inputNames, "Outputs:", session.outputNames);
    updateSplash("Model loaded!", 100);

    // Short delay so user sees the success state
    await sleep(600);

    // Transition to app
    $splash.classList.add("fade-out");
    setTimeout(() => {
      $splash.classList.add("hidden");
      $app.classList.remove("hidden");
      startCamera();
      checkFlashSupport();
    }, 600);

  } catch (err) {
    console.error("Model load failed:", err);
    updateSplash("Error loading model. Check console.", 0);
    $loaderBar.style.background = "#ef4444";
  }
}

function updateSplash(msg, pct) {
  $splashStatus.textContent = msg;
  $loaderBar.style.width = pct + "%";
}


// ─── Camera ───

async function startCamera() {
  // Stop any existing stream
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
  }

  try {
    const constraints = {
      video: {
        facingMode: { ideal: facingMode },
        width:  { ideal: 1280 },
        height: { ideal: 960 },
      },
      audio: false,
    };

    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    $video.srcObject = currentStream;
    await $video.play();

  } catch (err) {
    console.error("Camera error:", err);
    // Fallback: try any camera
    try {
      currentStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      $video.srcObject = currentStream;
      await $video.play();
    } catch (err2) {
      alert("Could not access camera. Please grant camera permission and reload.");
    }
  }
}


// ─── Capture & Multi-Frame Inference ───

/** Compute crop coordinates from video to nail region */
function getNailCropRect() {
  const vw = $video.videoWidth;
  const vh = $video.videoHeight;
  if (!vw || !vh) return null;

  const container = document.querySelector(".camera-container");
  const cRect = container.getBoundingClientRect();

  const svgAspect = 400 / 500;
  const containerAspect = cRect.width / cRect.height;

  let svgRenderW, svgRenderH, svgOffsetX, svgOffsetY;
  if (containerAspect > svgAspect) {
    svgRenderH = cRect.height;
    svgRenderW = cRect.height * svgAspect;
    svgOffsetX = (cRect.width - svgRenderW) / 2;
    svgOffsetY = 0;
  } else {
    svgRenderW = cRect.width;
    svgRenderH = cRect.width / svgAspect;
    svgOffsetX = 0;
    svgOffsetY = (cRect.height - svgRenderH) / 2;
  }

  const nailSVG = { x: 130, y: 115, w: 140, h: 105 };
  const scale = svgRenderW / 400;
  const nailPx = {
    x: svgOffsetX + nailSVG.x * scale,
    y: svgOffsetY + nailSVG.y * scale,
    w: nailSVG.w * scale,
    h: nailSVG.h * scale,
  };

  const videoAspect = vw / vh;
  let renderW, renderH, videoOffX, videoOffY;
  if (containerAspect > videoAspect) {
    renderW = cRect.width;
    renderH = cRect.width / videoAspect;
    videoOffX = 0;
    videoOffY = (cRect.height - renderH) / 2;
  } else {
    renderH = cRect.height;
    renderW = cRect.height * videoAspect;
    videoOffX = (cRect.width - renderW) / 2;
    videoOffY = 0;
  }

  const sx = ((nailPx.x - videoOffX) / renderW) * vw;
  const sy = ((nailPx.y - videoOffY) / renderH) * vh;
  const sw = (nailPx.w / renderW) * vw;
  const sh = (nailPx.h / renderH) * vh;

  return {
    x: Math.max(0, Math.round(sx)),
    y: Math.max(0, Math.round(sy)),
    w: Math.min(Math.round(sw), vw - Math.max(0, Math.round(sx))),
    h: Math.min(Math.round(sh), vh - Math.max(0, Math.round(sy))),
  };
}


/** Grab one frame from video and crop to nail region */
function grabNailFrame(crop) {
  $captureCanvas.width = crop.w;
  $captureCanvas.height = crop.h;
  const ctx = $captureCanvas.getContext("2d");
  ctx.drawImage($video, crop.x, crop.y, crop.w, crop.h, 0, 0, crop.w, crop.h);

  // Resize to model input
  const resizeCanvas = document.createElement("canvas");
  resizeCanvas.width = INPUT_SIZE;
  resizeCanvas.height = INPUT_SIZE;
  const rCtx = resizeCanvas.getContext("2d");
  rCtx.drawImage($captureCanvas, 0, 0, INPUT_SIZE, INPUT_SIZE);
  return rCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
}


/** Run inference on a single ImageData → Hb prediction */
async function inferSingle(imageData) {
  const pixels = imageData.data;
  const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const pixelCount = INPUT_SIZE * INPUT_SIZE;

  for (let i = 0; i < pixelCount; i++) {
    const r = pixels[i * 4]     / 255.0;
    const g = pixels[i * 4 + 1] / 255.0;
    const b = pixels[i * 4 + 2] / 255.0;
    float32Data[0 * pixelCount + i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    float32Data[1 * pixelCount + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    float32Data[2 * pixelCount + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }

  const inputTensor = new ort.Tensor("float32", float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const results = await session.run({ image: inputTensor });
  return results.hb_prediction.data[0];
}


/** IQR-based outlier rejection */
function filterOutliersIQR(values) {
  if (values.length < 4) return { kept: [...values], rejected: [] };

  const sorted = [...values].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  const iqr = q3 - q1;
  const lo = q1 - 1.5 * iqr;
  const hi = q3 + 1.5 * iqr;

  const kept = [];
  const rejected = [];
  for (const v of values) {
    if (v >= lo && v <= hi) kept.push(v);
    else rejected.push(v);
  }
  return { kept, rejected };
}


/** Compute median of an array */
function median(arr) {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}


/** Standard deviation */
function stdDev(arr) {
  if (arr.length < 2) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  const variance = arr.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (arr.length - 1);
  return Math.sqrt(variance);
}


/** Update the scanning overlay progress.
 * Pass `newHb` to append a dot; pass null/undefined to only update progress.
 */
function updateScanUI(frameIdx, totalFrames, newHb) {
  const CIRCUMFERENCE = 2 * Math.PI * 52; // r=52
  const progress = frameIdx / totalFrames;
  $scanRingProgress.style.strokeDashoffset = CIRCUMFERENCE * (1 - progress);
  $scanFrameNum.textContent = frameIdx;

  // Add a dot to the scatter strip
  if (newHb !== null && newHb !== undefined) {
    const val = newHb;
    const dot = document.createElement("div");
    dot.className = "scan-dot";
    // Height maps Hb 2–20 to 6–30px
    const h = Math.max(6, Math.min(30, ((val - 2) / 18) * 24 + 6));
    dot.style.height = h + "px";
    $scanDots.appendChild(dot);
  }
}


/** Mark outlier dots in the scatter strip */
function markOutlierDots(predictions, rejectedIndices) {
  const dots = $scanDots.querySelectorAll(".scan-dot");
  for (const idx of rejectedIndices) {
    if (dots[idx]) dots[idx].classList.add("outlier");
  }
}


/** Main multi-frame capture */
async function captureMultiFrame() {
  if (isProcessing) return;

  setInstructionMode("camera");

  resetForNewCapture();

  // Basic guard: ensure video has real frames
  if ($video.readyState < 2 || !$video.videoWidth || !$video.videoHeight) {
    alert("Camera not ready yet. Please wait a moment and try again.");
    return;
  }

  const crop = getNailCropRect();
  if (!crop) return;

  isProcessing = true;
  scanAborted = false;

  // Initial flash
  const flash = document.createElement("div");
  flash.className = "flash";
  document.querySelector(".camera-container").appendChild(flash);
  setTimeout(() => flash.remove(), 400);

  // Show scanning overlay
  $scanDots.innerHTML = "";
  $scanFrameNum.textContent = "0";
  $scanRingProgress.style.strokeDashoffset = 2 * Math.PI * 52;
  $scanOverlay.classList.remove("hidden");
  $guideOverlay.style.opacity = "0.3";

  const predictions = [];
  let invalidFrames = 0;
  let lastPreviewCanvas = null;

  try {
    for (let i = 0; i < MULTI_FRAME_COUNT; i++) {
      if (scanAborted) {
        console.log("Scan cancelled by user at frame", i);
        break;
      }

      // Grab frame
      const imageData = grabNailFrame(crop);

      // Reject blocked/black/overexposed/too-uniform frames
      const validity = isFrameValid(imageData);
      if (!validity.ok) {
        invalidFrames++;
        updateScanUI(i + 1, MULTI_FRAME_COUNT, null);

        // If it's impossible to reach MIN_VALID_FRAMES anymore, bail early
        const framesLeft = (MULTI_FRAME_COUNT - 1) - i;
        const maxPossible = predictions.length + framesLeft;
        if (maxPossible < MIN_VALID_FRAMES) {
          break;
        }

        if (i < MULTI_FRAME_COUNT - 1) {
          await sleep(FRAME_INTERVAL_MS);
        }
        continue;
      }

      // Save a preview canvas from the first valid frame, then refine near the middle
      if (!lastPreviewCanvas || i === Math.floor(MULTI_FRAME_COUNT / 2)) {
        lastPreviewCanvas = document.createElement("canvas");
        lastPreviewCanvas.width = crop.w;
        lastPreviewCanvas.height = crop.h;
        const pCtx = lastPreviewCanvas.getContext("2d");
        pCtx.drawImage($captureCanvas, 0, 0);
      }

      // Mini flash on each capture
      const mf = document.createElement("div");
      mf.className = "mini-flash";
      document.querySelector(".camera-container").appendChild(mf);
      setTimeout(() => mf.remove(), 150);

      // Run inference
      const hb = await inferSingle(imageData);
      predictions.push(hb);

      // Update UI
      updateScanUI(i + 1, MULTI_FRAME_COUNT, hb);

      // Wait between frames
      if (i < MULTI_FRAME_COUNT - 1) {
        await sleep(FRAME_INTERVAL_MS);
      }
    }

    if (scanAborted) {
      // Cancelled — silently return
      return;
    }

    if (predictions.length < MIN_VALID_FRAMES) {
      const msg = invalidFrames >= MULTI_FRAME_COUNT
        ? "No usable frames captured. Camera may be blocked/too dark/overexposed."
        : `Too few usable frames (${predictions.length}/${MULTI_FRAME_COUNT}). Please hold steady and retry.`;
      alert(msg);
      return;
    }

    // IQR outlier rejection
    const { kept, rejected } = filterOutliersIQR(predictions);

    // Find indices of rejected values to mark dots
    const rejectedIdxs = [];
    const usedRejected = new Set();
    for (let i = 0; i < predictions.length; i++) {
      const val = predictions[i];
      // Check if this value is in rejected (handle duplicates)
      const rIdx = rejected.findIndex((r, ri) => r === val && !usedRejected.has(ri));
      if (rIdx !== -1) {
        rejectedIdxs.push(i);
        usedRejected.add(rIdx);
      }
    }
    markOutlierDots(predictions, rejectedIdxs);

    // Brief pause to show outlier markings
    await sleep(400);

    // Compute final result
    const finalHb = median(kept);
    const sd = stdDev(kept);
    const range = kept.length > 0
      ? (Math.max(...kept) - Math.min(...kept))
      : 0;

    // Use preview canvas or fallback
    if (!lastPreviewCanvas) {
      lastPreviewCanvas = document.createElement("canvas");
      lastPreviewCanvas.width = crop.w;
      lastPreviewCanvas.height = crop.h;
      const ctx = lastPreviewCanvas.getContext("2d");
      ctx.drawImage($captureCanvas, 0, 0);
    }

    // Show result
    showResult(lastPreviewCanvas, finalHb, {
      framesUsed: kept.length,
      framesTotal: predictions.length,
      stdDev: sd,
      range: range,
    });

  } catch (err) {
    console.error("Multi-frame inference error:", err);
    alert("Inference failed: " + err.message);
  } finally {
    $scanOverlay.classList.add("hidden");
    $guideOverlay.style.opacity = "";
    isProcessing = false;
  }
}


/** Single-image inference (for file upload) */
async function processImage(sourceCanvas) {
  if (isProcessing) return;
  resetForNewCapture();
  isProcessing = true;

  $scanLine.classList.remove("hidden");

  try {
    const resizeCanvas = document.createElement("canvas");
    resizeCanvas.width = INPUT_SIZE;
    resizeCanvas.height = INPUT_SIZE;
    const rCtx = resizeCanvas.getContext("2d");
    rCtx.drawImage(sourceCanvas, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const imageData = rCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

    const validity = isFrameValid(imageData);
    if (!validity.ok) {
      alert("This image looks unusable (too dark/bright or blank). Please upload a clearer nail photo.");
      return;
    }

    const hb = await inferSingle(imageData);
    showResult(sourceCanvas, hb, null);

  } catch (err) {
    console.error("Inference error:", err);
    alert("Inference failed: " + err.message);
  } finally {
    $scanLine.classList.add("hidden");
    isProcessing = false;
  }
}


// ─── Result Display ───

function showResult(sourceCanvas, hbValue, stats) {
  // Clamp to reasonable range
  const hb = Math.max(2, Math.min(20, hbValue));

  // Draw preview
  $previewCanvas.width = sourceCanvas.width;
  $previewCanvas.height = sourceCanvas.height;
  const pCtx = $previewCanvas.getContext("2d");
  pCtx.drawImage(sourceCanvas, 0, 0);

  // Animate number
  animateValue($hbValue, 0, hb, 800);

  // Severity classification
  const severity = SEVERITY.find(s => hb >= s.min && hb < s.max) || SEVERITY[3];
  $severityBadge.textContent = severity.label;
  $severityBadge.className = "severity-badge " + severity.cls;

  // Gauge position: map Hb 0-18 -> 0-100%
  const pct = Math.max(0, Math.min(100, (hb / 18) * 100));
  setTimeout(() => { $gaugeMarker.style.left = pct + "%"; }, 50);

  // Color the number based on severity
  const colorMap = {
    severe:   "var(--red-400)",
    moderate: "var(--orange-500)",
    mild:     "var(--yellow-500)",
    normal:   "var(--green-500)",
  };
  $hbValue.style.color = colorMap[severity.cls] || "#fff";

  // Multi-frame stats (if available)
  if (stats && $multiFrameStats) {
    $statFramesUsed.textContent = stats.framesUsed + "/" + stats.framesTotal;
    $statStdDev.textContent = stats.stdDev.toFixed(2);
    $statRange.textContent = stats.range.toFixed(2);
    $multiFrameStats.classList.remove("hidden");
  } else if ($multiFrameStats) {
    $multiFrameStats.classList.add("hidden");
  }

  // Show result panel
  $resultSection.classList.remove("hidden");
}


function animateValue(el, start, end, duration) {
  const startTime = performance.now();
  const decimals = 1;

  function update(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = start + (end - start) * eased;
    el.textContent = current.toFixed(decimals);
    if (progress < 1) requestAnimationFrame(update);
  }

  requestAnimationFrame(update);
}


function hideResult() {
  $resultSection.classList.add("hidden");
  $gaugeMarker.style.left = "50%";
  $hbValue.textContent = "--";
  $hbValue.style.color = "#fff";
  if ($multiFrameStats) $multiFrameStats.classList.add("hidden");
}


// ─── File Upload ───

function handleFileUpload(file) {
  if (!file || !file.type.startsWith("image/")) return;

  setInstructionMode("upload");

  resetForNewCapture();

  const img = new Image();
  img.onload = () => {
    // Draw to canvas
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);

    processImage(canvas);
    URL.revokeObjectURL(img.src);
  };
  img.src = URL.createObjectURL(file);
}


// ─── Event Listeners ───

$captureBtn.addEventListener("click", captureMultiFrame);

$scanCancelBtn.addEventListener("click", () => {
  scanAborted = true;
});

$switchBtn.addEventListener("click", () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  flashOn = false;
  $flashBtn.classList.remove("flash-on");
  setInstructionMode("camera");
  startCamera();
  checkFlashSupport();
});

$uploadBtn.addEventListener("click", () => $fileInput.click());
$fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleFileUpload(e.target.files[0]);
  e.target.value = ""; // reset so same file can be re-selected
});

$closeResult.addEventListener("click", hideResult);
$retakeBtn.addEventListener("click", () => {
  setInstructionMode("camera");
  hideResult();
});

// Keyboard shortcut: Space to capture
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && $resultSection.classList.contains("hidden") && !isProcessing) {
    e.preventDefault();
    captureMultiFrame();
  }
  if (e.code === "Escape") {
    if (!$scanOverlay.classList.contains("hidden")) {
      scanAborted = true;
    } else {
      hideResult();
    }
  }
});


// ─── Flash / Torch ───

async function checkFlashSupport() {
  try {
    if (!currentStream) return;
    const track = currentStream.getVideoTracks()[0];
    if (!track) return;
    const caps = track.getCapabilities ? track.getCapabilities() : {};
    if (caps.torch) {
      $flashBtn.classList.remove("flash-unsupported");
    } else {
      $flashBtn.classList.add("flash-unsupported");
    }
  } catch {
    $flashBtn.classList.add("flash-unsupported");
  }
}

async function toggleFlash() {
  try {
    if (!currentStream) return;
    const track = currentStream.getVideoTracks()[0];
    if (!track) return;

    flashOn = !flashOn;
    await track.applyConstraints({ advanced: [{ torch: flashOn }] });
    $flashBtn.classList.toggle("flash-on", flashOn);
  } catch (err) {
    console.warn("Torch not supported:", err.message);
    $flashBtn.classList.add("flash-unsupported");
    flashOn = false;
  }
}

$flashBtn.addEventListener("click", toggleFlash);


// ─── Utility ───

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


// ─── Bootstrap ───

setInstructionMode("camera");
loadModel();
