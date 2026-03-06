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
const COLOR_MODEL_PATH = "model/hemolens_hybrid_color.onnx";
const BACKBONE_PATH = "model/hemolens_backbone.onnx";
const CATBOOST_PATH = "model/hemolens_catboost_head.onnx";
const DETECTOR_PATH = "model/nail_detector.onnx";
const INPUT_SIZE = 224;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];
const MULTI_FRAME_COUNT = 30;
const FRAME_INTERVAL_MS = 80;  // ~12.5 fps capture rate

// Inference options
let useFlipTTA = true; // average original + horizontally-flipped inference

// Debug / visualization
let debugEnabled = false;
const DEBUG_THUMBS_MAX = 12;

// Auto-capture when a nail is held with high confidence
const AUTO_CAPTURE_CONF        = 0.72; // min confidence to count toward auto-capture
const AUTO_CAPTURE_HOLD_FRAMES = 7;    // consecutive live-detect frames required (~1 s at 6 fps)
let   _highConfConsec          = 0;    // running counter of consecutive high-conf frames

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

// Prediction sanity gates — reject out-of-distribution / nonsensical results
const PREDICTION_SANITY = {
  minPlausible: 2.0,     // Hb below 2 g/dL is not compatible with life
  maxPlausible: 20.0,    // Hb above 20 g/dL is physiologically impossible
  maxStdDev:    3.0,     // if inter-frame std dev exceeds this, predictions are unreliable
  maxRange:     8.0,     // if inter-frame range exceeds this, predictions are unreliable
};


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

// Low-light warning banner
const $lowLightBanner   = document.getElementById("lowLightBanner");
const $lowLightFlashBtn = document.getElementById("lowLightFlashBtn");
const $lowLightDismiss  = document.getElementById("lowLightDismiss");

// Multi-frame scanning overlay
const $scanOverlay      = document.getElementById("scanOverlay");
const $scanRingProgress = document.getElementById("scanRingProgress");
const $scanFrameNum     = document.getElementById("scanFrameNum");
const $scanDots         = document.getElementById("scanDots");
const $scanCancelBtn    = document.getElementById("scanCancelBtn");
const $scanThumbs        = document.getElementById("scanThumbs");

// Multi-frame stats
const $multiFrameStats  = document.getElementById("multiFrameStats");
const $statFramesUsed   = document.getElementById("statFramesUsed");
const $statStdDev       = document.getElementById("statStdDev");
const $statRange        = document.getElementById("statRange");
const $detectCanvas     = document.getElementById("detectCanvas");

// Debug elements (optional)
const $debugEnable     = document.getElementById("debugEnable");
const $ttaToggle       = document.getElementById("ttaToggle");
const $debugFramesGrid = document.getElementById("debugFramesGrid");


// ─── State ───

let session = null;          // ONNX Runtime inference session (CNN-only, fallback)
let colorSession = null;     // ONNX Runtime inference session (CNN + color Ridge)
let colorModelAvailable = false; // whether the dual-input colour model is loaded
let backboneSession = null;  // ONNX Runtime inference session (backbone → 1280-d features)
let catboostSession = null;  // ONNX Runtime inference session (CatBoost head 1347 → Hb)
let catboostAvailable = false; // whether the two-stage CatBoost model is loaded
let catboostInputName = "features";  // ONNX input name for CatBoost head
let catboostOutputName = "predictions"; // ONNX output name for CatBoost head
let nailDetector = null;     // NailDetector instance (YOLOv8-nano)
let detectorAvailable = false; // whether nail_detector.onnx loaded successfully
let currentStream = null;    // MediaStream
let facingMode = "environment";  // prefer rear camera
let isProcessing = false;
let flashOn = false;         // torch state
let scanAborted = false;     // cancel flag for multi-frame
let lightMonitorId = null;   // setInterval handle for ambient light check
let lowLightDismissed = false; // user dismissed the banner this session
let _liveDetectRunning = false; // live detection overlay loop active
let _liveDetectRAF = null;      // requestAnimationFrame handle
let _liveDetectLastRun = 0;     // timestamp of last detection run
const LIVE_DETECT_INTERVAL_MS = 120; // ~8 fps


// ─── Instruction Text ───

const INSTRUCTIONS = {
  camera: detectorAvailable
    ? "Point camera at your hand — nails will be detected automatically"
    : "Place one fingernail inside the frame — fill the box as much as possible",
  cameraDetector: "Point camera at your hand — nails will be detected automatically",
  cameraManual: "Place one fingernail inside the frame — fill the box as much as possible",
  upload: "Uploaded image — nails will be detected automatically. Use a well-lit photo of your hand.",
  uploadManual: "Uploaded image mode — analyzing one photo. Use a sharp, well-lit close-up of one fingernail.",
};

function setInstructionMode(mode) {
  if (!$instructionText) return;
  let txt;
  if (mode === "camera") {
    txt = detectorAvailable ? INSTRUCTIONS.cameraDetector : INSTRUCTIONS.cameraManual;
  } else if (mode === "upload") {
    txt = detectorAvailable ? INSTRUCTIONS.upload : INSTRUCTIONS.uploadManual;
  } else {
    txt = INSTRUCTIONS.cameraManual;
  }
  $instructionText.textContent = txt;
}


// ─── Live Detection Overlay ───

/** Start the nail detection overlay loop (no-op if detector not available). */
function startLiveDetect() {
  if (!detectorAvailable || !$detectCanvas) return;
  if (_liveDetectRunning) return; // already running
  _liveDetectRunning = true;
  _liveDetectRAF = requestAnimationFrame(_liveDetectTick);
}

/** Stop the detection overlay loop and clear the canvas. */
function stopLiveDetect() {
  _liveDetectRunning = false;
  _highConfConsec = 0;
  if (_liveDetectRAF !== null) {
    cancelAnimationFrame(_liveDetectRAF);
    _liveDetectRAF = null;
  }
  if ($detectCanvas) {
    const ctx = $detectCanvas.getContext("2d");
    ctx.clearRect(0, 0, $detectCanvas.width, $detectCanvas.height);
  }
}

function _liveDetectTick(now) {
  if (!_liveDetectRunning) return;
  if (now - _liveDetectLastRun >= LIVE_DETECT_INTERVAL_MS) {
    _liveDetectLastRun = now;
    _runLiveDetect(); // fire-and-forget
  }
  _liveDetectRAF = requestAnimationFrame(_liveDetectTick);
}

async function _runLiveDetect() {
  if (!_liveDetectRunning) return;
  // Pause during active scanning — inference loop is already calling the detector
  if (isProcessing) return;

  if (!$video || $video.readyState < 2 || !$video.videoWidth) return;
  if (!$detectCanvas) return;

  // Use the canvas's actual CSS-rendered size (never its intrinsic buffer size)
  const rect = $detectCanvas.getBoundingClientRect();
  const cw = Math.round(rect.width);
  const ch = Math.round(rect.height);
  if (!cw || !ch) return;

  // Sync pixel buffer to displayed size only when it changes
  if ($detectCanvas.width !== cw || $detectCanvas.height !== ch) {
    $detectCanvas.width  = cw;
    $detectCanvas.height = ch;
  }

  const ctx = $detectCanvas.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);

  // Hide boxes when the result panel is showing (camera is covered)
  if (!$resultSection.classList.contains("hidden")) return;

  try {
    const { imageData, width: vw, height: vh } = grabFullFrame();
    const detections = await nailDetector.detect(imageData, vw, vh);

    if (!_liveDetectRunning || isProcessing) return; // aborted while awaiting

    // Recompute in case the canvas was resized while awaiting
    const rect2 = $detectCanvas.getBoundingClientRect();
    const cw2 = Math.round(rect2.width);
    const ch2 = Math.round(rect2.height);
    if (!cw2 || !ch2 || $detectCanvas.width !== cw2 || $detectCanvas.height !== ch2) return;

    // Map video pixels → displayed CSS pixels for object-fit: cover
    // Cover: video fills the container, excess is clipped (no letterboxing visible)
    const videoAspect = vw / vh;
    const containerAspect = cw2 / ch2;
    let scale, offX, offY;
    if (containerAspect > videoAspect) {
      // Container is wider → scale to fill width, clip top/bottom
      scale = cw2 / vw;
      offX  = 0;
      offY  = (ch2 - vh * scale) / 2;
    } else {
      // Container is taller → scale to fill height, clip left/right
      scale = ch2 / vh;
      offX  = (cw2 - vw * scale) / 2;
      offY  = 0;
    }

    _drawDetectionBoxes(ctx, detections, scale, offX, offY, cw2, ch2);

    // ── Auto-capture: start scan immediately when nail confidence stays high ──
    const nailHits  = detections.filter(d => d.className === "nail");
    const bestConf  = nailHits.length > 0 ? nailHits[0].confidence : 0;
    if (!isProcessing && bestConf >= AUTO_CAPTURE_CONF) {
      _highConfConsec++;

      // Draw a bottom progress bar showing auto-capture imminence
      const pct = Math.min(_highConfConsec / AUTO_CAPTURE_HOLD_FRAMES, 1);
      const barH = 5;
      ctx.save();
      ctx.fillStyle = "rgba(52,211,153,0.25)";
      ctx.fillRect(0, ch2 - barH, cw2, barH);
      ctx.fillStyle = "#34d399";
      ctx.fillRect(0, ch2 - barH, cw2 * pct, barH);
      ctx.restore();

      if (_highConfConsec >= AUTO_CAPTURE_HOLD_FRAMES) {
        _highConfConsec = 0;
        captureMultiFrame();
      }
    } else {
      _highConfConsec = 0;
    }
  } catch (_) {
    // Silently ignore transient errors (model warming up, frame grab race, etc.)
  }
}

/** Draw a rounded rect, with fallback for older browsers. */
function _rrect(ctx, x, y, w, h, r) {
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x, y, w, h, r);
  } else {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y,     x + w, y + r,     r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x,     y + h, x,     y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x,     y,     x + r, y,         r);
    ctx.closePath();
  }
}

/**
 * Render detection boxes and a status badge onto the overlay canvas.
 * detections = [{ x, y, w, h, confidence, className }, ...] in video pixels.
 */
function _drawDetectionBoxes(ctx, detections, scale, offX, offY, cw, ch) {
  const nails = detections.filter(d => d.className === "nail");
  const skins = detections.filter(d => d.className === "skin");

  // ── Skin boxes: dim dashed outline ──
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "rgba(147, 197, 253, 0.40)";
  ctx.lineWidth = 1.5;
  for (const d of skins) {
    ctx.strokeRect(d.x * scale + offX, d.y * scale + offY, d.w * scale, d.h * scale);
  }
  ctx.restore();

  // ── Nail boxes: prominent green ──
  ctx.save();
  ctx.setLineDash([]);
  ctx.lineCap = "round";
  for (const d of nails) {
    const dx = d.x * scale + offX;
    const dy = d.y * scale + offY;
    const dw = d.w * scale;
    const dh = d.h * scale;
    const pct = Math.round(d.confidence * 100);

    // Translucent fill
    ctx.fillStyle = "rgba(52, 211, 153, 0.08)";
    ctx.fillRect(dx, dy, dw, dh);

    // Box outline
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 2;
    ctx.strokeRect(dx, dy, dw, dh);

    // Corner bracket marquees
    const bk = Math.min(14, dw * 0.22, dh * 0.22);
    ctx.strokeStyle = "#6ee7b7";
    ctx.lineWidth = 2.5;
    // TL
    ctx.beginPath(); ctx.moveTo(dx,        dy + bk); ctx.lineTo(dx,        dy);        ctx.lineTo(dx + bk,        dy);        ctx.stroke();
    // TR
    ctx.beginPath(); ctx.moveTo(dx + dw - bk, dy);        ctx.lineTo(dx + dw,   dy);        ctx.lineTo(dx + dw,   dy + bk);        ctx.stroke();
    // BL
    ctx.beginPath(); ctx.moveTo(dx,        dy + dh - bk); ctx.lineTo(dx,        dy + dh); ctx.lineTo(dx + bk,        dy + dh); ctx.stroke();
    // BR
    ctx.beginPath(); ctx.moveTo(dx + dw - bk, dy + dh);   ctx.lineTo(dx + dw,   dy + dh); ctx.lineTo(dx + dw,   dy + dh - bk); ctx.stroke();

    // Confidence label badge just above (or below) the box
    const lbl = `Nail \u00b7 ${pct}%`;
    ctx.font = "bold 12px Inter, sans-serif";
    const tw = ctx.measureText(lbl).width;
    const bW = tw + 12;
    const bH = 20;
    const bx = dx;
    const by = dy > bH + 4 ? dy - bH - 4 : dy + dh + 4;
    ctx.fillStyle = "#065f46";
    ctx.beginPath();
    _rrect(ctx, bx, by, bW, bH, 4);
    ctx.fill();
    ctx.fillStyle = "#6ee7b7";
    ctx.fillText(lbl, bx + 6, by + bH - 5);
  }
  ctx.restore();

  // ── Status badge — top-right corner ──
  const statusLabel = nails.length > 0
    ? (nails.length === 1 ? "\u2713 Nail detected" : `\u2713 ${nails.length} nails`)
    : "No nail detected";
  const statusBg = nails.length > 0 ? "rgba(5, 78, 50, 0.88)"   : "rgba(120, 20, 20, 0.80)";
  const statusFg = nails.length > 0 ? "#6ee7b7"                  : "#fca5a5";

  ctx.save();
  ctx.font = "bold 13px Inter, sans-serif";
  const sbw = ctx.measureText(statusLabel).width + 20;
  const sbh = 26;
  const sbx = cw - sbw - 10;
  const sby = 10;
  ctx.fillStyle = statusBg;
  ctx.beginPath();
  _rrect(ctx, sbx, sby, sbw, sbh, 6);
  ctx.fill();
  ctx.fillStyle = statusFg;
  ctx.fillText(statusLabel, sbx + 10, sby + 18);
  ctx.restore();
}


// ─── Ambient Light Monitoring ───

const LIGHT_CHECK_INTERVAL_MS = 1500;  // how often we sample (ms)
const LOW_LIGHT_LUMA_THRESHOLD = 0.13; // mean luma below this => warn
const LOW_LIGHT_CONSEC_NEEDED  = 2;    // consecutive low readings before showing banner
let   _lowLightConsec = 0;             // rolling counter

// Training-data photo profile (from analyze_training_photos.py)
// Used to give the user a live quality indicator
const TRAINING_PROFILE = {
  luma:   { min: 0.55, max: 0.76 },   // nail crop luma [0-1] P5-P95 (142-194 /255)
  rbRatio:{ min: 1.18, max: 1.49 },   // R/B channel ratio — warm neutral light
  sat:    { min: 0.16, max: 0.33 },   // saturation [0-1] P5-P95 (41-85 /255)
};

/** Sample a centre crop from the live video feed and return luma + color stats. */
function sampleVideoStats() {
  if (!$video || $video.readyState < 2 || !$video.videoWidth) return null;

  // Sample a small 64×64 centre crop for speed
  const size = 64;
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = size;
  tmpCanvas.height = size;
  const ctx = tmpCanvas.getContext("2d");

  const vw = $video.videoWidth;
  const vh = $video.videoHeight;
  const sx = Math.floor((vw - size) / 2);
  const sy = Math.floor((vh - size) / 2);
  ctx.drawImage($video, sx, sy, size, size, 0, 0, size, size);

  const imgData = ctx.getImageData(0, 0, size, size);
  const lumaStats = computeLumaStats(imgData);

  // Also compute R/B ratio and saturation from the sampled patch
  const d = imgData.data;
  const n = imgData.width * imgData.height;
  let rSum = 0, gSum = 0, bSum = 0, satSum = 0;
  for (let i = 0; i < n; i++) {
    const r = d[i * 4], g = d[i * 4 + 1], b = d[i * 4 + 2];
    rSum += r; gSum += g; bSum += b;
    const mx = Math.max(r, g, b);
    const mn = Math.min(r, g, b);
    satSum += (mx === 0) ? 0 : (mx - mn) / mx;
  }
  const rMean = rSum / n;
  const bMean = bSum / n;

  return {
    mean: lumaStats.mean,
    rbRatio: bMean > 0 ? rMean / bMean : 1,
    saturation: satSum / n,
  };
}

/** Start periodic light level checks on the camera feed. */
function startLightMonitor() {
  stopLightMonitor();
  _lowLightConsec = 0;
  lowLightDismissed = false;

  lightMonitorId = setInterval(() => {
    // Don't check while scanning or showing results
    if (isProcessing || !$resultSection.classList.contains("hidden")) return;
    if (lowLightDismissed) return;

    const stats = sampleVideoStats();
    if (stats === null) return;

    // Low-light detection
    if (stats.mean < LOW_LIGHT_LUMA_THRESHOLD) {
      _lowLightConsec++;
      if (_lowLightConsec >= LOW_LIGHT_CONSEC_NEEDED) {
        showLowLightBanner();
      }
    } else {
      _lowLightConsec = 0;
      hideLowLightBanner();
    }

    // Live quality indicator
    updateLightingQuality(stats);
  }, LIGHT_CHECK_INTERVAL_MS);
}

function stopLightMonitor() {
  if (lightMonitorId !== null) {
    clearInterval(lightMonitorId);
    lightMonitorId = null;
  }
}

/** Update the live lighting quality indicator chip. */
function updateLightingQuality(stats) {
  const $indicator = document.getElementById("lightingQuality");
  if (!$indicator) return;

  const issues = [];

  // Brightness check against training profile
  if (stats.mean < TRAINING_PROFILE.luma.min) {
    issues.push("too dark");
  } else if (stats.mean > TRAINING_PROFILE.luma.max) {
    issues.push("too bright");
  }

  // Color temperature check (R/B ratio)
  if (stats.rbRatio < TRAINING_PROFILE.rbRatio.min) {
    issues.push("too blue/cool");
  } else if (stats.rbRatio > TRAINING_PROFILE.rbRatio.max) {
    issues.push("too warm/yellow");
  }

  // Saturation check
  if (stats.saturation < TRAINING_PROFILE.sat.min) {
    issues.push("low color");
  } else if (stats.saturation > TRAINING_PROFILE.sat.max) {
    issues.push("oversaturated");
  }

  if (issues.length === 0) {
    $indicator.textContent = "✓ Lighting looks good";
    $indicator.className = "lighting-quality quality-good";
  } else if (issues.length === 1) {
    $indicator.textContent = "⚠ " + issues[0];
    $indicator.className = "lighting-quality quality-warn";
  } else {
    $indicator.textContent = "✗ " + issues.slice(0, 2).join(", ");
    $indicator.className = "lighting-quality quality-bad";
  }
}

function showLowLightBanner() {
  if (!$lowLightBanner) return;
  // Hide the flash CTA if torch isn't available
  if ($flashBtn.classList.contains("flash-unsupported")) {
    $lowLightFlashBtn.style.display = "none";
  } else {
    $lowLightFlashBtn.style.display = "";
  }
  $lowLightBanner.classList.remove("hidden");
}

function hideLowLightBanner() {
  if (!$lowLightBanner) return;
  $lowLightBanner.classList.add("hidden");
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
  hideLowLightBanner();
  _highConfConsec = 0;  // clear auto-capture counter

  // Reset scan overlay state
  scanAborted = false;
  $scanDots.innerHTML = "";
  $scanFrameNum.textContent = "0";
  $scanRingProgress.style.strokeDashoffset = 2 * Math.PI * 52;
  $scanOverlay.classList.add("hidden");
  if ($guideOverlay) $guideOverlay.style.opacity = "";

  // Reset debug UI
  if ($debugFramesGrid) $debugFramesGrid.innerHTML = "";
  if ($scanThumbs) {
    $scanThumbs.innerHTML = "";
    $scanThumbs.classList.add("hidden");
  }
}


// ─── Debug Wiring ───

function initDebugControls() {
  if ($debugEnable) {
    debugEnabled = Boolean($debugEnable.checked);
    $debugEnable.addEventListener("change", (e) => {
      debugEnabled = Boolean(e.target.checked);
      if ($scanThumbs) {
        if (debugEnabled) $scanThumbs.classList.remove("hidden");
        else $scanThumbs.classList.add("hidden");
      }
    });
  }
  if ($ttaToggle) {
    useFlipTTA = Boolean($ttaToggle.checked);
    $ttaToggle.addEventListener("change", (e) => {
      useFlipTTA = Boolean(e.target.checked);
    });
  }

  if ($scanThumbs) {
    if (debugEnabled) $scanThumbs.classList.remove("hidden");
    else $scanThumbs.classList.add("hidden");
  }
}


// ─── Model Loading ───

async function loadModel() {
  try {
    updateSplash("Initializing runtime...", 10);

    // Point WASM binaries to CDN explicitly
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";

    // Single-threaded to avoid SharedArrayBuffer / COOP/COEP requirements
    ort.env.wasm.numThreads = 1;

    updateSplash("Downloading Hb model (~10 MB)...", 20);

    // Fetch Hb model as ArrayBuffer for reliable loading
    const response = await fetch(MODEL_PATH);
    if (!response.ok) {
      throw new Error("Model fetch failed: HTTP " + response.status);
    }
    const modelBuffer = await response.arrayBuffer();
    console.log("Hb model downloaded:", modelBuffer.byteLength, "bytes");

    updateSplash("Loading Hb inference engine...", 50);

    session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    console.log("Hb session created. Inputs:", session.inputNames, "Outputs:", session.outputNames);

    // Try loading two-stage CatBoost model (backbone + CatBoost head)
    // Priority: CatBoost (MAE≈1.30) > Ridge+Color (MAE≈1.46) > CNN-only
    updateSplash("Loading CatBoost model...", 58);
    try {
      const [bbResp, cbResp] = await Promise.all([
        fetch(BACKBONE_PATH),
        fetch(CATBOOST_PATH),
      ]);
      if (!bbResp.ok) throw new Error("Backbone HTTP " + bbResp.status);
      if (!cbResp.ok) throw new Error("CatBoost HTTP " + cbResp.status);
      const [bbBuf, cbBuf] = await Promise.all([
        bbResp.arrayBuffer(),
        cbResp.arrayBuffer(),
      ]);
      const ortOpts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
      backboneSession = await ort.InferenceSession.create(bbBuf, ortOpts);
      catboostSession = await ort.InferenceSession.create(cbBuf, ortOpts);
      catboostInputName = catboostSession.inputNames[0];
      catboostOutputName = catboostSession.outputNames[0];
      catboostAvailable = true;
      console.log("[CatBoost] Loaded. Backbone inputs:", backboneSession.inputNames,
                  "CatBoost input:", catboostInputName, "output:", catboostOutputName);
    } catch (cbErr) {
      console.warn("[CatBoost] Not available:", cbErr.message);
      catboostAvailable = false;
    }

    // Try loading the dual-input colour model (non-blocking — fall back to CNN-only)
    updateSplash("Loading colour model...", 64);
    try {
      const colorResp = await fetch(COLOR_MODEL_PATH);
      if (!colorResp.ok) throw new Error("HTTP " + colorResp.status);
      const colorBuf = await colorResp.arrayBuffer();
      colorSession = await ort.InferenceSession.create(colorBuf, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      colorModelAvailable = true;
      console.log("[ColorModel] Loaded. Inputs:", colorSession.inputNames, "Outputs:", colorSession.outputNames);
    } catch (colorErr) {
      console.warn("[ColorModel] Not available, using CNN-only model:", colorErr.message);
      colorModelAvailable = false;
    }

    // Try loading the nail detector (non-blocking — fallback to guide overlay if absent)
    updateSplash("Loading nail detector...", 78);
    try {
      nailDetector = new NailDetector({ confThreshold: 0.35, iouThreshold: 0.45 });
      await nailDetector.load(DETECTOR_PATH);
      detectorAvailable = true;
      console.log("[NailDetector] Loaded successfully — auto-detection enabled");
    } catch (detErr) {
      console.warn("[NailDetector] Not available, falling back to manual guide overlay:", detErr.message);
      detectorAvailable = false;
    }

    updateSplash("Ready!", 100);

    // Short delay so user sees the success state
    await sleep(600);

    // Transition to app
    $splash.classList.add("fade-out");
    setTimeout(() => {
      $splash.classList.add("hidden");
      $app.classList.remove("hidden");
      initDebugControls();

      // Hide the guide overlay if detector is available
      if (detectorAvailable && $guideOverlay) {
        $guideOverlay.style.display = "none";
      }

      startCamera().then(() => {
        checkFlashSupport();
        startLightMonitor();
        startLiveDetect();
      });
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

  // Match training/val preprocessing:
  // Resize (shorter side = 256, preserve aspect) -> CenterCrop 224
  return preprocessCanvasToImageData($captureCanvas);
}


/** Grab full frame from video as ImageData (for nail detector) */
function grabFullFrame() {
  const vw = $video.videoWidth;
  const vh = $video.videoHeight;
  const canvas = document.createElement("canvas");
  canvas.width = vw;
  canvas.height = vh;
  const ctx = canvas.getContext("2d");
  ctx.drawImage($video, 0, 0, vw, vh);
  return { canvas, imageData: ctx.getImageData(0, 0, vw, vh), width: vw, height: vh };
}


/** Crop a detected nail bounding box from a canvas, preprocess for Hb model */
function cropDetectedNail(srcCanvas, detection) {
  // Add 10% padding around the detection box
  const pad = 0.10;
  const px = Math.max(0, Math.round(detection.x - detection.w * pad));
  const py = Math.max(0, Math.round(detection.y - detection.h * pad));
  const pw = Math.min(Math.round(detection.w * (1 + 2 * pad)), srcCanvas.width - px);
  const ph = Math.min(Math.round(detection.h * (1 + 2 * pad)), srcCanvas.height - py);

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = pw;
  cropCanvas.height = ph;
  const ctx = cropCanvas.getContext("2d");
  ctx.drawImage(srcCanvas, px, py, pw, ph, 0, 0, pw, ph);

  return preprocessCanvasToImageData(cropCanvas);
}


/** Pick the best nail detection from a list (highest confidence, class=nail) */
function pickBestNail(detections) {
  const nails = detections.filter(d => d.className === "nail");
  if (nails.length === 0) return null;
  // Pick the one with largest area × confidence product (prefer big, confident nails)
  nails.sort((a, b) => (b.w * b.h * b.confidence) - (a.w * a.h * a.confidence));
  return nails[0];
}


// ─── Preprocessing (match training) ───

function resizeShortestSide(srcCanvas, targetShortSide) {
  const sw = srcCanvas.width;
  const sh = srcCanvas.height;
  const shortSide = Math.min(sw, sh);
  if (!shortSide) return srcCanvas;

  const scale = targetShortSide / shortSide;
  const dw = Math.max(1, Math.round(sw * scale));
  const dh = Math.max(1, Math.round(sh * scale));

  const out = document.createElement("canvas");
  out.width = dw;
  out.height = dh;
  const octx = out.getContext("2d");
  octx.drawImage(srcCanvas, 0, 0, dw, dh);
  return out;
}

function centerCrop(srcCanvas, cropSize) {
  const out = document.createElement("canvas");
  out.width = cropSize;
  out.height = cropSize;
  const octx = out.getContext("2d");

  const sx = Math.max(0, Math.floor((srcCanvas.width - cropSize) / 2));
  const sy = Math.max(0, Math.floor((srcCanvas.height - cropSize) / 2));
  octx.drawImage(srcCanvas, sx, sy, cropSize, cropSize, 0, 0, cropSize, cropSize);
  return out;
}

function preprocessCanvasToImageData(srcCanvas) {
  const resized = resizeShortestSide(srcCanvas, 256);
  const cropped = centerCrop(resized, INPUT_SIZE);
  const ctx = cropped.getContext("2d");
  return ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
}


/**
 * Run inference on a single ImageData → Hb prediction.
 *
 * Model priority (highest accuracy first):
 *   1. CatBoost two-stage (backbone → features → CatBoost)  MAE≈1.30
 *   2. Ridge + colour dual-input ONNX                       MAE≈1.46
 *   3. CNN-only single-input ONNX (fallback)
 *
 * For CatBoost: TTA averages *features* before the CatBoost head,
 * matching training where each crop's feature was TTA-averaged.
 */
async function inferSingle(imageData, colorFeatures) {
  const useCatBoost  = catboostAvailable && backboneSession && catboostSession && colorFeatures;
  const useColorModel = !useCatBoost && colorModelAvailable && colorSession && colorFeatures;

  function toTensorData(flipX) {
    const pixels = imageData.data;
    const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    const pixelCount = INPUT_SIZE * INPUT_SIZE;

    for (let y = 0; y < INPUT_SIZE; y++) {
      for (let x = 0; x < INPUT_SIZE; x++) {
        const dx = x;
        const sx = flipX ? (INPUT_SIZE - 1 - x) : x;
        const di = y * INPUT_SIZE + dx;
        const si = y * INPUT_SIZE + sx;
        const p = si * 4;
        const r = pixels[p]     / 255.0;
        const g = pixels[p + 1] / 255.0;
        const b = pixels[p + 2] / 255.0;

        float32Data[0 * pixelCount + di] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        float32Data[1 * pixelCount + di] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        float32Data[2 * pixelCount + di] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
      }
    }
    return float32Data;
  }

  // ── CatBoost two-stage path ──
  if (useCatBoost) {
    async function backboneFeatures(flipX) {
      const t = new ort.Tensor("float32", toTensorData(flipX), [1, 3, INPUT_SIZE, INPUT_SIZE]);
      const r = await backboneSession.run({ image: t });
      return r.features.data;                       // Float32Array(1280)
    }

    // TTA: average backbone features (matches training pipeline)
    const feat1 = await backboneFeatures(false);
    let avgFeat;
    if (useFlipTTA) {
      const feat2 = await backboneFeatures(true);
      avgFeat = new Float32Array(feat1.length);
      for (let i = 0; i < feat1.length; i++) avgFeat[i] = (feat1[i] + feat2[i]) / 2;
    } else {
      avgFeat = new Float32Array(feat1);            // copy
    }

    // Concatenate CNN features + colour features → 1347-d
    const combined = new Float32Array(avgFeat.length + colorFeatures.length);
    combined.set(avgFeat, 0);
    combined.set(colorFeatures, avgFeat.length);

    const combinedTensor = new ort.Tensor("float32", combined, [1, combined.length]);
    const feeds = {};
    feeds[catboostInputName] = combinedTensor;
    const results = await catboostSession.run(feeds);
    const hb = results[catboostOutputName].data[0];
    return typeof hb === "number" ? hb : hb[0];     // CatBoost may return [1,1]
  }

  // ── Ridge colour or CNN-only path ──
  async function runOnce(flipX) {
    const inputTensor = new ort.Tensor("float32", toTensorData(flipX), [1, 3, INPUT_SIZE, INPUT_SIZE]);

    if (useColorModel) {
      const colorTensor = new ort.Tensor("float32", colorFeatures, [1, colorFeatures.length]);
      const results = await colorSession.run({ image: inputTensor, color_features: colorTensor });
      return results.hb_prediction.data[0];
    } else {
      const results = await session.run({ image: inputTensor });
      return results.hb_prediction.data[0];
    }
  }

  const y1 = await runOnce(false);
  if (!useFlipTTA) return y1;
  const y2 = await runOnce(true);
  return (y1 + y2) / 2.0;
}


// ─── Debug Rendering Helpers ───

function imageDataToThumbCanvas(imageData, size) {
  const tmp = document.createElement("canvas");
  tmp.width = INPUT_SIZE;
  tmp.height = INPUT_SIZE;
  const tctx = tmp.getContext("2d");
  tctx.putImageData(imageData, 0, 0);

  const out = document.createElement("canvas");
  out.width = size;
  out.height = size;
  const octx = out.getContext("2d");
  octx.imageSmoothingEnabled = true;
  octx.drawImage(tmp, 0, 0, size, size);
  return out;
}

function makeFrameTile(imageData, labelText) {
  const tile = document.createElement("div");
  tile.className = "debug-frame";

  const canvas = imageDataToThumbCanvas(imageData, 64);
  tile.appendChild(canvas);

  const label = document.createElement("div");
  label.className = "debug-label";
  label.textContent = labelText;
  tile.appendChild(label);

  return { tile, label };
}

function makeScanThumb(imageData, labelText) {
  const tile = document.createElement("div");
  tile.className = "scan-thumb";

  const canvas = imageDataToThumbCanvas(imageData, 48);
  tile.appendChild(canvas);

  const label = document.createElement("div");
  label.className = "scan-thumb-label";
  label.textContent = labelText;
  tile.appendChild(label);

  return { tile, label };
}

function renderScanThumbStrip(frameTiles) {
  if (!$scanThumbs || !debugEnabled) return;
  $scanThumbs.classList.remove("hidden");
  $scanThumbs.innerHTML = "";

  const last = frameTiles.slice(-DEBUG_THUMBS_MAX);
  for (const t of last) {
    const clone = t.scanTile;
    if (clone) $scanThumbs.appendChild(clone);
  }
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

  // When detector is NOT available, fall back to guide overlay crop
  const useDetector = detectorAvailable && nailDetector && nailDetector.isLoaded;
  let crop = null;
  if (!useDetector) {
    crop = getNailCropRect();
    if (!crop) return;
  }

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
  if ($guideOverlay) $guideOverlay.style.opacity = "0.3";

  const predictions = [];
  const frameTiles = []; // one per captured frame (including invalid)
  const predIdxToFrameIdx = []; // maps prediction index -> frame index
  let invalidFrames = 0;
  let noDetectionFrames = 0; // frames where detector found no nails
  let lastPreviewCanvas = null;

  try {
    for (let i = 0; i < MULTI_FRAME_COUNT; i++) {
      if (scanAborted) {
        console.log("Scan cancelled by user at frame", i);
        break;
      }

      let imageData;

      let frameColorFeatures = null; // per-frame colour features (or null)

      if (useDetector) {
        // ── Auto-detection path: detect nails, crop the best one ──
        const frame = grabFullFrame();
        const detections = await nailDetector.detect(frame.imageData, frame.width, frame.height);
        const bestNail = pickBestNail(detections);

        if (!bestNail) {
          noDetectionFrames++;
          updateScanUI(i + 1, MULTI_FRAME_COUNT, null);

          // Create placeholder tile for debug
          const placeholderData = new ImageData(INPUT_SIZE, INPUT_SIZE);
          const dbg = $debugFramesGrid ? makeFrameTile(placeholderData, `#${i + 1} no nail`) : null;
          const scan = $scanThumbs ? makeScanThumb(placeholderData, `no nail`) : null;
          const tiles = { dbgTile: dbg?.tile, dbgLabel: dbg?.label, scanTile: scan?.tile, scanLabel: scan?.label };
          frameTiles.push(tiles);
          if (tiles.dbgTile) {
            tiles.dbgTile.classList.add("invalid");
            $debugFramesGrid.appendChild(tiles.dbgTile);
          }
          renderScanThumbStrip(frameTiles);

          const framesLeft = (MULTI_FRAME_COUNT - 1) - i;
          const maxPossible = predictions.length + framesLeft;
          if (maxPossible < MIN_VALID_FRAMES) break;
          if (i < MULTI_FRAME_COUNT - 1) await sleep(FRAME_INTERVAL_MS);
          continue;
        }

        // Crop and preprocess the detected nail
        imageData = cropDetectedNail(frame.canvas, bestNail);

        // ── Colour features: extract from raw nail + skin ROI pixels ──
        if (colorModelAvailable && typeof extractColorFeatures === "function") {
          try {
            const nailROI = getRoiImageData(frame.canvas, bestNail);
            const skins = detections.filter(d => d.className === "skin");
            if (skins.length > 0) {
              // Pick the largest (or most confident) skin detection
              skins.sort((a, b) => (b.w * b.h * b.confidence) - (a.w * a.h * a.confidence));
              const skinROI = getRoiImageData(frame.canvas, skins[0]);
              frameColorFeatures = extractColorFeatures(nailROI, skinROI);
            }
          } catch (cfErr) {
            console.warn("[ColorFeatures] Extraction failed for frame", i, cfErr.message);
          }
        }

        // Save preview from best detection
        if (!lastPreviewCanvas || i === Math.floor(MULTI_FRAME_COUNT / 2)) {
          const pad = 0.10;
          const px = Math.max(0, Math.round(bestNail.x - bestNail.w * pad));
          const py = Math.max(0, Math.round(bestNail.y - bestNail.h * pad));
          const pw = Math.min(Math.round(bestNail.w * (1 + 2 * pad)), frame.width - px);
          const ph = Math.min(Math.round(bestNail.h * (1 + 2 * pad)), frame.height - py);
          lastPreviewCanvas = document.createElement("canvas");
          lastPreviewCanvas.width = pw;
          lastPreviewCanvas.height = ph;
          const pCtx = lastPreviewCanvas.getContext("2d");
          pCtx.drawImage(frame.canvas, px, py, pw, ph, 0, 0, pw, ph);
        }

      } else {
        // ── Manual guide overlay path (original behavior) ──
        imageData = grabNailFrame(crop);

        // Save a preview canvas from the first valid frame, then refine near the middle
        if (!lastPreviewCanvas || i === Math.floor(MULTI_FRAME_COUNT / 2)) {
          lastPreviewCanvas = document.createElement("canvas");
          lastPreviewCanvas.width = crop.w;
          lastPreviewCanvas.height = crop.h;
          const pCtx = lastPreviewCanvas.getContext("2d");
          pCtx.drawImage($captureCanvas, 0, 0);
        }
      }

      // Create debug tiles early (so invalid frames are also visible)
      const dbg = $debugFramesGrid ? makeFrameTile(imageData, `#${i + 1}`) : null;
      const scan = $scanThumbs ? makeScanThumb(imageData, `#${i + 1}`) : null;
      const tiles = { dbgTile: dbg?.tile, dbgLabel: dbg?.label, scanTile: scan?.tile, scanLabel: scan?.label };
      frameTiles.push(tiles);
      if (tiles.dbgTile && $debugFramesGrid) $debugFramesGrid.appendChild(tiles.dbgTile);

      // Reject blocked/black/overexposed/too-uniform frames
      const validity = isFrameValid(imageData);
      if (!validity.ok) {
        invalidFrames++;
        updateScanUI(i + 1, MULTI_FRAME_COUNT, null);

        if (tiles.dbgTile) tiles.dbgTile.classList.add("invalid");
        if (tiles.scanTile) tiles.scanTile.classList.add("invalid");
        if (tiles.dbgLabel) tiles.dbgLabel.textContent = `#${i + 1} invalid`;
        if (tiles.scanLabel) tiles.scanLabel.textContent = "invalid";

        renderScanThumbStrip(frameTiles);

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

      // Mini flash on each capture
      const mf = document.createElement("div");
      mf.className = "mini-flash";
      document.querySelector(".camera-container").appendChild(mf);
      setTimeout(() => mf.remove(), 150);

      // Run inference (pass colour features when available)
      const hb = await inferSingle(imageData, frameColorFeatures);
      predictions.push(hb);
      predIdxToFrameIdx.push(frameTiles.length - 1);

      if (tiles.dbgLabel) tiles.dbgLabel.textContent = `#${i + 1} ${hb.toFixed(1)}`;
      if (tiles.scanLabel) tiles.scanLabel.textContent = hb.toFixed(1);

      renderScanThumbStrip(frameTiles);

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
      const msg = (invalidFrames + noDetectionFrames) >= MULTI_FRAME_COUNT
        ? useDetector
          ? "No nails detected in any frame. Make sure your fingernails are visible to the camera."
          : "No usable frames captured. Camera may be blocked/too dark/overexposed."
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

    // Mark outliers in thumbnail tiles
    for (const predIdx of rejectedIdxs) {
      const frameIdx = predIdxToFrameIdx[predIdx];
      const t = frameTiles[frameIdx];
      if (!t) continue;
      if (t.dbgTile) t.dbgTile.classList.add("outlier");
      if (t.scanTile) t.scanTile.classList.add("outlier");
    }

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

    // Sanity check: reject nonsensical predictions
    const isOutOfRange = finalHb < PREDICTION_SANITY.minPlausible || finalHb > PREDICTION_SANITY.maxPlausible;
    const isHighVariance = sd > PREDICTION_SANITY.maxStdDev || range > PREDICTION_SANITY.maxRange;

    if (isOutOfRange) {
      alert(
        `Result rejected: predicted Hb = ${finalHb.toFixed(1)} g/dL is outside the physiologically plausible range (${PREDICTION_SANITY.minPlausible}–${PREDICTION_SANITY.maxPlausible} g/dL).\n\n` +
        `This usually means the camera is not pointing at a fingernail, or the lighting conditions are too far from the training data.\n\n` +
        `Std Dev: ${sd.toFixed(1)} g/dL · Range: ${range.toFixed(1)} g/dL`
      );
      return;
    }

    // Show result (with unreliable flag if high variance)
    showResult(lastPreviewCanvas, finalHb, {
      framesUsed: kept.length,
      framesTotal: predictions.length,
      stdDev: sd,
      range: range,
      unreliable: isHighVariance,
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
    const useDetector = detectorAvailable && nailDetector && nailDetector.isLoaded;

    if (useDetector) {
      // ── Auto-detection path: detect nails, run inference on each ──
      const w = sourceCanvas.width;
      const h = sourceCanvas.height;
      const ctx = sourceCanvas.getContext("2d");
      const fullImageData = ctx.getImageData(0, 0, w, h);

      const detections = await nailDetector.detect(fullImageData, w, h);
      const nails = detections.filter(d => d.className === "nail");

      if (nails.length === 0) {
        alert("No fingernails detected in this image. Please upload a photo showing your fingernails clearly.");
        return;
      }

      console.log(`[Upload] Detected ${nails.length} nail(s), running Hb inference on each...`);

      // Compute colour features once for the upload image (use first nail + first skin)
      let uploadColorFeatures = null;
      if (colorModelAvailable && typeof extractColorFeatures === "function") {
        try {
          const skins = detections.filter(d => d.className === "skin");
          if (nails.length > 0 && skins.length > 0) {
            skins.sort((a, b) => (b.w * b.h * b.confidence) - (a.w * a.h * a.confidence));
            const nailROI = getRoiImageData(sourceCanvas, nails[0]);
            const skinROI = getRoiImageData(sourceCanvas, skins[0]);
            uploadColorFeatures = extractColorFeatures(nailROI, skinROI);
          }
        } catch (cfErr) {
          console.warn("[ColorFeatures] Extraction failed for upload:", cfErr.message);
        }
      }

      // Run inference on each detected nail and average
      const nailPredictions = [];
      for (const nail of nails) {
        const nailImageData = cropDetectedNail(sourceCanvas, nail);
        const validity = isFrameValid(nailImageData);
        if (!validity.ok) {
          console.log(`[Upload] Nail at (${nail.x.toFixed(0)}, ${nail.y.toFixed(0)}) invalid, skipping`);
          continue;
        }
        const hb = await inferSingle(nailImageData, uploadColorFeatures);
        console.log(`[Upload] Nail conf=${nail.confidence.toFixed(2)} → Hb=${hb.toFixed(1)}`);
        nailPredictions.push(hb);
      }

      if (nailPredictions.length === 0) {
        alert("Detected nails were too dark/bright/blurry. Please upload a better lit photo.");
        return;
      }

      // Use median of all nail predictions
      const finalHb = median(nailPredictions);

      // OOD check
      if (finalHb < PREDICTION_SANITY.minPlausible || finalHb > PREDICTION_SANITY.maxPlausible) {
        alert(
          `Result rejected: predicted Hb = ${finalHb.toFixed(1)} g/dL is outside the plausible range (${PREDICTION_SANITY.minPlausible}–${PREDICTION_SANITY.maxPlausible} g/dL).\n\n` +
          `This image is likely out-of-distribution. Please try with a different photo.`
        );
        return;
      }

      showResult(sourceCanvas, finalHb, nailPredictions.length > 1 ? {
        framesUsed: nailPredictions.length,
        framesTotal: nails.length,
        stdDev: stdDev(nailPredictions),
        range: nailPredictions.length > 1 ? Math.max(...nailPredictions) - Math.min(...nailPredictions) : 0,
        unreliable: false,
      } : null);

    } else {
      // ── Original single-crop path ──
      const imageData = preprocessCanvasToImageData(sourceCanvas);

      const validity = isFrameValid(imageData);
      if (!validity.ok) {
        alert("This image looks unusable (too dark/bright or blank). Please upload a clearer nail photo.");
        return;
      }

      const hb = await inferSingle(imageData);

      // Single-frame OOD check
      if (hb < PREDICTION_SANITY.minPlausible || hb > PREDICTION_SANITY.maxPlausible) {
        alert(
          `Result rejected: predicted Hb = ${hb.toFixed(1)} g/dL is outside the plausible range (${PREDICTION_SANITY.minPlausible}–${PREDICTION_SANITY.maxPlausible} g/dL).\n\n` +
          `This image is likely out-of-distribution. Please try with a clearer fingernail photo.`
        );
        return;
      }

      showResult(sourceCanvas, hb, null);
    }

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
  // Clamp to displayable range (physiological boundaries)
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

  // Unreliable result warning
  const $unreliableWarning = document.getElementById("unreliableWarning");
  if ($unreliableWarning) {
    if (stats && stats.unreliable) {
      $unreliableWarning.classList.remove("hidden");
      // Override badge to show unreliable
      $severityBadge.textContent = severity.label + " (unreliable)";
      $severityBadge.className = "severity-badge unreliable";
      $hbValue.style.color = "var(--yellow-500, #eab308)";
    } else {
      $unreliableWarning.classList.add("hidden");
    }
  }

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
  hideLowLightBanner();
  setInstructionMode("camera");
  stopLiveDetect();
  startCamera().then(() => {
    checkFlashSupport();
    startLightMonitor();
    startLiveDetect();
  });
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
    // If user manually turned on flash, dismiss the low-light warning
    if (flashOn) {
      hideLowLightBanner();
      lowLightDismissed = true;
    }
  } catch (err) {
    console.warn("Torch not supported:", err.message);
    $flashBtn.classList.add("flash-unsupported");
    flashOn = false;
  }
}

$flashBtn.addEventListener("click", toggleFlash);

// Low-light banner actions
if ($lowLightFlashBtn) {
  $lowLightFlashBtn.addEventListener("click", async () => {
    if (!flashOn) await toggleFlash();
    hideLowLightBanner();
    lowLightDismissed = true;
  });
}
if ($lowLightDismiss) {
  $lowLightDismiss.addEventListener("click", () => {
    hideLowLightBanner();
    lowLightDismissed = true;
  });
}


// ─── Utility ───

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


// ─── Bootstrap ───

setInstructionMode("camera");
loadModel();
