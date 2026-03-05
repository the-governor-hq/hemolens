/* ================================================================
   HemoLens Web Demo - Application Logic
   ================================================================
   Pipeline:
     1. Load ONNX model via ONNX Runtime Web (WASM backend)
     2. Stream camera via getUserMedia (prefer rear camera)
     3. On capture: extract nail region from guide overlay
     4. Preprocess: resize 224×224, ImageNet normalize, CHW layout
     5. Run inference -> Hb (g/dL)
     6. Display result with severity classification
   ================================================================ */

// ─── Configuration ───

const MODEL_PATH = "model/hemolens_hybrid_web.onnx";
const INPUT_SIZE = 224;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];

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


// ─── State ───

let session = null;          // ONNX Runtime inference session
let currentStream = null;    // MediaStream
let facingMode = "environment";  // prefer rear camera
let isProcessing = false;


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


// ─── Capture ───

function captureFrame() {
  if (isProcessing) return;

  const vw = $video.videoWidth;
  const vh = $video.videoHeight;
  if (!vw || !vh) return;

  // Flash effect
  const flash = document.createElement("div");
  flash.className = "flash";
  document.querySelector(".camera-container").appendChild(flash);
  setTimeout(() => flash.remove(), 400);

  // The guide overlay SVG viewBox is 400×400
  // The nail zone is at: x=120, y=130, w=160, h=120
  // Map from SVG coords to video coords
  const container = document.querySelector(".camera-container");
  const cRect = container.getBoundingClientRect();

  // SVG viewBox preserveAspectRatio="xMidYMid meet" -> compute actual SVG render area
  const svgAspect = 1; // 400/400
  const containerAspect = cRect.width / cRect.height;

  let svgRenderW, svgRenderH, svgOffsetX, svgOffsetY;
  if (containerAspect > svgAspect) {
    // Container is wider than SVG -> SVG height fills, width is centered
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

  // Nail region in SVG coords
  const nailSVG = { x: 120, y: 130, w: 160, h: 120 };

  // Convert to pixel coords within container
  const scale = svgRenderW / 400;
  const nailPx = {
    x: svgOffsetX + nailSVG.x * scale,
    y: svgOffsetY + nailSVG.y * scale,
    w: nailSVG.w * scale,
    h: nailSVG.h * scale,
  };

  // Convert container pixels to video pixels
  // Video is object-fit: cover -> need to compute the mapping
  const videoAspect = vw / vh;
  let renderW, renderH, videoOffX, videoOffY;
  if (containerAspect > videoAspect) {
    // Container wider than video -> video width fills
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

  // Map nail rect from container coords to video coords
  const sx = ((nailPx.x - videoOffX) / renderW) * vw;
  const sy = ((nailPx.y - videoOffY) / renderH) * vh;
  const sw = (nailPx.w / renderW) * vw;
  const sh = (nailPx.h / renderH) * vh;

  // Clamp to video bounds
  const cropX = Math.max(0, Math.round(sx));
  const cropY = Math.max(0, Math.round(sy));
  const cropW = Math.min(Math.round(sw), vw - cropX);
  const cropH = Math.min(Math.round(sh), vh - cropY);

  // Draw crop to capture canvas
  $captureCanvas.width = cropW;
  $captureCanvas.height = cropH;
  const ctx = $captureCanvas.getContext("2d");
  ctx.drawImage($video, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

  // Run inference
  processImage($captureCanvas);
}


async function processImage(sourceCanvas) {
  if (isProcessing) return;
  isProcessing = true;

  // Show scanning animation
  $scanLine.classList.remove("hidden");

  try {
    // 1. Resize to 224×224
    const resizeCanvas = document.createElement("canvas");
    resizeCanvas.width = INPUT_SIZE;
    resizeCanvas.height = INPUT_SIZE;
    const rCtx = resizeCanvas.getContext("2d");
    rCtx.drawImage(sourceCanvas, 0, 0, INPUT_SIZE, INPUT_SIZE);

    // 2. Get pixel data
    const imageData = rCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const pixels = imageData.data; // RGBA

    // 3. Preprocess: normalize with ImageNet stats, CHW layout
    const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    const pixelCount = INPUT_SIZE * INPUT_SIZE;

    for (let i = 0; i < pixelCount; i++) {
      const r = pixels[i * 4]     / 255.0;
      const g = pixels[i * 4 + 1] / 255.0;
      const b = pixels[i * 4 + 2] / 255.0;

      // CHW layout: [C, H, W]
      float32Data[0 * pixelCount + i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]; // R
      float32Data[1 * pixelCount + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]; // G
      float32Data[2 * pixelCount + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]; // B
    }

    // 4. Create tensor
    const inputTensor = new ort.Tensor("float32", float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);

    // 5. Run inference
    const results = await session.run({ image: inputTensor });
    const hbPrediction = results.hb_prediction.data[0];

    // 6. Show result
    showResult(sourceCanvas, hbPrediction);

  } catch (err) {
    console.error("Inference error:", err);
    alert("Inference failed: " + err.message);
  } finally {
    $scanLine.classList.add("hidden");
    isProcessing = false;
  }
}


// ─── Result Display ───

function showResult(sourceCanvas, hbValue) {
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
}


// ─── File Upload ───

function handleFileUpload(file) {
  if (!file || !file.type.startsWith("image/")) return;

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

$captureBtn.addEventListener("click", captureFrame);

$switchBtn.addEventListener("click", () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  startCamera();
});

$uploadBtn.addEventListener("click", () => $fileInput.click());
$fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleFileUpload(e.target.files[0]);
  e.target.value = ""; // reset so same file can be re-selected
});

$closeResult.addEventListener("click", hideResult);
$retakeBtn.addEventListener("click", hideResult);

// Keyboard shortcut: Space to capture
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !$resultSection.classList.contains("hidden") === false) {
    e.preventDefault();
    captureFrame();
  }
  if (e.code === "Escape") hideResult();
});


// ─── Utility ───

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


// ─── Bootstrap ───

loadModel();
