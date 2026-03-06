/* ================================================================
   HemoLens — Color Feature Extraction (JavaScript)
   ================================================================
   Computes the same 67 hand-crafted color statistics that the
   Python pipeline (extract_color_features.py) produces, so the
   dual-input ONNX model receives identical features at inference.

   Colour-space conversions match OpenCV's uint8 encoding:
     LAB:  L ∈ [0, 255]  A ∈ [0, 255]  B ∈ [0, 255]   (128 = zero)
     HSV:  H ∈ [0, 179]  S ∈ [0, 255]  V ∈ [0, 255]

   Usage:
     const feats = extractColorFeatures(nailImageData, skinImageData);
     // feats is a Float32Array of length 67
   ================================================================ */

// ─── Colour-Space Helpers ────────────────────────────────────────────────

/**
 * sRGB [0-255] → linear RGB [0-1].
 * Inverse of the sRGB companding function.
 */
function srgbToLinear(c) {
  c /= 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/**
 * Convert one RGB pixel to CIE-LAB (OpenCV uint8 scale).
 * Pipeline: sRGB → linear → XYZ (D65) → CIELAB → OpenCV uint8 encoding.
 */
function rgbToLabCV(r, g, b) {
  // sRGB → linear
  const rl = srgbToLinear(r);
  const gl = srgbToLinear(g);
  const bl = srgbToLinear(b);

  // Linear RGB → XYZ (D65 illuminant)
  let x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375;
  let y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750;
  let z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041;

  // Normalise by D65 white point
  x /= 0.95047;
  y /= 1.00000;
  z /= 1.08883;

  // XYZ → CIELAB
  const eps = 0.008856;
  const kappa = 903.3;
  const fx = x > eps ? Math.cbrt(x) : (kappa * x + 16) / 116;
  const fy = y > eps ? Math.cbrt(y) : (kappa * y + 16) / 116;
  const fz = z > eps ? Math.cbrt(z) : (kappa * z + 16) / 116;

  const Ls = 116 * fy - 16;   // L* ∈ [0, 100]
  const as = 500 * (fx - fy); // a* ∈ ~[-128, 127]
  const bs = 200 * (fy - fz); // b* ∈ ~[-128, 127]

  // Map to OpenCV uint8 scale: L*·255/100, a*+128, b*+128
  return [
    Ls * 255 / 100,
    as + 128,
    bs + 128,
  ];
}

/**
 * Convert one RGB pixel to HSV (OpenCV uint8 scale).
 * H ∈ [0, 179], S ∈ [0, 255], V ∈ [0, 255].
 */
function rgbToHsvCV(r, g, b) {
  const rf = r / 255;
  const gf = g / 255;
  const bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const d = max - min;

  let h = 0;
  if (d > 0) {
    if (max === rf)      h = 60 * (((gf - bf) / d) % 6);
    else if (max === gf) h = 60 * (((bf - rf) / d) + 2);
    else                 h = 60 * (((rf - gf) / d) + 4);
    if (h < 0) h += 360;
  }

  const s = max > 0 ? d / max : 0;
  const v = max;

  // OpenCV uint8 encoding
  return [
    h / 2,       // H: 0-360 → 0-179
    s * 255,     // S: 0-1   → 0-255
    v * 255,     // V: 0-1   → 0-255
  ];
}


// ─── Statistics Helpers ──────────────────────────────────────────────────

function mean(arr) {
  if (arr.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

function std(arr) {
  if (arr.length === 0) return 0;
  const m = mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += (arr[i] - m) * (arr[i] - m);
  return Math.sqrt(s / arr.length);
}

function medianVal(arr) {
  if (arr.length === 0) return 0;
  const sorted = Float64Array.from(arr).sort();
  const mid = sorted.length >> 1;
  return sorted.length & 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function percentile(arr, p) {
  if (arr.length === 0) return 0;
  const sorted = Float64Array.from(arr).sort();
  const k = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(k);
  const hi = Math.ceil(k);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (k - lo);
}


// ─── Per-ROI Feature Extraction ──────────────────────────────────────────

/**
 * Extract 27 colour statistics from an ImageData ROI.
 *
 * @param {ImageData} imageData  — ROI pixel data (RGBA).
 * @param {string}    prefix     — "nail_" or "skin_".
 * @returns {Object}  Map of feature names → values.
 */
function _extractRoiFeatures(imageData, prefix) {
  const px = imageData.data;
  const n = imageData.width * imageData.height;

  // Pre-allocate channel arrays
  const rArr = new Float64Array(n);
  const gArr = new Float64Array(n);
  const bArr = new Float64Array(n);
  const labL = new Float64Array(n);
  const labA = new Float64Array(n);
  const labB = new Float64Array(n);
  const hsvH = new Float64Array(n);
  const hsvS = new Float64Array(n);
  const hsvV = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const off = i * 4;
    const r = px[off];
    const g = px[off + 1];
    const b = px[off + 2];

    rArr[i] = r;
    gArr[i] = g;
    bArr[i] = b;

    const lab = rgbToLabCV(r, g, b);
    labL[i] = lab[0];
    labA[i] = lab[1];
    labB[i] = lab[2];

    const hsv = rgbToHsvCV(r, g, b);
    hsvH[i] = hsv[0];
    hsvS[i] = hsv[1];
    hsvV[i] = hsv[2];
  }

  const features = {};

  // RGB: mean, std, median, p5, p95 per channel (15)
  for (const [name, arr] of [["R", rArr], ["G", gArr], ["B", bArr]]) {
    features[`${prefix}rgb_${name}_mean`]   = mean(arr);
    features[`${prefix}rgb_${name}_std`]    = std(arr);
    features[`${prefix}rgb_${name}_median`] = medianVal(arr);
    features[`${prefix}rgb_${name}_p5`]     = percentile(arr, 5);
    features[`${prefix}rgb_${name}_p95`]    = percentile(arr, 95);
  }

  // LAB: mean, std per channel (6)
  for (const [name, arr] of [["L", labL], ["A", labA], ["B_lab", labB]]) {
    features[`${prefix}lab_${name}_mean`] = mean(arr);
    features[`${prefix}lab_${name}_std`]  = std(arr);
  }

  // HSV: mean, std per channel (6)
  for (const [name, arr] of [["H", hsvH], ["S", hsvS], ["V", hsvV]]) {
    features[`${prefix}hsv_${name}_mean`] = mean(arr);
    features[`${prefix}hsv_${name}_std`]  = std(arr);
  }

  return features;  // 27 features
}


// ─── Combined Feature Vector ─────────────────────────────────────────────

/**
 * Canonical feature ordering — must match the ONNX model's color_features
 * input (see export_hybrid_color.py :: COLOR_FEATURE_NAMES).
 */
const COLOR_FEATURE_ORDER = [
  // Nail ROI — RGB (15)
  "nail_rgb_R_mean", "nail_rgb_R_std", "nail_rgb_R_median", "nail_rgb_R_p5", "nail_rgb_R_p95",
  "nail_rgb_G_mean", "nail_rgb_G_std", "nail_rgb_G_median", "nail_rgb_G_p5", "nail_rgb_G_p95",
  "nail_rgb_B_mean", "nail_rgb_B_std", "nail_rgb_B_median", "nail_rgb_B_p5", "nail_rgb_B_p95",
  // Nail ROI — LAB (6)
  "nail_lab_L_mean", "nail_lab_L_std",
  "nail_lab_A_mean", "nail_lab_A_std",
  "nail_lab_B_lab_mean", "nail_lab_B_lab_std",
  // Nail ROI — HSV (6)
  "nail_hsv_H_mean", "nail_hsv_H_std",
  "nail_hsv_S_mean", "nail_hsv_S_std",
  "nail_hsv_V_mean", "nail_hsv_V_std",
  // Skin ROI — RGB (15)
  "skin_rgb_R_mean", "skin_rgb_R_std", "skin_rgb_R_median", "skin_rgb_R_p5", "skin_rgb_R_p95",
  "skin_rgb_G_mean", "skin_rgb_G_std", "skin_rgb_G_median", "skin_rgb_G_p5", "skin_rgb_G_p95",
  "skin_rgb_B_mean", "skin_rgb_B_std", "skin_rgb_B_median", "skin_rgb_B_p5", "skin_rgb_B_p95",
  // Skin ROI — LAB (6)
  "skin_lab_L_mean", "skin_lab_L_std",
  "skin_lab_A_mean", "skin_lab_A_std",
  "skin_lab_B_lab_mean", "skin_lab_B_lab_std",
  // Skin ROI — HSV (6)
  "skin_hsv_H_mean", "skin_hsv_H_std",
  "skin_hsv_S_mean", "skin_hsv_S_std",
  "skin_hsv_V_mean", "skin_hsv_V_std",
  // Cross-ROI contrast (9)
  "contrast_rgb_R_mean", "contrast_rgb_G_mean", "contrast_rgb_B_mean",
  "contrast_lab_L_mean", "contrast_lab_A_mean", "contrast_lab_B_lab_mean",
  "contrast_hsv_H_mean", "contrast_hsv_S_mean", "contrast_hsv_V_mean",
  // Cross-ROI ratios (3)
  "ratio_rgb_R", "ratio_rgb_G", "ratio_rgb_B",
  // Redness index (1)
  "redness_index",
];

const N_COLOR_FEATURES = COLOR_FEATURE_ORDER.length; // 67


/**
 * Compute the 67-element colour-feature vector from nail and skin ROIs.
 *
 * @param {ImageData} nailImageData — raw nail ROI pixels (RGBA, any size).
 * @param {ImageData} skinImageData — raw skin ROI pixels (RGBA, any size).
 * @returns {Float32Array}  67-element vector in canonical order.
 */
function extractColorFeatures(nailImageData, skinImageData) {
  const nailFeats = _extractRoiFeatures(nailImageData, "nail_");
  const skinFeats = _extractRoiFeatures(skinImageData, "skin_");
  const features = { ...nailFeats, ...skinFeats };

  // Contrast: nail_mean − skin_mean for every _mean channel
  const nailMeanKeys = Object.keys(nailFeats).filter(k => k.includes("_mean"));
  for (const nk of nailMeanKeys) {
    const sk = nk.replace("nail_", "skin_");
    if (sk in skinFeats) {
      features[nk.replace("nail_", "contrast_")] = nailFeats[nk] - skinFeats[sk];
    }
  }

  // Ratio: nail_rgb_{R,G,B}_mean / (skin_rgb_{R,G,B}_mean + ε)
  const EPS = 1e-6;
  for (const ch of ["R", "G", "B"]) {
    const nk = `nail_rgb_${ch}_mean`;
    const sk = `skin_rgb_${ch}_mean`;
    features[`ratio_rgb_${ch}`] = nailFeats[nk] / (skinFeats[sk] + EPS);
  }

  // Redness index: nail_lab_A_mean / (skin_lab_A_mean + ε)
  features["redness_index"] =
    nailFeats["nail_lab_A_mean"] / (skinFeats["skin_lab_A_mean"] + EPS);

  // Assemble Float32Array in canonical order
  const vec = new Float32Array(N_COLOR_FEATURES);
  for (let i = 0; i < N_COLOR_FEATURES; i++) {
    vec[i] = features[COLOR_FEATURE_ORDER[i]] ?? 0;
  }
  return vec;
}


/**
 * Get raw (un-preprocessed) ROI pixel data from a canvas and bounding box.
 *
 * @param {HTMLCanvasElement} srcCanvas — full frame canvas.
 * @param {{ x: number, y: number, w: number, h: number }} bbox — detection.
 * @returns {ImageData}
 */
function getRoiImageData(srcCanvas, bbox) {
  const x = Math.max(0, Math.round(bbox.x));
  const y = Math.max(0, Math.round(bbox.y));
  const w = Math.min(Math.round(bbox.w), srcCanvas.width - x);
  const h = Math.min(Math.round(bbox.h), srcCanvas.height - y);
  if (w <= 0 || h <= 0) return new ImageData(1, 1); // fallback
  const ctx = srcCanvas.getContext("2d");
  return ctx.getImageData(x, y, w, h);
}


// ─── Expose on window ────────────────────────────────────────────────────

window.extractColorFeatures = extractColorFeatures;
window.getRoiImageData       = getRoiImageData;
window.N_COLOR_FEATURES      = N_COLOR_FEATURES;
