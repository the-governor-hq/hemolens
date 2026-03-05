# HemoLens 🔬

**An experimental, on-device pipeline for non-invasive hemoglobin level estimation.**

A two-stage browser pipeline: **YOLOv8-nano nail detector** → **MobileNetV4 Hb regression**, both running as ONNX models client-side via ONNX Runtime Web. Trained on the 2024 Yakimov et al. fingernail dataset (250 patients, single center).

- **Nail detection**: mAP50 = 0.995, Precision = 0.982, Recall = 1.000 (YOLOv8n, 320×320)
- **Hb regression CV MAE: 1.91 ± 0.27 g/dL** (3-fold session-aware CV, 17 session groups)
- Best *offline hybrid* holdout result (CNN + color features + CatBoost): test MAE = 1.305 g/dL on 38 patients (3 test sessions — likely optimistic)
- Deployed *web* model (ONNX, CNN-only Ridge head): test MAE ≈ 1.52 g/dL when averaging 3 crops per patient
- **Severe anemia detection is unreliable** — test MAE = 3.93 g/dL on severe cases (n=3)

The web app auto-detects nail regions in each frame, then runs multi-frame Hb inference (30 captures with IQR outlier rejection and median aggregation) to reduce noise. No images leave the device.

---

## Project Structure

```
HemoLens/
├── research/          # Jupyter notebooks — EDA, feature extraction, baselines
├── model/             # Training pipelines (nail detector + Hb regressor), ONNX export
├── web-demo/          # Browser PWA — two-stage ONNX inference, camera, offline-capable
├── assets/            # Labels and supplementary files
└── data/              # Raw images, metadata, processed crops & features
```

## Overview

HemoLens estimates hemoglobin (Hb) levels from smartphone camera images of fingernail beds — no blood draw required. The pipeline runs entirely in the browser with two ONNX models:

1. **Nail Detection (Stage 1)** — YOLOv8-nano (11.6 MB ONNX) detects nail bounding boxes in camera frames at 320×320. Trained on 1,500 annotated boxes (nail + skin classes) from the Yakimov dataset. Achieves mAP50 = 0.995 for nails with perfect recall.
2. **Hb Regression (Stage 2)** — Frozen MobileNetV4-Conv-Small backbone (1280-dim features) + Ridge regression head (~10 MB ONNX). Runs on each detected nail crop (224×224, ImageNet-normalized). Multi-frame median reduces noise.
3. **Web App** — Progressive Web App (PWA) via ONNX Runtime Web (WASM). Camera capture, auto nail detection, multi-frame inference (30 frames → IQR outlier rejection → median), WHO severity classification, image upload mode. Fully offline-capable.

### Supporting pipelines

- **Research** — EDA on the Yakimov et al. (2024) fingernail image dataset, color-space feature extraction (RGB, LAB, HSV), and traditional ML baselines (Ridge, SVR, Gradient Boosting).
- **Offline Hybrid Model** — Frozen MobileNetV4 features + 67 handcrafted color features → CatBoost regression head. Best single-split test MAE = 1.305 g/dL, but color features are not available in the browser pipeline.

## Baseline Results (Notebook 03)

| Model | Test MAE (g/dL) | Test R² | CV MAE (g/dL) |
|-------|-----------------|---------|----------------|
| Ridge Regression | 1.575 | 0.384 | 1.635 |
| Random Forest | 1.610 | 0.307 | 1.602 |
| Gradient Boosting | 1.649 | 0.288 | 1.751 |
| Lasso Regression | 1.692 | 0.324 | 1.578 |
| SVR (RBF) | 1.774 | 0.231 | 1.845 |

*Hand-crafted color features on 51 dimensions.*

## Deep Learning Results

> **Important context:** Test metrics are from a 38-patient holdout (3 sessions). The small test set and limited session diversity mean these numbers are noisy. Cross-validation (CV) gives a more honest — and notably worse — performance estimate. The CV R² is negative, meaning the model does not reliably outperform predicting the population mean under session-aware cross-validation.

### Best Model: MobileNetV4-Conv-Small + CatBoost (Hybrid)

| Approach | Val MAE (g/dL) | Test MAE (g/dL) | Test R² | CV MAE (3-fold) |
|----------|----------------|-----------------|---------|------------------|
| **MobileNetV4 + CatBoost** | **1.520** | **1.305** | **0.460** | 1.912 ± 0.27 |
| MobileNetV4 + Ridge | 1.566 | 1.459 | 0.431 | 1.912 ± 0.27 |
| MobileNetV4 + Ridge+PCA128 | 1.563 | 1.453 | 0.436 | — |
| MobileNetV4 + ElasticNet | 1.645 | 1.541 | 0.356 | — |
| CNN-only Ridge (no color feats) | 1.501 | 1.525 | 0.408 | — |
| End-to-end fine-tuned MobileNetV4 | 2.891 | — | -1.61 | — |

**Per-severity test MAE (CatBoost, n = 38 patients):**

| Severity | n | Test MAE (g/dL) |
|----------|---|------------------|
| Severe (<8) | 3 | 3.932 |
| Moderate (8–11) | 3 | 1.139 |
| Mild (11–13) | 13 | 0.695 |
| Normal (≥13) | 19 | 1.334 |

### Backbone Sweep (4 Backbones × 4 Heads)

| Backbone | Best Head | Test MAE | Test R² | CV MAE |
|----------|-----------|----------|---------|--------|
| **mobilenetv4_conv_small** | **CatBoost** | **1.305** | **0.460** | 1.912 ± 0.27 |
| efficientnet_b0 | CatBoost | 1.459 | 0.431 | **1.527 ± 0.32** |
| tf_efficientnetv2_b0 | Ridge+PCA32 | 1.497 | 0.469 | 1.563 ± 0.32 |
| mobilenetv3_small_100 | CatBoost | 1.444 | 0.399 | 1.612 ± 0.29 |

The hybrid frozen-backbone approach beats end-to-end fine-tuning by ~2× on this 250-patient dataset. EfficientNet-B0 has the best cross-validation stability.

## Key Features

- **Two-stage pipeline**: YOLOv8-nano nail detection → MobileNetV4 Hb regression, both ONNX, both client-side.
- **Automatic nail detection**: No manual alignment needed. The detector finds nail bounding boxes in each frame (mAP50 = 0.995, recall = 1.000). Falls back gracefully to a finger guide overlay if detector fails to load.
- **Multi-frame inference**: Captures 30 frames at ~12.5 fps, rejects outliers via IQR, takes the median — significantly more robust than single-shot prediction.
- **Privacy-preserving**: No images are ever uploaded; everything stays in the browser.
- **Offline-capable PWA**: Service worker caches all assets including both ONNX models. Works without network after first load.
- **Lightweight models**: Nail detector 11.6 MB + Hb regressor ~10 MB. Total ~22 MB, runs on any device with WebAssembly.
- **Image upload mode**: Upload a photo → detector finds all nails → runs Hb on each crop → reports median with per-nail breakdown.
- **Web-ready**: ONNX Runtime Web (WASM) — works in any modern browser with a camera.

## Dataset

Based on the open-access fingernail bed image dataset from:

> Yakimov, P. et al. (2024). *Non-invasive hemoglobin estimation from fingernail bed images using deep learning.*

- 250 patients, 3 nail ROIs per patient → **750 cropped training images**
- Hb range: 4.4–16.9 g/dL
- Session-aware splits (GroupShuffleSplit on measurement date): 70% train / 15% val / 15% test
- 20 unique measurement sessions — zero cross-split leakage

## Quick Start

### 1. Research & EDA
```bash
cd research
pip install -r requirements.txt
jupyter lab
```

### 2. Prepare Dataset & Train
```bash
cd model
pip install -r requirements.txt

# Hb regression pipeline
python prepare_dataset.py                              # crop nail ROIs → data/processed/
python train_hybrid.py --config configs/mobilenet_edge.yaml  # hybrid: frozen CNN features + CatBoost
python export_hybrid.py                                # export Hb model to ONNX

# Nail detector pipeline
python prepare_yolo_dataset.py                         # convert annotations → YOLO format
python train_nail_detector.py                          # YOLOv8n training + ONNX export → web-demo/model/
```

### 3. Run the Web Demo
```bash
cd web-demo
python serve.py            # HTTPS on localhost:8443
# or
python serve.py --http --port 8080   # HTTP on localhost:8080
```

## Architecture

### Two-Stage Inference Pipeline

```
Camera frame (any resolution)
  ┌─── Stage 1: Nail Detection ───────────────────────────────┐
  │ Letterbox resize → 320×320                                │
  │ YOLOv8-nano (11.6 MB ONNX)                               │
  │ → Nail bounding boxes (class: nail/skin, conf, x,y,w,h)  │
  │ → NMS (IoU 0.45, conf 0.35) → best nail crop             │
  └───────────────────────────────────────────────────────────┘
        ↓ crop nail ROI from original frame
  ┌─── Stage 2: Hb Regression ───────────────────────────────┐
  │ Resize (shorter→256) → CenterCrop 224×224                │
  │ ImageNet normalize                                        │
  │ MobileNetV4-Conv-Small (frozen) → [1280-dim]             │
  │ Ridge Regression → Hb (g/dL)                             │
  │ ~10 MB ONNX                                              │
  └───────────────────────────────────────────────────────────┘
        ↓
  Hb prediction (single frame)
```

### Nail Detector (YOLOv8-nano)

| Metric | Nail class | Skin class | Overall |
|--------|-----------|-----------|---------|
| Precision | 0.982 | 0.912 | 0.947 |
| Recall | 1.000 | 0.909 | 0.955 |
| mAP50 | 0.995 | 0.868 | 0.931 |

- Trained on 1,500 bounding boxes (750 nail + 750 skin) from 250 images
- 151 / 61 / 38 train / val / test split (session-aware)
- Early stopped at epoch 58 (best at epoch 38)
- ONNX opset 13, 320×320 static input, output shape (1, 6, 2100)

### Hb Regression

### Hb Regression

**Training strategy for 250 patients:**
- **Hybrid pipeline**: Frozen ImageNet backbone as feature extractor — avoids overfitting with small data
- 3 nail crops per patient averaged to one 1280-dim feature vector
- 67 handcrafted color features (RGB/LAB/HSV mean, std, median, P5, P95 + redness index) → 1347-dim input (offline model only)
- Session-aware splits (GroupShuffleSplit on measurement date) to prevent leakage
- 3-fold session-aware cross-validation: **MAE = 1.912 ± 0.27 g/dL** (17 session groups, CV R² < 0)
- End-to-end ONNX export (backbone + Ridge head) for edge deployment
- **Note:** The deployed web model uses CNN-only Ridge (no color features), which is weaker than the offline CatBoost+color model

**Multi-frame inference pipeline (web app):**

```
Capture button pressed
  → 30 frames captured at 80 ms intervals (~12.5 fps)
  → Per frame:
      1. Nail detector runs on full frame → best nail bounding box
      2. Crop detected nail from original frame
      3. Resize (shorter side → 256) → center-crop 224×224 → ImageNet normalize
      4. Hb regression model → single Hb prediction
  → Collect 30 Hb predictions
  → IQR outlier rejection (drop values outside Q1 − 1.5·IQR … Q3 + 1.5·IQR)
  → Median of remaining predictions → final Hb estimate
  → Display with confidence stats (std dev, range, frames used)
```

**Image upload mode:**
```
User uploads photo
  → Nail detector finds all nails in image
  → Each nail: crop → Hb regression → individual prediction
  → Median of all nail predictions → final Hb estimate
  → All nail detections shown as overlay boxes
```

## ⚠️ Limitations

- **Tiny dataset**: 250 patients from a single center. Results may not generalize to other populations, skin tones, cameras, or lighting conditions.
- **Severe anemia detection is unreliable**: The model's error on severe cases (Hb < 8 g/dL) is ~4 g/dL — this is the most clinically important class and the model fails on it.
- **CV R² is negative**: Under session-aware cross-validation, the model does not reliably outperform predicting the population mean. The favorable holdout test R² (0.46) likely reflects a lucky 3-session test split.
- **Train/deploy gap**: The best offline model (CatBoost + color features) cannot run in the browser. The deployed web model (CNN-only Ridge) is weaker.
- **Multi-frame reduces variance, not bias**: If the model systematically mispredicts under certain conditions, 30 frames will converge to the wrong answer with high confidence.
- **No true uncertainty estimation**: The reported confidence stats (std dev, range) measure inter-frame consistency, not prediction accuracy.
- **WHO severity bins**: The code uses general-population thresholds (8/11/13 g/dL). Real WHO guidelines are sex- and age-specific.

## ⚠️ Disclaimer

> **This project is NOT a medical device.** HemoLens is a personal research project and is provided strictly for educational and experimental purposes. It has not been validated in a clinical setting, is not approved by any regulatory body (FDA, CE, Health Canada, etc.), and must **never** be used to make medical decisions, diagnose anemia, or replace a blood test.
>
> **Do not rely on this software for any health-related purpose.** If you suspect anemia or any medical condition, consult a qualified healthcare professional and get a proper complete blood count (CBC).
>
> The authors assume no liability for any use or misuse of this software.

## Models

| Model | File | Size | Input | Purpose |
|-------|------|------|-------|---------|
| YOLOv8-nano | `web-demo/model/nail_detector.onnx` | 11.6 MB | 1×3×320×320 | Nail detection (stage 1) |
| MobileNetV4 + Ridge | `web-demo/model/hemolens_hybrid_web.onnx` | ~10 MB | 1×3×224×224 | Hb regression (stage 2) |

Both models use ONNX opset 13 and run via ONNX Runtime Web (WASM backend).

## Requirements

| Component       | Stack                              |
|-----------------|-------------------------------------|
| Research        | Python 3.10+, Jupyter, pandas, matplotlib, seaborn |
| Hb Model Training | PyTorch 2.x, timm, CatBoost, scikit-learn |
| Nail Detector   | ultralytics (YOLOv8), onnxslim |
| Web App         | Modern browser with WebAssembly + camera (Chrome, Firefox, Safari, Edge) |
| Dev Server      | Python 3 (any static HTTPS server works) |

## License

MIT — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome. Please open an issue before submitting a PR for major changes.
