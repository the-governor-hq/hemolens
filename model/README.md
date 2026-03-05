# model/

Training pipelines for both HemoLens models: **nail detector** (YOLOv8-nano) and **Hb regressor** (MobileNetV4 + Ridge/CatBoost).

## Hb Regression

Supports any timm backbone — default edge config uses **MobileNetV4-Conv-Small** (~2.5M params, ~2.5 MB INT8).

The primary approach is a **hybrid pipeline**: frozen pretrained backbone as feature extractor + CatBoost/Ridge regression head. This outperforms end-to-end fine-tuning by ~2× in the small-data regime (250 patients).

- Best holdout test MAE = **1.305 g/dL** (38 test patients, 3 sessions — likely optimistic)
- **3-fold session-aware CV MAE = 1.91 ± 0.27 g/dL** (CV R² < 0 — more honest generalizable estimate)
- Deployed web model (CNN-only Ridge, no color features): test MAE ≈ 1.52 g/dL

## Nail Detection (YOLOv8-nano)

- **mAP50 = 0.995** for nails, 0.868 for skin, 0.931 overall
- Precision = 0.982, Recall = 1.000 (nails)
- 1,500 annotated boxes (750 nail + 750 skin) from 250 images
- ONNX export: 11.6 MB, 320×320 static input, opset 13

## Files

| File | Description |
|------|-------------|
| `prepare_dataset.py` | Crop nail ROIs, session-aware splits → `data/processed/` |
| `prepare_yolo_dataset.py` | Convert annotations to YOLO format for nail detector |
| `train_nail_detector.py` | Train YOLOv8-nano + ONNX export → `web-demo/model/` |
| `train_hybrid.py` | **Recommended** — Frozen backbone feature extraction + Ridge/MLP |
| `train.py` | End-to-end fine-tuning with progressive unfreezing (for comparison) |
| `sweep_hybrid.py` | Backbone × head grid search (4 backbones × 4 heads) |
| `extract_color_features.py` | Extract 67 color features per patient |
| `export_hybrid.py` | Export hybrid model to ONNX (backbone + Ridge head) |
| `export_tflite.py` | Convert PyTorch checkpoint → ONNX → TFLite (INT8/FP16) |
| `dataset.py` | Custom `Dataset` class for fingernail images + Hb labels |
| `transforms.py` | Augmentation pipelines (color jitter, random erasing, etc.) |
| `configs/mobilenet_edge.yaml` | **Edge-optimized** — MobileNetV4-Conv-Small config |
| `configs/vit_base.yaml` | ViT-Small/16 config (larger, research-grade) |

## Usage

```bash
# 1. Prepare dataset (crop nail ROIs, create train/val/test splits)
python prepare_dataset.py

# 2. Train hybrid Hb model (recommended)
python train_hybrid.py --config configs/mobilenet_edge.yaml

# 3. Export Hb model to ONNX (full backbone + Ridge head)
python export_hybrid.py

# 4. Train nail detector + export ONNX
python prepare_yolo_dataset.py    # convert annotations → YOLO format
python train_nail_detector.py     # YOLOv8n training + ONNX export → web-demo/model/

# 5. (Optional) End-to-end fine-tuning for comparison
python train.py --config configs/mobilenet_edge.yaml
```

## Results

> **Note:** Hb test metrics are from a 38-patient holdout (3 sessions). The small test set means these numbers are noisy. Cross-validation gives a more honest estimate.

### Nail Detector (YOLOv8-nano, test set — 38 images)

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| **Nail** | **0.982** | **1.000** | **0.995** |
| Skin | 0.912 | 0.909 | 0.868 |
| Overall | 0.947 | 0.955 | 0.931 |

Early stopped at epoch 58 (best at 38). Trained on 151 images, validated on 61.

### Hybrid Hb Models (MobileNetV4-Conv-Small backbone)

| Head | Val MAE | Test MAE | Test R² | CV MAE (3-fold) |
|------|---------|----------|---------|------------------|
| **CatBoost** | **1.520** | **1.305** | **0.460** | 1.912 ± 0.27 |
| Ridge | 1.566 | 1.459 | 0.431 | — |
| Ridge+PCA128 | 1.563 | 1.453 | 0.436 | — |
| Ridge+PCA32 | 1.545 | 1.527 | 0.411 | — |
| ElasticNet | 1.645 | 1.541 | 0.356 | — |
| CNN-only Ridge (edge) | 1.501 | 1.525 | 0.408 | — |
| End-to-end fine-tuned | 2.891 | — | -1.61 | — |

**Per-severity test MAE (CatBoost, 38 patients):**

| Severity | n | Test MAE (g/dL) |
|----------|---|------------------|
| Severe (<8) | 3 | 3.932 |
| Moderate (8–11) | 3 | 1.139 |
| Mild (11–13) | 13 | 0.695 |
| Normal (≥13) | 19 | 1.334 |

**Cross-validation:** 3-fold session-aware CV (Ridge on train+val, 17 session groups): MAE = 1.91 ± 0.27 g/dL, R² < 0 (model does not outperform predicting the mean under CV).

### Backbone Sweep

| Backbone | Best Head | Test MAE | Test R² | CV MAE |
|----------|-----------|----------|---------|--------|
| **mobilenetv4_conv_small** | **CatBoost** | **1.305** | **0.460** | 1.912 ± 0.27 |
| efficientnet_b0 | CatBoost | 1.459 | 0.431 | **1.527 ± 0.32** |
| tf_efficientnetv2_b0 | Ridge+PCA32 | 1.497 | 0.469 | 1.563 ± 0.32 |
| mobilenetv3_small_100 | CatBoost | 1.444 | 0.399 | 1.612 ± 0.29 |

## Architecture (Edge — Hybrid)

```
Input (224×224×3)
  → MobileNetV4-Conv-Small (frozen pretrained, timm) → Global Pool → [1280-dim]
  → CatBoost / Ridge Regression
  → Hb (g/dL)
```

## Training Strategy (Small-Data Regime)

- **250 patients**, 3 nail crops each → averaged to patient-level 1280-dim CNN features
- 67 handcrafted color features (RGB/LAB/HSV mean, std, median, P5, P95 + redness index) → 1347-dim
- CatBoost regression (best) or Ridge with RidgeCV α-selection
- Session-aware splits (GroupShuffleSplit on measurement date, 20 sessions) to prevent leakage
- 3-fold session-aware cross-validation (17 session groups)
- Frozen backbone: no fine-tuning → eliminates overfitting on small data
- End-to-end ONNX export: backbone + linear head in one model
- **Deployed web model uses CNN-only Ridge** (color features not available on-device) — weaker than offline CatBoost

## Limitations

- **250 patients, single center** — too small for medical ML; results may not generalize
- **Severe anemia MAE ≈ 3.9 g/dL** — unreliable on the most clinically critical cases
- **CV R² < 0** — the model does not reliably outperform predicting the population mean under cross-validation
- **Val R² ≈ 0.09 vs test R² = 0.46** — the gap suggests a lucky test split, not robust generalization
- **WHO severity bins** in code use general thresholds (8/11/13 g/dL); real guidelines are sex- and age-specific
