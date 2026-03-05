# model/

PyTorch training pipeline for the HemoLens regression model with TFLite export.

Supports any timm backbone — default edge config uses **MobileNetV4-Conv-Small** (~2.5M params, ~2.5 MB INT8).

The primary approach is a **hybrid pipeline**: frozen pretrained backbone as feature extractor + CatBoost/Ridge regression head. This significantly outperforms end-to-end fine-tuning in the small-data regime (250 patients). Best test MAE = **1.305 g/dL** (R² = 0.46).

## Files

| File | Description |
|------|-------------|
| `prepare_dataset.py` | Crop nail ROIs, session-aware splits → `data/processed/` |
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

# 2. Train hybrid model (recommended)
python train_hybrid.py --config configs/mobilenet_edge.yaml

# 3. Export to ONNX (full backbone + Ridge head)
python export_hybrid.py

# 4. (Optional) End-to-end fine-tuning for comparison
python train.py --config configs/mobilenet_edge.yaml
```

## Results

### Hybrid Models (MobileNetV4-Conv-Small backbone)

| Head | Val MAE | Test MAE | Test R² | CV MAE (3-fold) |
|------|---------|----------|---------|------------------|
| **CatBoost** | **1.520** | **1.305** | **0.460** | 1.912 ± 0.27 |
| Ridge | 1.566 | 1.459 | 0.431 | — |
| Ridge+PCA128 | 1.563 | 1.453 | 0.436 | — |
| Ridge+PCA32 | 1.545 | 1.527 | 0.411 | — |
| ElasticNet | 1.645 | 1.541 | 0.356 | — |
| CNN-only Ridge (edge) | 1.501 | 1.525 | 0.408 | — |
| End-to-end fine-tuned | 2.891 | — | -1.61 | — |

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
