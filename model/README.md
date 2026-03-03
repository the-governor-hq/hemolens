# model/

PyTorch training pipeline for the HemoLens regression model with TFLite export.

Supports any timm backbone — default edge config uses **MobileNetV4-Conv-Small** (~2.5M params, ~2.5 MB INT8).

The primary approach is a **hybrid pipeline**: frozen pretrained backbone as feature extractor + Ridge regression head. This significantly outperforms end-to-end fine-tuning in the small-data regime (250 patients).

## Files

| File | Description |
|------|-------------|
| `prepare_dataset.py` | Crop nail ROIs from raw photos → `data/processed/nail_crops/` |
| `train_hybrid.py` | **Recommended** — Frozen backbone feature extraction + Ridge/MLP |
| `train.py` | End-to-end fine-tuning with progressive unfreezing (for comparison) |
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

| Approach | Val MAE | Test MAE | Val R² | CV MAE |
|----------|---------|----------|--------|--------|
| **Hybrid (frozen CNN + color → Ridge)** | **1.690** | **1.365** | **0.483** | **1.656 ± 0.18** |
| Ridge+PCA128 | 1.717 | 1.402 | 0.485 | — |
| ElasticNet | 1.909 | 1.406 | 0.377 | — |
| End-to-end fine-tuned | 2.315 | — | 0.143 | — |

## Architecture (Edge — Hybrid)

```
Input (224×224×3)
  → MobileNetV4-Conv-Small (frozen pretrained, timm) → Global Pool → [1280-dim]
  → Ridge Regression (sklearn, exported as Linear(1280→1))
  → Hb (g/dL)
```

## Training Strategy (Small-Data Regime)

- **250 patients**, 3 nail crops each → averaged to patient-level 1280-dim CNN features
- 51 handcrafted color features (RGB, LAB, HSV) concatenated → 1331-dim
- Ridge regression with 5-fold cross-validation (α selected via RidgeCV)
- Patient-level stratified splits (70/15/15) to prevent leakage
- Frozen backbone: no fine-tuning → eliminates overfitting on small data
- End-to-end ONNX export: backbone + linear head in one model
