# model/

PyTorch training pipeline for the HemoLens regression model with TFLite export.

Supports any timm backbone — default edge config uses **MobileNetV4-Conv-Small** (~2.5M params, ~1 MB INT8).

## Files

| File | Description |
|------|-------------|
| `prepare_dataset.py` | Crop nail ROIs from raw photos → `data/processed/nail_crops/` |
| `train.py` | Backbone-agnostic training with progressive unfreezing |
| `export_tflite.py` | Convert PyTorch checkpoint → ONNX → TFLite (INT8/FP16) |
| `dataset.py` | Custom `Dataset` class for fingernail images + Hb labels |
| `transforms.py` | Augmentation pipelines (color jitter, random erasing, etc.) |
| `configs/mobilenet_edge.yaml` | **Edge-optimized** — MobileNetV4-Conv-Small, heavy regularization |
| `configs/vit_base.yaml` | ViT-Small/16 config (larger, research-grade) |

## Usage

```bash
# 1. Prepare dataset (crop nail ROIs, create train/val/test splits)
python prepare_dataset.py

# 2. Train edge model
python train.py --config configs/mobilenet_edge.yaml

# 3. Export to TFLite
python export_tflite.py --checkpoint checkpoints/best_model.pth --quantize int8
```

## Architecture (Edge)

```
Input (224×224×3)
  → MobileNetV4-Conv-Small (pretrained, timm) → Global Pool → [1280-dim]
  → Linear(1280→64) → ReLU → Dropout(0.4)
  → Linear(64→1) → Hb (g/dL)
```

## Training Strategy (Small-Data Regime)

- **750 nail-crop images** from 250 patients (3 crops/patient)
- Patient-level stratified splits (70/15/15) to prevent leakage
- Progressive unfreezing: head-only → full fine-tuning with differential LR
- Aggressive augmentation: color jitter, rotation ±20°, random erasing, random resized crop
- Label noise (σ=0.1 g/dL) + Huber loss (δ=1.5) for outlier robustness
- Early stopping with patience=15
