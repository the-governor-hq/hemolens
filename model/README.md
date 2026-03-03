# model/

PyTorch training pipeline for the HemoLens ViT regression model, plus TFLite export with INT8 quantization.

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training loop — ViT backbone + regression head |
| `export_tflite.py` | Convert PyTorch checkpoint → ONNX → TFLite (INT8) |
| `dataset.py` | Custom `Dataset` class for fingernail images + Hb labels |
| `transforms.py` | Training/validation augmentation pipelines |
| `configs/vit_base.yaml` | Hyperparameter configuration |

## Usage

```bash
# Train
python train.py --config configs/vit_base.yaml

# Export to TFLite
python export_tflite.py --checkpoint checkpoints/best_model.pth --quantize int8
```

## Architecture

```
Input (224×224×3) → ViT-Small/16 (pretrained, timm) → [CLS] token → Linear(384→128) → ReLU → Dropout → Linear(128→1) → Hb (g/dL)
```
