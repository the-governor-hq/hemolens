# assets/

Quantized TFLite models and label files for on-device inference.

## Files

| File | Description |
|------|-------------|
| `hemolens_model_int8.tflite` | INT8 quantized ViT model (~5-10 MB) |
| `hemolens_model_float16.tflite` | Float16 model (higher accuracy, ~20 MB) |
| `labels.txt` | Severity class labels for output interpretation |

## Generating Models

```bash
cd ../model
python export_tflite.py --checkpoint checkpoints/best_model.pth --quantize int8
```

The exported `.tflite` files will be placed in this directory automatically.
