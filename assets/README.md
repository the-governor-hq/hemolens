# assets/

Quantized TFLite models and label files for on-device inference.

## Files

| File | Description | Size |
|------|-------------|------|
| `hemolens_model_int8.tflite` | INT8 quantized MobileNetV4-Conv-Small | ~1 MB |
| `hemolens_model_float16.tflite` | Float16 model (higher accuracy) | ~5 MB |
| `labels.txt` | Severity class labels for output interpretation | — |

## Generating Models

```bash
cd ../model
python prepare_dataset.py
python train.py --config configs/mobilenet_edge.yaml
python export_tflite.py --checkpoint checkpoints/best_model.pth --quantize int8
```

The exported `.tflite` files will be placed in this directory automatically.

## On-Device Performance (Expected)

| Metric | INT8 | Float16 |
|--------|------|--------|
| Model size | ~1 MB | ~5 MB |
| Inference (Pixel 6) | <5 ms | ~10 ms |
| Inference (iPhone 13) | <3 ms | ~8 ms |
