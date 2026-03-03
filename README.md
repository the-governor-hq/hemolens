# HemoLens 🔬

**A high-performance, offline Flutter engine for non-invasive hemoglobin level estimation.**

Powered by MobileNetV4-Conv-Small and optimized for on-device inference (~1 MB INT8, <5 ms latency) using the 2024 Yakimov et al. fingernail dataset.

---

## Project Structure

```
HemoLens/
├── research/          # Jupyter notebooks — EDA, feature extraction, baselines
├── model/             # PyTorch training, dataset prep & TFLite export
├── flutter_plugin/    # Dart/C++ plugin for on-device inference
├── example/           # Demo Flutter app with real-time camera detection
├── assets/            # Quantized .tflite models and labels
└── data/              # Raw images, metadata, processed crops & features
```

## Overview

HemoLens estimates hemoglobin (Hb) levels from smartphone camera images of fingernail beds — no blood draw required. The pipeline:

1. **Research** — EDA on the Yakimov et al. (2024) fingernail image dataset, color-space feature extraction (RGB, LAB, HSV), and traditional ML baselines (Ridge, SVR, Gradient Boosting).
2. **Model** — Fine-tune a MobileNetV4-Conv-Small backbone with a regression head on nail-bed ROI crops. Progressive unfreezing, aggressive augmentation, and Huber loss for small-data robustness. Export to TFLite with INT8 quantization.
3. **Flutter Plugin** — On-device inference engine wrapping TFLite via `dart:ffi` / platform channels, with real-time camera preprocessing.
4. **Example App** — A turnkey Flutter application demonstrating live Hb estimation from the device camera.

## Baseline Results (Notebook 03)

| Model | Test MAE (g/dL) | Test R² | CV MAE (g/dL) |
|-------|-----------------|---------|----------------|
| Ridge Regression | 1.575 | 0.384 | 1.635 |
| Random Forest | 1.610 | 0.307 | 1.602 |
| Gradient Boosting | 1.649 | 0.288 | 1.751 |
| Lasso Regression | 1.692 | 0.324 | 1.578 |
| SVR (RBF) | 1.774 | 0.231 | 1.845 |

*Hand-crafted color features on 51 dimensions. Target for DL model: MAE < 1.0 g/dL, R² > 0.60.*

## Key Features

- **Offline-first**: All inference runs on-device — no cloud dependency.
- **Sub-5ms latency**: INT8-quantized MobileNetV4-Conv-Small targeting mobile NPU/GPU delegates.
- **~1 MB model**: Edge-optimized — 2.5M params quantized to INT8.
- **Cross-platform**: Android (NNAPI) and iOS (Core ML delegate) support.
- **Privacy-preserving**: No images leave the device.

## Dataset

Based on the open-access fingernail bed image dataset from:

> Yakimov, P. et al. (2024). *Non-invasive hemoglobin estimation from fingernail bed images using deep learning.*

- 250 patients, 3 nail ROIs per patient → **750 cropped training images**
- Hb range: 4.4–16.9 g/dL
- Patient-level stratified splits: 70% train / 15% val / 15% test

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
python prepare_dataset.py                              # crop nail ROIs → data/processed/
python train.py --config configs/mobilenet_edge.yaml   # train edge model
python export_tflite.py --checkpoint checkpoints/best_model.pth --quantize int8
```

### 3. Run the Example App
```bash
cd example
flutter pub get
flutter run
```

## Architecture

```
Input (224×224×3)
  → MobileNetV4-Conv-Small (pretrained, timm)
  → Global Pool → [1280-dim]
  → Linear(1280→64) → ReLU → Dropout(0.4)
  → Linear(64→1) → Hb (g/dL)
```

**Training strategy for 750 samples:**
- Progressive unfreezing: head-only for 5 epochs, then full fine-tuning with 10× lower backbone LR
- Heavy augmentation: color jitter, rotation, random erasing, random resized crop
- Label noise (σ=0.1 g/dL) for regression smoothing
- Huber loss (δ=1.5) — robust to Hb measurement outliers
- Early stopping with patience=15

## Requirements

| Component       | Stack                              |
|-----------------|-------------------------------------|
| Research        | Python 3.10+, Jupyter, pandas, matplotlib, seaborn |
| Model Training  | PyTorch 2.x, timm, TFLite converter |
| Flutter Plugin  | Flutter 3.22+, Dart 3.4+           |
| On-device       | TFLite Runtime, Android NDK / Xcode |

## License

MIT — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome. Please open an issue before submitting a PR for major changes.
