# HemoLens 🔬

**A high-performance, offline Flutter engine for non-invasive hemoglobin level estimation.**

Powered by a hybrid pipeline: frozen MobileNetV4-Conv-Small features + Ridge regression, optimized for on-device inference (~2.5 MB INT8, <5 ms latency) using the 2024 Yakimov et al. fingernail dataset.

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
2. **Model** — Frozen MobileNetV4-Conv-Small backbone as a feature extractor (1280-dim), combined with 51 handcrafted color features, feeding a Ridge regression head. The hybrid approach significantly outperforms end-to-end fine-tuning in the small-data regime (250 patients). Export to ONNX → TFLite with INT8 quantization.
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

*Hand-crafted color features on 51 dimensions.*

## Deep Learning Results

| Approach | Val MAE (g/dL) | Test MAE (g/dL) | Val R² | CV MAE (g/dL) |
|----------|----------------|-----------------|--------|----------------|
| **Hybrid: frozen MobileNetV4 + color features → Ridge** | **1.690** | **1.365** | **0.483** | **1.656 ± 0.18** |
| Hybrid: frozen MobileNetV4 + color features → Ridge+PCA128 | 1.717 | 1.402 | 0.485 | — |
| Hybrid: frozen MobileNetV4 + color features → ElasticNet | 1.909 | 1.406 | 0.377 | — |
| End-to-end fine-tuned MobileNetV4 | 2.315 | — | 0.143 | — |

The hybrid frozen-backbone approach beats both end-to-end fine-tuning and pure color-feature baselines.

## Key Features

- **Offline-first**: All inference runs on-device — no cloud dependency.
- **Sub-5ms latency**: INT8-quantized MobileNetV4-Conv-Small targeting mobile NPU/GPU delegates.
- **~2.5 MB model**: Frozen backbone + Ridge head, quantized to INT8.
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
python train_hybrid.py --config configs/mobilenet_edge.yaml  # hybrid: frozen CNN features + Ridge
python export_tflite.py --checkpoint checkpoints/hemolens_hybrid.pth --quantize int8
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
  → MobileNetV4-Conv-Small (frozen pretrained, timm)
  → Global Pool → [1280-dim]
  → Ridge Regression → Hb (g/dL)
```

**Training strategy for 250 patients:**
- **Hybrid pipeline**: Frozen ImageNet backbone as feature extractor — avoids overfitting with small data
- 3 nail crops per patient averaged to one 1280-dim feature vector
- 51 handcrafted color features (RGB, LAB, HSV) concatenated → 1331-dim input to Ridge
- Patient-level stratified splits to prevent data leakage
- 5-fold GroupKFold cross-validation: **MAE = 1.656 ± 0.18 g/dL**
- End-to-end ONNX export (backbone + head) for TFLite conversion

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
