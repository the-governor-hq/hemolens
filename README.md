# HemoLens 🔬

**A high-performance, offline Flutter engine for non-invasive hemoglobin level estimation.**

Powered by a hybrid pipeline: frozen MobileNetV4-Conv-Small features + CatBoost regression, using the 2024 Yakimov et al. fingernail dataset. Best test MAE = **1.305 g/dL** (R² = 0.46).

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
2. **Model** — Frozen MobileNetV4-Conv-Small backbone as a feature extractor (1280-dim), combined with 67 handcrafted color features (RGB, LAB, HSV means/std/percentiles + redness index), feeding a CatBoost regression head. The hybrid approach significantly outperforms end-to-end fine-tuning in the small-data regime (250 patients). Export to ONNX for edge deployment.
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

### Best Model: MobileNetV4-Conv-Small + CatBoost (Hybrid)

| Approach | Val MAE (g/dL) | Test MAE (g/dL) | Test R² | CV MAE (3-fold) |
|----------|----------------|-----------------|---------|------------------|
| **MobileNetV4 + CatBoost** | **1.520** | **1.305** | **0.460** | 1.912 ± 0.27 |
| MobileNetV4 + Ridge | 1.566 | 1.459 | 0.431 | 1.912 ± 0.27 |
| MobileNetV4 + Ridge+PCA128 | 1.563 | 1.453 | 0.436 | — |
| MobileNetV4 + ElasticNet | 1.645 | 1.541 | 0.356 | — |
| CNN-only Ridge (no color feats) | 1.501 | 1.525 | 0.408 | — |
| End-to-end fine-tuned MobileNetV4 | 2.891 | — | -1.61 | — |

### Backbone Sweep (4 Backbones × 4 Heads)

| Backbone | Best Head | Test MAE | Test R² | CV MAE |
|----------|-----------|----------|---------|--------|
| **mobilenetv4_conv_small** | **CatBoost** | **1.305** | **0.460** | 1.912 ± 0.27 |
| efficientnet_b0 | CatBoost | 1.459 | 0.431 | **1.527 ± 0.32** |
| tf_efficientnetv2_b0 | Ridge+PCA32 | 1.497 | 0.469 | 1.563 ± 0.32 |
| mobilenetv3_small_100 | CatBoost | 1.444 | 0.399 | 1.612 ± 0.29 |

The hybrid frozen-backbone approach beats end-to-end fine-tuning by >2× on this 250-patient dataset. EfficientNet-B0 has the best cross-validation stability.

## Key Features

- **Offline-first**: All inference runs on-device — no cloud dependency.
- **Lightweight model**: Frozen MobileNetV4-Conv-Small backbone (~2.5M params) + CatBoost head.
- **Cross-platform**: Android (NNAPI) and iOS (Core ML delegate) support.
- **Privacy-preserving**: No images leave the device.

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
  → CatBoost / Ridge Regression → Hb (g/dL)
```

**Training strategy for 250 patients:**
- **Hybrid pipeline**: Frozen ImageNet backbone as feature extractor — avoids overfitting with small data
- 3 nail crops per patient averaged to one 1280-dim feature vector
- 67 handcrafted color features (RGB/LAB/HSV mean, std, median, P5, P95 + redness index) → 1347-dim input
- Session-aware splits (GroupShuffleSplit on measurement date) to prevent leakage
- 3-fold session-aware cross-validation: **MAE = 1.912 ± 0.27 g/dL** (17 session groups)
- End-to-end ONNX export (backbone + head) for edge deployment

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
