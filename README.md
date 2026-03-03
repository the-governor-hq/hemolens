# HemoLens 🔬

**A high-performance, offline Flutter engine for non-invasive hemoglobin level estimation.**

Powered by Vision Transformers (ViT) and optimized for on-device inference using the 2024 Yakimov et al. fingernail dataset.

---

## Project Structure

```
HemoLens/
├── research/          # Jupyter Notebooks, EDA, and metadata analysis
├── model/             # PyTorch/Keras training scripts & TFLite export
├── flutter_plugin/    # Dart/C++ code for on-device inference
├── example/           # Demo Flutter app showcasing real-time detection
└── assets/            # Quantized .tflite models and labels
```

## Overview

HemoLens estimates hemoglobin (Hb) levels from smartphone camera images of fingernail beds — no blood draw required. The pipeline:

1. **Research** — Exploratory data analysis on the Yakimov et al. (2024) fingernail image dataset, feature extraction, and baseline modeling.
2. **Model** — Train a Vision Transformer (ViT) regression head in PyTorch, fine-tune on fingernail ROI crops, and export to TFLite with INT8 quantization.
3. **Flutter Plugin** — On-device inference engine wrapping TFLite via `dart:ffi` / platform channels, with real-time camera preprocessing.
4. **Example App** — A turnkey Flutter application demonstrating live Hb estimation from the device camera.

## Key Features

- **Offline-first**: All inference runs on-device — no cloud dependency.
- **Sub-50ms latency**: INT8-quantized ViT model targeting mobile NPU/GPU delegates.
- **Cross-platform**: Android (NNAPI) and iOS (Core ML delegate) support.
- **Privacy-preserving**: No images leave the device.

## Dataset

Based on the open-access fingernail bed image dataset from:

> Yakimov, P. et al. (2024). *Non-invasive hemoglobin estimation from fingernail bed images using deep learning.* 

The dataset includes labeled fingernail images with corresponding Hb values (g/dL) from capillary blood samples.

## Quick Start

### 1. Research & EDA
```bash
cd research
pip install -r requirements.txt
jupyter lab
```

### 2. Train the Model
```bash
cd model
pip install -r requirements.txt
python train.py --config configs/vit_base.yaml
python export_tflite.py --checkpoint best_model.pth --quantize int8
```

### 3. Run the Example App
```bash
cd example
flutter pub get
flutter run
```

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
