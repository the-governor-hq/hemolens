# flutter_plugin/

The core HemoLens Flutter plugin — wraps TFLite inference for on-device hemoglobin estimation.

Designed around the edge-optimized MobileNetV4-Conv-Small backbone (~1 MB INT8, <5 ms inference).

## Architecture

```
hemolens.dart (public API)
├── src/hemolens_engine.dart         # TFLite model loading + inference
├── src/hemolens_preprocessor.dart   # Camera frame → normalized tensor (224×224×3)
├── src/hemolens_result.dart         # Typed result with Hb value, confidence, severity
└── src/hemolens_config.dart         # Inference configuration (threads, GPU delegate)
```

## Usage

```dart
import 'package:hemolens/hemolens.dart';

final engine = HemoLensEngine();
await engine.initialize(modelPath: 'assets/hemolens_model_int8.tflite');

// From raw camera pixels
final result = await engine.estimateFromPixels(rgbBytes, width, height);
print('Hb: ${result.hbValue} g/dL (confidence: ${result.confidence})');
print('Severity: ${result.severity()}');  // normal / mild / moderate / severe
print('Latency: ${result.inferenceTimeMs} ms');

// Cleanup
engine.dispose();
```

## Features

- **Sub-5ms inference** with INT8 quantization on modern mobile devices
- **GPU delegate support** via `HemoLensConfig(useGpuDelegate: true)`
- **WHO severity classification** — normal, mild, moderate, severe anemia
- **Confidence scoring** based on physiological plausibility of predictions
- **ImageNet normalization** built into the preprocessing pipeline
