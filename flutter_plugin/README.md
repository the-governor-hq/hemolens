# flutter_plugin/

The core HemoLens Flutter plugin — wraps TFLite inference for on-device hemoglobin estimation.

## Architecture

```
hemolens.dart (public API)
├── src/hemolens_inference.dart      # TFLite model loading + inference
├── src/hemolens_preprocessor.dart   # Camera frame → tensor conversion
├── src/hemolens_result.dart         # Typed result model
└── src/hemolens_platform_interface.dart
```

## Usage

```dart
import 'package:hemolens/hemolens.dart';

final engine = HemoLensEngine();
await engine.initialize(modelPath: 'assets/hemolens_model_int8.tflite');

// From camera frame
final result = await engine.estimateFromImage(cameraImage);
print('Hb: ${result.hbValue} g/dL (confidence: ${result.confidence})');

// Cleanup
engine.dispose();
```
