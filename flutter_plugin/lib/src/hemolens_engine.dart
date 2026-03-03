import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'hemolens_config.dart';
import 'hemolens_preprocessor.dart';
import 'hemolens_result.dart';

/// Core inference engine for HemoLens.
///
/// Handles TFLite model loading, input preprocessing, inference execution,
/// and result interpretation.
///
/// ```dart
/// final engine = HemoLensEngine();
/// await engine.initialize();
/// final result = await engine.estimateFromPixels(rgbBytes, width, height);
/// engine.dispose();
/// ```
class HemoLensEngine {
  Interpreter? _interpreter;
  bool _isInitialized = false;

  /// Whether the engine is ready for inference.
  bool get isInitialized => _isInitialized;

  /// Initialize the TFLite interpreter.
  ///
  /// [modelPath] — Path to the .tflite model asset.
  /// [config] — Optional inference configuration.
  Future<void> initialize({
    String modelPath = 'assets/hemolens_model_int8.tflite',
    HemoLensConfig config = const HemoLensConfig(),
  }) async {
    final options = InterpreterOptions()..threads = config.numThreads;

    // Enable GPU delegate if requested
    if (config.useGpuDelegate) {
      // Platform-specific delegate setup handled by tflite_flutter
      options.addDelegate(GpuDelegateV2());
    }

    _interpreter = await Interpreter.fromAsset(modelPath, options: options);
    _isInitialized = true;

    final inputShape = _interpreter!.getInputTensor(0).shape;
    final outputShape = _interpreter!.getOutputTensor(0).shape;
    print('[HemoLens] Model loaded: input=$inputShape, output=$outputShape');
  }

  /// Estimate hemoglobin from raw RGB pixel data.
  ///
  /// [rgbBytes] — Raw RGB pixel bytes (H × W × 3).
  /// [width] — Image width in pixels.
  /// [height] — Image height in pixels.
  Future<HemoLensResult> estimateFromPixels(
    Uint8List rgbBytes,
    int width,
    int height,
  ) async {
    _ensureInitialized();

    final stopwatch = Stopwatch()..start();

    // Preprocess: resize, normalize, convert to float32 tensor
    final inputTensor = HemoLensPreprocessor.preprocess(
      rgbBytes,
      width,
      height,
      targetSize: 224,
    );

    // Run inference
    final outputBuffer = Float32List(1);
    _interpreter!.run(inputTensor, outputBuffer);

    stopwatch.stop();

    final hbValue = outputBuffer[0];

    return HemoLensResult(
      hbValue: hbValue,
      confidence: _computeConfidence(hbValue),
      inferenceTimeMs: stopwatch.elapsedMilliseconds,
      timestamp: DateTime.now(),
    );
  }

  /// Compute a simple confidence score based on physiological plausibility.
  ///
  /// Normal Hb range: 7–20 g/dL. Values at extremes get lower confidence.
  double _computeConfidence(double hbValue) {
    const minPlausible = 4.0;
    const maxPlausible = 22.0;
    const normalLow = 10.0;
    const normalHigh = 17.0;

    if (hbValue < minPlausible || hbValue > maxPlausible) return 0.0;
    if (hbValue >= normalLow && hbValue <= normalHigh) return 1.0;

    if (hbValue < normalLow) {
      return (hbValue - minPlausible) / (normalLow - minPlausible);
    }
    return (maxPlausible - hbValue) / (maxPlausible - normalHigh);
  }

  /// Release model resources.
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isInitialized = false;
    print('[HemoLens] Engine disposed.');
  }

  void _ensureInitialized() {
    if (!_isInitialized || _interpreter == null) {
      throw StateError(
        'HemoLensEngine not initialized. Call initialize() first.',
      );
    }
  }
}
