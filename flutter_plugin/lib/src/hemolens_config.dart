/// Configuration for HemoLens inference engine.
class HemoLensConfig {
  /// Number of threads for TFLite interpreter.
  final int numThreads;

  /// Whether to use GPU delegate for inference.
  final bool useGpuDelegate;

  /// Model input size (square).
  final int inputSize;

  /// Whether to apply nail-bed ROI extraction before inference.
  final bool autoExtractROI;

  const HemoLensConfig({
    this.numThreads = 4,
    this.useGpuDelegate = false,
    this.inputSize = 224,
    this.autoExtractROI = true,
  });

  @override
  String toString() =>
      'HemoLensConfig(threads=$numThreads, gpu=$useGpuDelegate, '
      'inputSize=$inputSize, autoROI=$autoExtractROI)';
}
