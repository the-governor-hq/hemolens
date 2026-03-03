/// Typed result from HemoLens inference.
class HemoLensResult {
  /// Estimated hemoglobin level in g/dL.
  final double hbValue;

  /// Confidence score in [0.0, 1.0] based on physiological plausibility.
  final double confidence;

  /// Inference latency in milliseconds.
  final int inferenceTimeMs;

  /// Timestamp of the prediction.
  final DateTime timestamp;

  const HemoLensResult({
    required this.hbValue,
    required this.confidence,
    required this.inferenceTimeMs,
    required this.timestamp,
  });

  /// Whether the estimated Hb suggests anemia (WHO thresholds).
  ///
  /// Female: < 12.0 g/dL, Male: < 13.0 g/dL
  bool isAnemic({bool isMale = false}) {
    return isMale ? hbValue < 13.0 : hbValue < 12.0;
  }

  /// Severity classification based on WHO guidelines.
  ///
  /// - Severe: < 7.0 g/dL
  /// - Moderate: 7.0–9.9 g/dL
  /// - Mild: 10.0–11.9/12.9 g/dL (F/M)
  /// - Normal: >= 12.0/13.0 g/dL (F/M)
  String severity({bool isMale = false}) {
    if (hbValue < 7.0) return 'severe';
    if (hbValue < 10.0) return 'moderate';
    final threshold = isMale ? 13.0 : 12.0;
    if (hbValue < threshold) return 'mild';
    return 'normal';
  }

  @override
  String toString() =>
      'HemoLensResult(hb=${hbValue.toStringAsFixed(1)} g/dL, '
      'confidence=${(confidence * 100).toStringAsFixed(0)}%, '
      '${inferenceTimeMs}ms)';

  /// Convert to a JSON-compatible map.
  Map<String, dynamic> toJson() => {
        'hb_value': hbValue,
        'confidence': confidence,
        'inference_time_ms': inferenceTimeMs,
        'timestamp': timestamp.toIso8601String(),
      };
}
