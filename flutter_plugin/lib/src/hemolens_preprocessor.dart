import 'dart:typed_data';
import 'package:image/image.dart' as img;

/// Image preprocessing for HemoLens model input.
///
/// Converts raw camera RGB bytes into a normalized float32 tensor
/// matching the model's expected input format (224×224×3, ImageNet normalization).
class HemoLensPreprocessor {
  /// ImageNet normalization constants.
  static const List<double> mean = [0.485, 0.456, 0.406];
  static const List<double> std = [0.229, 0.224, 0.225];

  /// Preprocess raw RGB bytes into a float32 input tensor.
  ///
  /// Steps:
  /// 1. Decode raw bytes into image
  /// 2. Resize to [targetSize × targetSize]
  /// 3. Normalize with ImageNet mean/std
  /// 4. Return as [1, targetSize, targetSize, 3] float32 buffer
  static Float32List preprocess(
    Uint8List rgbBytes,
    int width,
    int height, {
    int targetSize = 224,
  }) {
    // Create image from raw bytes
    final image = img.Image.fromBytes(
      width: width,
      height: height,
      bytes: rgbBytes.buffer,
      numChannels: 3,
    );

    // Resize with bilinear interpolation
    final resized = img.copyResize(
      image,
      width: targetSize,
      height: targetSize,
      interpolation: img.Interpolation.linear,
    );

    // Allocate output tensor: [1, H, W, 3]
    final tensorSize = 1 * targetSize * targetSize * 3;
    final tensor = Float32List(tensorSize);

    int idx = 0;
    for (int y = 0; y < targetSize; y++) {
      for (int x = 0; x < targetSize; x++) {
        final pixel = resized.getPixel(x, y);

        // Normalize: (pixel / 255.0 - mean) / std
        tensor[idx++] = (pixel.r / 255.0 - mean[0]) / std[0]; // R
        tensor[idx++] = (pixel.g / 255.0 - mean[1]) / std[1]; // G
        tensor[idx++] = (pixel.b / 255.0 - mean[2]) / std[2]; // B
      }
    }

    return tensor;
  }

  /// Extract the nail-bed ROI from a camera frame.
  ///
  /// Uses center-crop heuristic — a more advanced version could use
  /// a lightweight detection model.
  static img.Image extractNailROI(img.Image frame, {double cropRatio = 0.6}) {
    final cropW = (frame.width * cropRatio).round();
    final cropH = (frame.height * cropRatio).round();
    final x = (frame.width - cropW) ~/ 2;
    final y = (frame.height - cropH) ~/ 2;

    return img.copyCrop(frame, x: x, y: y, width: cropW, height: cropH);
  }
}
