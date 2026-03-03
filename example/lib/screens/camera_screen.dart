import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:hemolens/hemolens.dart';

class CameraScreen extends StatefulWidget {
  final HemoLensEngine engine;

  const CameraScreen({super.key, required this.engine});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _cameraController;
  bool _isCameraReady = false;
  bool _isProcessing = false;
  HemoLensResult? _lastResult;
  final List<HemoLensResult> _history = [];

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    // Prefer back camera
    final camera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    _cameraController = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await _cameraController!.initialize();
    setState(() => _isCameraReady = true);
  }

  Future<void> _captureAndEstimate() async {
    if (_isProcessing || _cameraController == null) return;

    setState(() => _isProcessing = true);

    try {
      final image = await _cameraController!.takePicture();
      final bytes = await image.readAsBytes();

      // For demo purposes — in production, convert CameraImage stream directly
      // This simplified flow uses captured JPEG bytes
      // TODO: Implement direct YUV420 → RGB conversion for real-time streaming
      final result = await widget.engine.estimateFromPixels(
        Uint8List.fromList(bytes),
        224, // placeholder width
        224, // placeholder height
      );

      setState(() {
        _lastResult = result;
        _history.insert(0, result);
        if (_history.length > 10) _history.removeLast();
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan Fingernail'),
        actions: [
          if (_history.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.history),
              onPressed: () => _showHistory(context),
            ),
        ],
      ),
      body: Column(
        children: [
          // Camera preview
          Expanded(
            flex: 3,
            child: _isCameraReady
                ? Stack(
                    alignment: Alignment.center,
                    children: [
                      CameraPreview(_cameraController!),
                      // Nail placement guide
                      Container(
                        width: 200,
                        height: 120,
                        decoration: BoxDecoration(
                          border: Border.all(
                            color: Colors.white.withOpacity(0.8),
                            width: 2,
                          ),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Center(
                          child: Text(
                            'Place fingernail here',
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.9),
                              fontSize: 12,
                              shadows: [
                                Shadow(blurRadius: 4, color: Colors.black54),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ],
                  )
                : const Center(child: CircularProgressIndicator()),
          ),

          // Results panel
          Expanded(
            flex: 2,
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: theme.colorScheme.surfaceContainerHighest,
                borderRadius:
                    const BorderRadius.vertical(top: Radius.circular(24)),
              ),
              child: Column(
                children: [
                  if (_lastResult != null) ...[
                    // Hb value
                    Text(
                      '${_lastResult!.hbValue.toStringAsFixed(1)} g/dL',
                      style: theme.textTheme.displaySmall?.copyWith(
                        fontWeight: FontWeight.bold,
                        color: _getHbColor(_lastResult!),
                      ),
                    ),
                    const SizedBox(height: 8),
                    // Severity chip
                    Chip(
                      label: Text(
                        _lastResult!.severity().toUpperCase(),
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                      backgroundColor: _getHbColor(_lastResult!).withOpacity(0.2),
                    ),
                    const SizedBox(height: 8),
                    // Metadata
                    Text(
                      'Confidence: ${(_lastResult!.confidence * 100).toStringAsFixed(0)}% · '
                      '${_lastResult!.inferenceTimeMs}ms',
                      style: theme.textTheme.bodySmall,
                    ),
                  ] else
                    Text(
                      'Tap the button to estimate\nhemoglobin level',
                      textAlign: TextAlign.center,
                      style: theme.textTheme.bodyLarge?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),

                  const Spacer(),

                  // Capture button
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton.icon(
                      onPressed:
                          _isCameraReady && !_isProcessing
                              ? _captureAndEstimate
                              : null,
                      icon: _isProcessing
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: Colors.white,
                              ),
                            )
                          : const Icon(Icons.camera),
                      label: Text(_isProcessing ? 'Analyzing...' : 'Capture'),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Color _getHbColor(HemoLensResult result) {
    switch (result.severity()) {
      case 'severe':
        return Colors.red.shade700;
      case 'moderate':
        return Colors.orange.shade700;
      case 'mild':
        return Colors.amber.shade700;
      default:
        return Colors.green.shade700;
    }
  }

  void _showHistory(BuildContext context) {
    showModalBottomSheet(
      context: context,
      builder: (ctx) => ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: _history.length,
        itemBuilder: (ctx, i) {
          final r = _history[i];
          return ListTile(
            leading: CircleAvatar(
              backgroundColor: _getHbColor(r).withOpacity(0.2),
              child: Text(
                r.hbValue.toStringAsFixed(1),
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: _getHbColor(r),
                ),
              ),
            ),
            title: Text('${r.hbValue.toStringAsFixed(1)} g/dL'),
            subtitle: Text(
              '${r.severity()} · ${r.inferenceTimeMs}ms · '
              '${r.timestamp.hour}:${r.timestamp.minute.toString().padLeft(2, '0')}',
            ),
          );
        },
      ),
    );
  }
}
