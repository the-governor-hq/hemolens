# web-demo/

Browser-based PWA for HemoLens — runs two ONNX models directly in the browser using [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).

## Quick Start

```bash
cd web-demo
python serve.py
```

Then open **https://localhost:8443** (accept the self-signed certificate warning).

For plain HTTP: `python serve.py --http --port 8080` (camera only works on localhost).

## Features

- **Automatic nail detection**: YOLOv8-nano finds nail bounding boxes — no manual alignment needed
- **Two-stage inference**: Nail detection (320×320) → Hb regression (224×224), both ONNX
- **Multi-frame capture**: 30 frames → IQR outlier rejection → median Hb estimate
- **Image upload**: Upload a photo → detects all nails → runs Hb on each → median result
- **Offline-capable PWA**: Service worker caches both models + all assets
- **On-device only**: No images are ever uploaded — everything stays in the browser
- **Graceful fallback**: If nail detector fails to load, falls back to SVG finger guide overlay
- **Front/rear camera** toggle with flash/torch support
- **Mobile-first** responsive design

## How It Works

1. Two ONNX models load in the browser via WASM:
   - `nail_detector.onnx` (11.6 MB) — YOLOv8-nano for nail detection
   - `hemolens_hybrid_web.onnx` (~10 MB) — MobileNetV4 + Ridge for Hb regression
2. **Camera mode**: User presses capture → 30 frames grabbed at ~12.5 fps → per frame: detect best nail → crop → Hb inference → collect predictions → IQR filter → median
3. **Upload mode**: Detector finds all nails in the image → runs Hb on each crop → reports median with per-nail breakdown
4. Result displayed with WHO severity badge (Severe / Moderate / Mild / Normal) and confidence stats (std dev, range, frames used)

## Files

| File | Description |
|------|-------------|
| `index.html` | PWA shell — camera UI, result display, SVG guide overlay |
| `app.js` | Main app logic — model loading, capture, inference, UI updates |
| `nail_detector.js` | `NailDetector` class — YOLOv8 ONNX inference, letterbox, NMS |
| `style.css` | Responsive styles |
| `sw.js` | Service worker — caches models + assets for offline use |
| `serve.py` | Dev server (HTTPS with self-signed cert, or HTTP) |
| `manifest.json` | PWA manifest |
| `model/hemolens_hybrid_web.onnx` | Hb regression model (~10 MB) |
| `model/nail_detector.onnx` | Nail detection model (11.6 MB) |

## Requirements

- Modern browser with WebAssembly support (Chrome, Firefox, Safari, Edge)
- Camera access (requires HTTPS or localhost)
- Python 3 (for the dev server) — or any static file server with HTTPS

## Notes

- The `serve.py` script auto-generates a self-signed HTTPS certificate (requires OpenSSL)
- Use `--http` flag to serve plain HTTP (camera only works on localhost)
- Total model payload is ~22 MB — first load may take a few seconds on slow connections
- Service worker (v2) caches everything for offline use after first load
