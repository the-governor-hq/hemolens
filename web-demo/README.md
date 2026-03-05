# web-demo/

Browser-based demo for HemoLens — runs the ONNX model directly in the browser using [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).

## Quick Start

```bash
cd web-demo
python serve.py
```

Then open **https://localhost:8443** (accept the self-signed certificate warning).

## Features

- **Live camera** with SVG overlay guiding finger placement
- **On-device inference** via ONNX Runtime Web (WASM) — no server-side computation
- **Real-time result** with WHO severity classification and animated gauge
- **Image upload** support for testing with saved images
- **Front/rear camera** toggle
- **Mobile-first** responsive design

## How It Works

1. The ONNX model (`hemolens_hybrid_web.onnx`, ~10 MB) loads in the browser via WASM
2. User positions their fingernail in the guide overlay
3. On capture, the nail region is cropped from the camera feed
4. Image is resized (shorter side → 256), center-cropped to 224×224, and normalized (ImageNet stats)
5. Inference runs client-side → outputs Hb estimation in g/dL
6. Result displayed with severity badge (Severe / Moderate / Mild / Normal)

## Requirements

- Modern browser with WebAssembly support (Chrome, Firefox, Safari, Edge)
- Camera access (requires HTTPS or localhost)
- Python 3 (for the dev server) — or any static file server with HTTPS

## Notes

- The `serve.py` script auto-generates a self-signed HTTPS certificate (requires OpenSSL)
- Use `--http` flag to serve plain HTTP (camera only works on localhost)
- The model file is ~10 MB — first load may take a few seconds on slow connections
