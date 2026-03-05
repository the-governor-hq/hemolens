"""
Minimal HTTPS dev server for HemoLens web demo.
Camera (getUserMedia) requires a secure context — either HTTPS or localhost.

Usage:
    python serve.py              # serves on https://localhost:8443
    python serve.py --port 3000  # custom port
"""

import argparse
import http.server
import os
import ssl
import subprocess
import sys
import tempfile
from pathlib import Path


def generate_self_signed_cert():
    """Generate a self-signed cert using OpenSSL (if available) or fall back to HTTP."""
    cert_dir = Path(tempfile.gettempdir()) / "hemolens-dev-cert"
    cert_dir.mkdir(exist_ok=True)
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    if cert_file.exists() and key_file.exists():
        return str(cert_file), str(key_file)

    try:
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_file), "-out", str(cert_file),
            "-days", "365", "-nodes",
            "-subj", "/CN=localhost"
        ], check=True, capture_output=True)
        return str(cert_file), str(key_file)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def main():
    parser = argparse.ArgumentParser(description="HemoLens Web Demo Server")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--http", action="store_true", help="Use plain HTTP (camera won't work except on localhost)")
    args = parser.parse_args()

    # Change to web-demo directory
    web_dir = Path(__file__).parent
    os.chdir(web_dir)

    # Set up CORS and MIME types
    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {
            **http.server.SimpleHTTPRequestHandler.extensions_map,
            ".onnx": "application/octet-stream",
            ".wasm": "application/wasm",
            ".js": "application/javascript",
            ".mjs": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
        }

        def end_headers(self):
            # CORS headers
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            super().end_headers()

    server = http.server.HTTPServer(("0.0.0.0", args.port), Handler)

    if not args.http:
        cert_file, key_file = generate_self_signed_cert()
        if cert_file:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert_file, key_file)
            server.socket = ctx.wrap_socket(server.socket, server_side=True)
            protocol = "https"
        else:
            print("[WARN] OpenSSL not found. Falling back to HTTP.")
            print("[WARN] Camera will only work on localhost, not LAN IPs.")
            protocol = "http"
    else:
        protocol = "http"

    url = f"{protocol}://localhost:{args.port}"
    print(f"\n  HemoLens Web Demo")
    print(f"  {url}\n")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
