/* ================================================================
   HemoLens — Service Worker (offline-first PWA)
   ================================================================
   Strategy:
     - Cache-first for app shell (HTML, CSS, JS, icons, model)
     - Network-first for external resources (fonts, CDN scripts)
     - Stale-while-revalidate for model updates
   ================================================================ */

const CACHE_NAME = "hemolens-v4";

// App shell — files that must be cached for offline use
const APP_SHELL = [
  "./",
  "./index.html",
  "./style.css",
  "./app.js",
  "./nail_detector.js",
  "./manifest.json",
  "./model/hemolens_hybrid_web.onnx",
  "./model/nail_detector.onnx",
  "./icons/icon-192x192.png",
  "./icons/icon-512x512.png",
  "./icons/apple-touch-icon.png",
  "./icons/favicon-32x32.png",
  "./icons/favicon.ico",
];

// External resources we also want to cache (best-effort)
const EXTERNAL_URLS = [
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort.min.js",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-simd.wasm",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm.wasm",
];

// ─── Install ───
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(async (cache) => {
      // Cache app shell (must succeed)
      await cache.addAll(APP_SHELL);
      // Cache external resources (best-effort, don't block install)
      for (const url of EXTERNAL_URLS) {
        try {
          await cache.add(url);
        } catch (e) {
          console.warn("[SW] Could not cache external:", url, e);
        }
      }
    })
  );
  // Do NOT call skipWaiting() here — activation is triggered by the page
  // via a SKIP_WAITING message (or falls through to normal browser takeover).
  // This prevents the new SW from cutting in while a scan is in progress.
  // If you want instant takeover without user prompt, uncomment the line below:
  // self.skipWaiting();
});

// ─── Activate ───
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => {
        // Claim all open clients immediately
        return self.clients.claim();
      })
      .then(() => {
        // Notify all clients that a new version is active — they will reload
        return self.clients.matchAll({ type: "window" }).then((clients) => {
          clients.forEach((client) => client.postMessage({ type: "SW_UPDATED" }));
        });
      })
  );
});

// ─── Message handling (controlled activation) ───
self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});

// ─── Fetch ───
self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Skip non-GET requests
  if (request.method !== "GET") return;

  // For navigation requests (HTML pages), use network-first
  if (request.mode === "navigate") {
    event.respondWith(networkFirst(request));
    return;
  }

  // ONNX model — stale-while-revalidate: serve cached copy instantly,
  // fetch fresh version in the background so next visit gets the update.
  if (request.url.includes(".onnx")) {
    event.respondWith(staleWhileRevalidate(request));
    return;
  }

  // Google Fonts — stale-while-revalidate
  if (request.url.includes("fonts.googleapis.com") ||
      request.url.includes("fonts.gstatic.com")) {
    event.respondWith(staleWhileRevalidate(request));
    return;
  }

  // Everything else — cache-first with network fallback
  event.respondWith(cacheFirst(request));
});

// ─── Caching strategies ───

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    return new Response("Offline", { status: 503 });
  }
}

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    return cached || new Response("Offline", { status: 503, headers: { "Content-Type": "text/plain" } });
  }
}

async function staleWhileRevalidate(request) {
  const cached = await caches.match(request);
  const fetchPromise = fetch(request).then((response) => {
    if (response.ok) {
      caches.open(CACHE_NAME).then((cache) => cache.put(request, response.clone()));
    }
    return response;
  }).catch(() => cached);

  return cached || fetchPromise;
}
