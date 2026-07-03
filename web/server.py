"""Server ringan untuk situs + demo model deteksi jalan berlubang.

Hanya memakai pustaka standar Python + NumPy + Pillow (tanpa Flask). Endpoint:

    GET  /api/health           -> status & info model
    POST /api/predict          -> klasifikasi citra (body = byte gambar mentah)
    GET  /<path lain>          -> berkas statis dari POTHOLES_STATIC (bila diset)

Citra diproses sama persis dengan pipeline pelatihan: RGB, resize 48x48,
skala [0,1], standardisasi (x-mu)/sigma, lalu diumpankan ke ensembel LeNet-5
(rata-rata softmax beberapa seed + test-time augmentation flip horizontal).
Bobot per-seed dimuat dari weights_seed*.npz (fallback weights.npz tunggal).

Variabel lingkungan (opsional):
    POTHOLES_HOME     direktori berisi cnn/, weights.npz (+ weights_seed*.npz),
                      norm_stats.json
    POTHOLES_STATIC   direktori berkas statis untuk disajikan (mode situs penuh)
    POTHOLES_PORT     port dengar (default 8091)
    POTHOLES_BIND     alamat bind (default 127.0.0.1; pakai 0.0.0.0 di kontainer)
"""
from __future__ import annotations

import io
import json
import mimetypes
import os
import posixpath
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

STATIC_DIR = os.environ.get("POTHOLES_STATIC")  # None = nonaktif (mode API saja)

# ---------------------------------------------------------------------------
# Lokasi sumber daya (mendukung layout lokal maupun layout deploy)
# ---------------------------------------------------------------------------
HOME = os.environ.get("POTHOLES_HOME")
if HOME:
    sys.path.insert(0, HOME)
    WEIGHTS = os.path.join(HOME, "weights.npz")
    NORM = os.path.join(HOME, "norm_stats.json")
else:
    # Layout proyek lokal.
    PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJ)
    WEIGHTS = os.path.join(PROJ, "experiments", "weights.npz")
    NORM = os.path.join(PROJ, "data", "processed", "norm_stats.json")

try:
    from cnn.model import LeNet5  # layout deploy: /opt/potholes/cnn
except ModuleNotFoundError:
    from src.cnn.model import LeNet5  # layout lokal

CLASS_NAMES = ["normal", "pothole"]
IMG_SIZE = 48                 # cocok dengan config.IMG_SIZE saat pelatihan
IMG_CHANNELS = 3              # RGB
ENSEMBLE_SEEDS = [42, 7, 123] # rata-rata softmax beberapa model
MAX_BYTES = 8 * 1024 * 1024   # batas 8 MB per unggahan


def _ensemble_paths():
    """Berkas bobot per-seed bila tersedia (untuk ensembel); jika tidak, [WEIGHTS]."""
    paths = [WEIGHTS.replace(".npz", f"_seed{s}.npz") for s in ENSEMBLE_SEEDS]
    paths = [p for p in paths if os.path.exists(p)]
    return paths or [WEIGHTS]


# ---------------------------------------------------------------------------
# Muat model (ensembel) sekali saat startup
# ---------------------------------------------------------------------------
_PATHS = _ensemble_paths()
print(f"Memuat {len(_PATHS)} bobot ensembel ...")
MODELS = [
    LeNet5(seed=42, in_channels=IMG_CHANNELS, img_size=IMG_SIZE).load(p)
    for p in _PATHS
]
with open(NORM) as f:
    _stats = json.load(f)
MU, SIGMA = float(_stats["mu"]), float(_stats["sigma"])
print(f"Model siap: ensembel {len(MODELS)} model + TTA. mu={MU:.4f} sigma={SIGMA:.4f}")


def preprocess(image_bytes):
    """Byte gambar -> tensor (1,3,48,48) sesuai pipeline pelatihan (RGB 48x48)."""
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float64) / 255.0     # (H, W, 3) di [0,1]
    chw = np.transpose(arr, (2, 0, 1))                 # (3, H, W)
    x = (chw - MU) / SIGMA                             # standardisasi skalar global
    return x[np.newaxis, :, :, :]                      # (1, 3, H, W)


def _tta_proba(model, x):
    """Rata-rata softmax citra asli + flip horizontal (test-time augmentation)."""
    p = model.predict_proba(x)
    p_flip = model.predict_proba(x[:, :, :, ::-1])
    return 0.5 * (p + p_flip)


def predict(image_bytes):
    x = preprocess(image_bytes)
    # Ensembel: rata-ratakan softmax (dengan TTA) di seluruh model.
    proba = np.mean([_tta_proba(m, x) for m in MODELS], axis=0)[0]
    pred = int(np.argmax(proba))
    return {
        "pred": pred,
        "label": CLASS_NAMES[pred],
        "proba": {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "PotholesDemo/1.0"

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.rstrip("/") == "/api/health":
            self._send_json({
                "status": "ok",
                "model": f"ensembel {len(MODELS)}x LeNet-5 (NumPy from scratch) + TTA",
                "classes": CLASS_NAMES,
                "input": f"{IMG_SIZE}x{IMG_SIZE} RGB",
            })
        elif STATIC_DIR:
            self._serve_static(path)
        else:
            self._send_json({"error": "not found"}, 404)

    def _serve_static(self, path):
        """Sajikan berkas statis dari STATIC_DIR dengan proteksi path traversal."""
        rel = unquote(path).lstrip("/")
        if rel == "" or rel.endswith("/"):
            rel += "index.html"
        # Normalisasi & cegah keluar dari STATIC_DIR.
        safe = posixpath.normpath(rel)
        if safe.startswith("..") or safe.startswith("/"):
            self._send_json({"error": "forbidden"}, 403)
            return
        full = os.path.join(STATIC_DIR, safe)
        if not os.path.isfile(full):
            # Fallback ke index.html untuk rute tanpa berkas (SPA-friendly).
            full = os.path.join(STATIC_DIR, "index.html")
            if not os.path.isfile(full):
                self._send_json({"error": "not found"}, 404)
                return
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        try:
            with open(full, "rb") as f:
                body = f.read()
        except OSError:
            self._send_json({"error": "not found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # HTML selalu revalidasi (agar rilis baru tak tersangkut cache edge);
        # aset lain boleh di-cache karena URL-nya sudah diberi versi (?v=).
        if full.endswith(".html"):
            self.send_header("Cache-Control", "no-cache")
        else:
            self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.rstrip("/") != "/api/predict":
            self._send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_BYTES:
                self._send_json({"error": "ukuran tidak valid"}, 400)
                return
            data = self.rfile.read(length)
            result = predict(data)
            self._send_json(result)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"gagal memproses: {exc}"}, 400)

    def log_message(self, fmt, *args):  # senyapkan log akses default
        pass


def main():
    port = int(os.environ.get("POTHOLES_PORT", "8091"))
    bind = os.environ.get("POTHOLES_BIND", "127.0.0.1")
    server = ThreadingHTTPServer((bind, port), Handler)
    mode = "situs + API" if STATIC_DIR else "API saja"
    print(f"Server ({mode}) berjalan di http://{bind}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
