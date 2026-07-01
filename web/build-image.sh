#!/usr/bin/env bash
# Bangun image kontainer potholes & impor ke cluster k3d 'homelab'.
# Pemakaian: web/build-image.sh [TAG]   (default TAG=v1)
set -euo pipefail

TAG="${1:-v1}"
IMAGE="potholes:${TAG}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT

echo ">> Menyiapkan konteks build di $CTX"
mkdir -p "$CTX/app" "$CTX/www/content" "$CTX/www/figures" "$CTX/www/samples"

# --- Aplikasi (server + model) ---
cp "$ROOT/web/server.py" "$CTX/app/"
cp -r "$ROOT/src/cnn" "$CTX/app/cnn"
rm -rf "$CTX/app/cnn/__pycache__"
cp "$ROOT/experiments/weights.npz" "$CTX/app/weights.npz"
cp "$ROOT/data/processed/norm_stats.json" "$CTX/app/norm_stats.json"

# --- Statis ---
cp "$ROOT"/web/static/* "$CTX/www/"
cp "$ROOT"/experiments/figures/*.png "$CTX/www/figures/"
cp "$ROOT"/docs/*.md "$CTX/www/content/"
cp "$ROOT/makalah/makalah-deteksi-jalan-berlubang-cnn.md" "$CTX/www/content/makalah.md"

# --- Sampel demo (kecil) ---
python3 - "$ROOT" "$CTX" <<'PY'
import sys, glob, os
from PIL import Image
root, ctx = sys.argv[1], sys.argv[2]
for cls, short in [("pothole", "pothole"), ("normal", "normal")]:
    for i, f in enumerate(sorted(glob.glob(f"{root}/data/raw/{cls}/*.jpg"))[:3], 1):
        im = Image.open(f).convert("RGB"); im.thumbnail((256, 256))
        im.save(os.path.join(ctx, "www", "samples", f"{short}_{i}.jpg"), quality=85)
PY

cp "$ROOT/web/Dockerfile" "$CTX/Dockerfile"

echo ">> docker build -t $IMAGE"
docker build -t "$IMAGE" "$CTX"

echo ">> k3d image import $IMAGE -c homelab"
k3d image import "$IMAGE" -c homelab

echo ">> Selesai. Image '$IMAGE' tersedia di cluster homelab."
