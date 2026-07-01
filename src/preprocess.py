"""Preprocessing citra: grayscale -> resize 32x32 -> normalisasi -> augmentasi -> split.

Tahapan (tiap langkah dijelaskan di makalah):

1. Grayscale  : Y = 0.299 R + 0.587 G + 0.114 B  (dihitung manual dari kanal RGB).
2. Resize     : ke 32 x 32 piksel (ukuran masukan LeNet-5).
3. Normalisasi: skala ke [0,1] lalu standardisasi (x - mu) / sigma memakai
                statistik data latih.
4. Augmentasi : (hanya data latih) flip horizontal + pergeseran kecil, untuk
                memperbanyak variasi dan mengurangi overfitting.
5. Split      : train/val/test = 70/15/15, stratified (proporsi kelas terjaga).

Keluaran: data/processed/dataset.npz dan norm_stats.json.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def to_grayscale(rgb):
    """Y = 0.299R + 0.587G + 0.114B (manual, bukan konversi bawaan)."""
    r, g, b = config.GRAYSCALE_COEFF
    return rgb[..., 0] * r + rgb[..., 1] * g + rgb[..., 2] * b


def load_image(path):
    """Buka -> resize -> grayscale -> array [0,1] berbentuk (1, H, W)."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize(
            (config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR
        )
        arr = np.asarray(im, dtype=np.float64)      # (H, W, 3), 0..255
    gray = to_grayscale(arr) / 255.0                # (H, W) di [0,1]
    return gray[np.newaxis, :, :]                   # (1, H, W)


def read_labels():
    rows = []
    with open(config.LABELS_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["path"], int(r["label"])))
    return rows


def cap_per_class(rows, max_per_class):
    if not max_per_class:
        return rows
    counts, kept = {}, []
    for path, label in rows:
        if counts.get(label, 0) < max_per_class:
            kept.append((path, label))
            counts[label] = counts.get(label, 0) + 1
    return kept


# ----- Augmentasi manual (ruang [0,1]) -----
def hflip(img):
    return img[:, :, ::-1].copy()


def shift(img, dy, dx):
    """Geser citra sejauh (dy, dx) piksel, isi tepi dengan 0."""
    c, h, w = img.shape
    out = np.zeros_like(img)
    ys, ye = max(0, dy), min(h, h + dy)
    xs, xe = max(0, dx), min(w, w + dx)
    sys_, sye = max(0, -dy), min(h, h - dy)
    sxs, sxe = max(0, -dx), min(w, w - dx)
    out[:, ys:ye, xs:xe] = img[:, sys_:sye, sxs:sxe]
    return out


def augment(x, y):
    """Tambah flip horizontal + dua pergeseran kecil untuk tiap citra latih."""
    aug_x, aug_y = [x], [y]
    aug_x.append(np.stack([hflip(im) for im in x]))
    aug_y.append(y)
    aug_x.append(np.stack([shift(im, 2, 0) for im in x]))
    aug_y.append(y)
    aug_x.append(np.stack([shift(im, 0, -2) for im in x]))
    aug_y.append(y)
    return np.concatenate(aug_x), np.concatenate(aug_y)


def save_samples(x01, y, path):
    """Simpan grid contoh citra hasil preprocessing untuk makalah."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    fig, axes = plt.subplots(2, 6, figsize=(10, 3.6))
    rng = np.random.default_rng(0)
    for row, label in enumerate([0, 1]):
        idxs = rng.choice(np.where(y == label)[0], size=6, replace=False)
        for col, idx in enumerate(idxs):
            ax = axes[row, col]
            ax.imshow(x01[idx, 0], cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(config.CLASS_NAMES[label], fontsize=11)
    fig.suptitle("Contoh citra hasil preprocessing (grayscale 32x32)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    np.random.seed(config.SEED)
    rows = cap_per_class(read_labels(), config.MAX_PER_CLASS)
    print(f"Memuat {len(rows)} gambar ...")

    X, y = [], []
    for path, label in rows:
        abspath = os.path.join(config.ROOT_DIR, path)
        try:
            X.append(load_image(abspath))
            y.append(label)
        except Exception as exc:  # noqa: BLE001
            print(f"  lewati {path}: {exc}")
    X = np.stack(X).astype(np.float64)   # (N, 1, 32, 32) di [0,1]
    y = np.array(y, dtype=np.int64)
    print(f"Bentuk data: X={X.shape}, y={y.shape}")

    # Split stratified: pertama pisahkan test, lalu val dari sisa.
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        X, y, test_size=config.TEST_RATIO, stratify=y, random_state=config.SEED
    )
    val_frac = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=config.SEED
    )
    print(f"Split -> train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    # Simpan contoh citra (sebelum standardisasi, masih [0,1]).
    save_samples(x_train, y_train,
                 os.path.join(config.FIGURES_DIR, "preprocessing_samples.png"))

    # Augmentasi hanya pada data latih.
    if config.AUGMENT:
        x_train, y_train = augment(x_train, y_train)
        print(f"Setelah augmentasi: train={len(y_train)}")

    # Normalisasi: standardisasi dengan statistik data latih.
    mu = float(x_train.mean())
    sigma = float(x_train.std() + 1e-8)
    x_train = (x_train - mu) / sigma
    x_val = (x_val - mu) / sigma
    x_test = (x_test - mu) / sigma

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    np.savez_compressed(
        config.DATASET_NPZ,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
    )
    with open(config.NORM_STATS_JSON, "w") as f:
        json.dump({"mu": mu, "sigma": sigma}, f, indent=2)

    print(f"\nmu={mu:.4f}  sigma={sigma:.4f}")
    print(f"dataset.npz -> {config.DATASET_NPZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
