"""Pelabelan dataset: susun daftar (path, label) ke labels.csv.

Gambar mentah sudah dipisah per folder kelas oleh download_data.py:
    data/raw/pothole/*  -> label 1
    data/raw/normal/*   -> label 0

Skrip ini memverifikasi gambar dapat dibuka, menulis labels.csv, dan mencetak
ringkasan jumlah per kelas (untuk tabel "hasil pelabelan" pada makalah).
"""
from __future__ import annotations

import csv
import glob
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def collect_files(class_dir):
    files = []
    for pat in IMG_EXTS:
        files.extend(glob.glob(os.path.join(class_dir, pat)))
    return sorted(set(files))


def main():
    os.makedirs(config.LABELED_DIR, exist_ok=True)
    rows = []
    counts = {name: 0 for name in config.CLASS_NAMES}

    for label, name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(config.RAW_DIR, name)
        if not os.path.isdir(class_dir):
            print(f"PERINGATAN: folder kelas tidak ada: {class_dir}")
            continue
        for path in collect_files(class_dir):
            try:
                with Image.open(path) as im:
                    im.verify()  # cek berkas tidak korup
            except Exception as exc:  # noqa: BLE001
                print(f"  lewati (korup): {path} ({exc})")
                continue
            rows.append((os.path.relpath(path, config.ROOT_DIR), label))
            counts[name] += 1

    with open(config.LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)

    total = sum(counts.values())
    print("Hasil Pelabelan")
    print("===============")
    for name in config.CLASS_NAMES:
        label = config.CLASS_NAMES.index(name)
        pct = 100 * counts[name] / total if total else 0
        print(f"  {name:8s} (label {label}): {counts[name]:5d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':8s}          : {total:5d}")
    print(f"\nlabels.csv -> {config.LABELS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
