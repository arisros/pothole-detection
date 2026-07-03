"""Konfigurasi terpusat untuk seluruh pipeline deteksi jalan berlubang.

Semua hyperparameter, jalur (path), dan seed dikumpulkan di sini agar
eksperimen reproducible dan mudah diubah dari satu tempat.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Jalur direktori
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
LABELED_DIR = os.path.join(DATA_DIR, "labeled")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")
FIGURES_DIR = os.path.join(EXPERIMENTS_DIR, "figures")

LABELS_CSV = os.path.join(LABELED_DIR, "labels.csv")
DATASET_NPZ = os.path.join(PROCESSED_DIR, "dataset.npz")
NORM_STATS_JSON = os.path.join(PROCESSED_DIR, "norm_stats.json")
WEIGHTS_NPZ = os.path.join(EXPERIMENTS_DIR, "weights.npz")
HISTORY_CSV = os.path.join(EXPERIMENTS_DIR, "history.csv")

# ---------------------------------------------------------------------------
# Kelas
# ---------------------------------------------------------------------------
# 0 = normal (jalan utuh), 1 = pothole (jalan berlubang)
CLASS_NAMES = ["normal", "pothole"]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
IMG_SIZE = 48            # citra di-resize ke IMG_SIZE x IMG_SIZE
IMG_CHANNELS = 3         # 3 = RGB (lebih banyak sinyal), 1 = grayscale
# Koefisien luminance ITU-R BT.601 untuk grayscale: Y = 0.299R + 0.587G + 0.114B
GRAYSCALE_COEFF = (0.299, 0.587, 0.114)

# Pembagian data (stratified)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Augmentasi luring (offline, saat preprocess). Dimatikan: augmentasi kini
# dilakukan daring (on-the-fly, acak per-epoch) di dalam trainer, sehingga model
# tidak pernah melihat citra yang sama persis dua kali (anti-overfitting).
AUGMENT = False
# Augmentasi daring acak per-batch (flip, geser, brightness, kontras, noise).
ONLINE_AUGMENT = True

# ---------------------------------------------------------------------------
# Arsitektur (varian LeNet-5 modern untuk RGB 48x48, 2 kelas)
# ---------------------------------------------------------------------------
CONV1_FILTERS = 6
CONV2_FILTERS = 16
KERNEL_SIZE = 5
POOL_SIZE = 2
FC1_UNITS = 120
FC2_UNITS = 84

# ---------------------------------------------------------------------------
# Pelatihan
# ---------------------------------------------------------------------------
OPTIMIZER = "adam"       # "adam" atau "sgd"
LEARNING_RATE = 1e-3     # Adam: ~1e-3 (untuk SGD pakai ~0.01)
MOMENTUM = 0.9           # hanya dipakai bila OPTIMIZER="sgd"
COSINE_LR = True         # jadwal cosine: lr meluruh mulus sepanjang latih
BATCH_SIZE = 32
EPOCHS = 40              # sedikit lebih panjang; bobot val-terbaik dipulihkan
SEED = 42                # seed global agar hasil reproducible
ENSEMBLE_SEEDS = [42, 7, 123]   # ensembel: rata-rata softmax beberapa model

# ---------------------------------------------------------------------------
# Regularisasi (anti-overfitting)
# ---------------------------------------------------------------------------
WEIGHT_DECAY = 1e-4      # L2 weight decay (hanya pada bobot W, bukan bias)
DROPOUT_P = 0.3          # peluang dropout pada lapisan fully-connected

# Batas jumlah sampel per kelas saat memuat (None = pakai semua).
# Berguna menjaga pelatihan murni-NumPy tetap selesai dalam waktu wajar.
MAX_PER_CLASS = 1000
