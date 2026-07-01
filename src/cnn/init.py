"""Inisialisasi bobot.

Inisialisasi yang baik menjaga varians aktivasi tetap stabil antar lapisan
sehingga pelatihan tidak meledak/menghilang (exploding/vanishing).

- **He** (cocok untuk ReLU):    W ~ N(0, 2 / n_in)
- **Xavier/Glorot** (linier/softmax): W ~ N(0, 1 / n_in)

n_in = jumlah masukan ke satu neuron (fan-in). Untuk lapisan konvolusi,
n_in = C * KH * KW.
"""
from __future__ import annotations

import numpy as np


def he_normal(shape, fan_in, rng):
    """Inisialisasi He: std = sqrt(2 / fan_in)."""
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=shape).astype(np.float64)


def xavier_normal(shape, fan_in, rng):
    """Inisialisasi Xavier: std = sqrt(1 / fan_in)."""
    std = np.sqrt(1.0 / fan_in)
    return rng.normal(0.0, std, size=shape).astype(np.float64)
