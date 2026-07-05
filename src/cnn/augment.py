"""Augmentasi daring (on-the-fly) ,  acak per-batch, per-epoch.

Berbeda dari augmentasi luring (offline) yang menghasilkan sekumpulan varian
tetap sekali di awal, augmentasi daring mengacak transformasi setiap kali batch
dilewatkan. Akibatnya model nyaris tak pernah melihat citra yang sama persis dua
kali, sehingga jauh lebih sulit menghafal data latih (anti-overfitting).

Semua transformasi ditulis manual dengan NumPy dan bekerja pada tensor citra
sudah terstandardisasi berbentuk (N, 1, H, W). Transformasi geometrik (flip,
geser) memakai padding replikasi tepi agar tidak memunculkan nilai "hitam" palsu
di ruang standardisasi; transformasi fotometrik (brightness, kontras, noise)
adalah operasi afin/aditif yang tetap sah di ruang standardisasi.
"""
from __future__ import annotations

import numpy as np


def _shift_replicate(img, dy, dx):
    """Geser citra (C, H, W) sejauh (dy, dx) piksel, isi tepi dengan replikasi.

    Memakai np.roll lalu menimpa baris/kolom yang "melipat" dengan nilai tepi
    (clamp), sehingga tak ada artefak nol pada tepian.
    """
    if dy == 0 and dx == 0:
        return img
    out = np.roll(img, shift=(dy, dx), axis=(1, 2))
    # Perbaiki tepi yang terlipat akibat np.roll dengan replikasi tepi.
    if dy > 0:
        out[:, :dy, :] = out[:, dy:dy + 1, :]
    elif dy < 0:
        out[:, dy:, :] = out[:, dy - 1:dy, :]
    if dx > 0:
        out[:, :, :dx] = out[:, :, dx:dx + 1]
    elif dx < 0:
        out[:, :, dx:] = out[:, :, dx - 1:dx]
    return out


def random_augment(x, rng, max_shift=3, brightness=0.15, contrast=0.15,
                   noise_std=0.05, flip_p=0.5):
    """Kembalikan salinan batch `x` (N, 1, H, W) yang teraugmentasi acak.

    Tiap citra diaugmentasi independen dengan:
      - flip horizontal (peluang `flip_p`)
      - geser integer dy, dx in [-max_shift, max_shift] (padding replikasi)
      - brightness: tambah konstanta U(-brightness, +brightness)
      - kontras: kali faktor U(1-contrast, 1+contrast)
      - noise Gaussian N(0, noise_std^2)

    `rng` adalah np.random.Generator agar reproducible mengikuti seed trainer.
    """
    n = x.shape[0]
    out = x.copy()
    for i in range(n):
        img = out[i]
        # Flip horizontal.
        if rng.random() < flip_p:
            img = img[:, :, ::-1]
        # Geser acak dengan replikasi tepi.
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        img = _shift_replicate(np.ascontiguousarray(img), dy, dx)
        # Fotometrik: kontras (kali) lalu brightness (tambah).
        if contrast:
            img = img * (1.0 + rng.uniform(-contrast, contrast))
        if brightness:
            img = img + rng.uniform(-brightness, brightness)
        # Noise Gaussian.
        if noise_std:
            img = img + rng.normal(0.0, noise_std, size=img.shape)
        out[i] = img
    return out
