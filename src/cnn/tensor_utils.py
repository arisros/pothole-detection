"""Utilitas im2col / col2im untuk operasi konvolusi.

Konvolusi naif berupa empat perulangan bersarang (filter x kanal x baris x kolom)
sangat lambat di Python. Trik *im2col* mengubah setiap jendela (window) konvolusi
menjadi satu kolom matriks, sehingga konvolusi menjadi satu perkalian matriks.

Bentuk loop naif yang setara (untuk satu citra, satu filter) adalah::

    for i in range(out_h):
        for j in range(out_w):
            wilayah = X[:, i*S : i*S+KH, j*S : j*S+KW]   # (C, KH, KW)
            O[f, i, j] = np.sum(wilayah * W[f]) + b[f]

im2col menyusun seluruh `wilayah` tersebut menjadi kolom-kolom matriks agar
operasi di atas dapat dihitung sekaligus dengan perkalian matriks W_col @ cols.
"""
from __future__ import annotations

import numpy as np


def conv_output_size(in_size: int, kernel: int, stride: int, pad: int) -> int:
    """Rumus ukuran feature map: (W - N + 2P) / S + 1 (slide ITB)."""
    return (in_size - kernel + 2 * pad) // stride + 1


def _get_im2col_indices(x_shape, kh, kw, stride, pad):
    """Hitung indeks (kanal, baris, kolom) untuk pengambilan seluruh jendela.

    Mengembalikan tiga array indeks yang, bila dipakai untuk meng-index citra
    ber-padding, langsung menghasilkan seluruh patch konvolusi.
    """
    n, c, h, w = x_shape
    out_h = conv_output_size(h, kh, stride, pad)
    out_w = conv_output_size(w, kw, stride, pad)

    # Indeks baris di dalam satu kernel, diulang untuk tiap kanal.
    i0 = np.repeat(np.arange(kh), kw)
    i0 = np.tile(i0, c)
    # Offset baris untuk tiap posisi jendela pada output.
    i1 = stride * np.repeat(np.arange(out_h), out_w)

    # Indeks kolom di dalam satu kernel, diulang untuk tiap kanal.
    j0 = np.tile(np.arange(kw), kh * c)
    # Offset kolom untuk tiap posisi jendela pada output.
    j1 = stride * np.tile(np.arange(out_w), out_h)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    # Indeks kanal untuk tiap baris matriks kolom.
    d = np.repeat(np.arange(c), kh * kw).reshape(-1, 1)
    return i, j, d, out_h, out_w


def im2col(x, kh, kw, stride, pad):
    """Ubah tensor citra (N, C, H, W) menjadi matriks kolom.

    Hasil berbentuk (C*KH*KW, N*out_h*out_w): tiap kolom = satu jendela konvolusi
    yang sudah diratakan. Urutan kolom: indeks spasial paling lambat, batch paling
    cepat (spatial * N + n).
    """
    x_padded = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant"
    )
    i, j, d, _, _ = _get_im2col_indices(x.shape, kh, kw, stride, pad)
    cols = x_padded[:, d, i, j]                 # (N, C*KH*KW, out_h*out_w)
    c = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(kh * kw * c, -1)
    return cols


def col2im(cols, x_shape, kh, kw, stride, pad):
    """Kebalikan im2col: akumulasikan gradien kolom kembali ke bentuk citra.

    Karena satu piksel masukan bisa muncul di banyak jendela (saat stride < kernel),
    kontribusinya dijumlahkan memakai ``np.add.at``.
    """
    n, c, h, w = x_shape
    h_pad, w_pad = h + 2 * pad, w + 2 * pad
    x_padded = np.zeros((n, c, h_pad, w_pad), dtype=cols.dtype)

    i, j, d, _, _ = _get_im2col_indices(x_shape, kh, kw, stride, pad)
    cols_reshaped = cols.reshape(c * kh * kw, -1, n).transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), d, i, j), cols_reshaped)

    if pad == 0:
        return x_padded
    return x_padded[:, :, pad:-pad, pad:-pad]
