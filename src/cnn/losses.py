"""Fungsi loss: Softmax + Cross-Entropy.

Softmax mengubah skor mentah z menjadi peluang:

    p_i = e^{z_i} / sum_j e^{z_j}

Cross-entropy untuk label benar y (one-hot):

    L = - sum_i y_i ln(p_i)

Gabungan keduanya memberi gradien yang sangat sederhana dan stabil:

    dL/dz = p - y

Itulah alasan softmax dan cross-entropy selalu diturunkan bersama.
"""
from __future__ import annotations

import numpy as np


def softmax(z):
    """Softmax stabil-numerik (kurangi maksimum tiap baris sebelum eksponen)."""
    z_shift = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z_shift)
    return exp / exp.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """Loss softmax + cross-entropy untuk klasifikasi multi-kelas."""

    def __init__(self):
        self._probs = None
        self._y = None
        self._n = None

    def forward(self, scores, y):
        """Hitung loss rata-rata.

        scores : (N, K) skor mentah dari lapisan terakhir.
        y      : (N,) indeks kelas benar (integer).
        """
        self._probs = softmax(scores)
        self._y = y
        self._n = scores.shape[0]

        # Ambil peluang kelas benar; tambahkan epsilon agar log aman.
        correct_logprobs = -np.log(self._probs[np.arange(self._n), y] + 1e-12)
        return float(np.mean(correct_logprobs))

    def backward(self):
        """Gradien gabungan: (p - y) / N."""
        grad = self._probs.copy()
        grad[np.arange(self._n), self._y] -= 1.0
        grad /= self._n
        return grad
