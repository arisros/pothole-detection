"""Optimizer: Stochastic Gradient Descent dengan momentum (SGDM).

Aturan pembaruan untuk tiap parameter theta:

    v <- mu * v - lr * grad
    theta <- theta + v

mu (momentum) menahan arah pembaruan sebelumnya sehingga konvergensi lebih
mulus dan cepat. Nilai lazim mu = 0.9, lr = 0.01 (sesuai contoh SGDM slide ITB).
"""
from __future__ import annotations

import numpy as np


class SGDMomentum:
    def __init__(self, layers, lr=0.01, momentum=0.9):
        self.layers = layers
        self.lr = lr
        self.momentum = momentum
        # Velocity disimpan per (indeks lapisan, nama parameter).
        self.velocities: dict[tuple[int, str], np.ndarray] = {}
        for li, layer in enumerate(layers):
            for name, p in layer.params.items():
                self.velocities[(li, name)] = np.zeros_like(p)

    def step(self):
        """Lakukan satu langkah pembaruan untuk semua parameter."""
        for li, layer in enumerate(self.layers):
            for name in layer.params:
                grad = layer.grads[name]
                v = self.velocities[(li, name)]
                v *= self.momentum
                v -= self.lr * grad
                # Pembaruan in-place agar referensi array di lapisan tetap.
                layer.params[name] += v

    def zero_grad(self):
        for layer in self.layers:
            for name in layer.grads:
                layer.grads[name][...] = 0.0
