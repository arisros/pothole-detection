"""Optimizer: Stochastic Gradient Descent dengan momentum (SGDM).

Aturan pembaruan untuk tiap parameter theta:

    g <- grad + weight_decay * theta   (hanya untuk bobot W; L2 regularisasi)
    v <- mu * v - lr * g
    theta <- theta + v

mu (momentum) menahan arah pembaruan sebelumnya sehingga konvergensi lebih
mulus dan cepat. Nilai lazim mu = 0.9, lr = 0.01 (sesuai contoh SGDM slide ITB).

weight_decay menambahkan penalti L2 (lambda/2 * ||W||^2) yang menyusutkan bobot
tiap langkah, menekan overfitting. Bias sengaja tidak diberi decay.
"""
from __future__ import annotations

import numpy as np


class SGDMomentum:
    def __init__(self, layers, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
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
                # L2 weight decay: hanya pada matriks/kernel bobot ("W"),
                # tidak pada bias ("b").
                if self.weight_decay and name == "W":
                    grad = grad + self.weight_decay * layer.params[name]
                v = self.velocities[(li, name)]
                v *= self.momentum
                v -= self.lr * grad
                # Pembaruan in-place agar referensi array di lapisan tetap.
                layer.params[name] += v

    def zero_grad(self):
        for layer in self.layers:
            for name in layer.grads:
                layer.grads[name][...] = 0.0


class Adam:
    """Adam ,  adaptive moment estimation (ditulis manual dengan NumPy).

    Menyimpan rata-rata bergerak momen pertama (m) dan kedua (v) tiap parameter,
    lalu memperbarui dengan koreksi bias:

        m <- b1*m + (1-b1)*g
        v <- b2*v + (1-b2)*g^2
        m_hat = m/(1-b1^t),  v_hat = v/(1-b2^t)
        theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)

    Weight decay L2 (opsional) ditambahkan ke gradien hanya untuk bobot "W".
    lr lazim untuk Adam jauh lebih kecil dari SGD (mis. 1e-3).
    """

    def __init__(self, layers, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m: dict[tuple[int, str], np.ndarray] = {}
        self.v: dict[tuple[int, str], np.ndarray] = {}
        for li, layer in enumerate(layers):
            for name, p in layer.params.items():
                self.m[(li, name)] = np.zeros_like(p)
                self.v[(li, name)] = np.zeros_like(p)

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for li, layer in enumerate(self.layers):
            for name in layer.params:
                grad = layer.grads[name]
                if self.weight_decay and name == "W":
                    grad = grad + self.weight_decay * layer.params[name]
                m = self.m[(li, name)]
                v = self.v[(li, name)]
                m *= b1
                m += (1.0 - b1) * grad
                v *= b2
                v += (1.0 - b2) * (grad * grad)
                m_hat = m / bc1
                v_hat = v / bc2
                layer.params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for layer in self.layers:
            for name in layer.grads:
                layer.grads[name][...] = 0.0
