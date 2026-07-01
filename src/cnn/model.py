"""Model LeNet-5 (disesuaikan untuk citra 32x32x1 dan 2 kelas).

Aliran dimensi tiap lapisan (rumus ukuran: (W - N + 2P)/S + 1):

    Input            1 x 32 x 32
    Conv1 6@5x5  ->  6 x 28 x 28   (32 - 5 + 0)/1 + 1 = 28
    MaxPool 2x2  ->  6 x 14 x 14   (28 - 2)/2 + 1     = 14
    Conv2 16@5x5 -> 16 x 10 x 10   (14 - 5)/1 + 1     = 10
    MaxPool 2x2  -> 16 x  5 x  5   (10 - 2)/2 + 1     = 5
    Flatten      -> 400
    FC-120       -> 120
    FC-84        ->  84
    FC-2         ->   2  (softmax: normal / pothole)

LeNet asli memakai average pooling + sigmoid. Di sini dipakai varian modern
ReLU + max pooling karena lebih stabil dilatih, namun tetap "minim magic":
seluruh forward & backward ditulis manual.
"""
from __future__ import annotations

import numpy as np

from .layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU
from .losses import softmax


class LeNet5:
    def __init__(self, num_classes=2, conv1=6, conv2=16, fc1=120, fc2=84,
                 kernel=5, pool=2, seed=42):
        rng = np.random.default_rng(seed)

        self.conv1 = Conv2D(1, conv1, kernel, stride=1, pad=0, rng=rng)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool)
        self.conv2 = Conv2D(conv1, conv2, kernel, stride=1, pad=0, rng=rng)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool)
        self.flatten = Flatten()

        # Ukuran flatten dihitung dari aliran dimensi 32->28->14->10->5.
        flat_dim = conv2 * 5 * 5
        self.fc1 = Dense(flat_dim, fc1, rng=rng, init="he")
        self.relu3 = ReLU()
        self.fc2 = Dense(fc1, fc2, rng=rng, init="he")
        self.relu4 = ReLU()
        self.fc3 = Dense(fc2, num_classes, rng=rng, init="xavier")

        # Urutan lapisan untuk forward/backward dan iterasi parameter.
        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten,
            self.fc1, self.relu3,
            self.fc2, self.relu4,
            self.fc3,
        ]

    def forward(self, x):
        """Jalankan seluruh lapisan; kembalikan skor mentah (logit) (N, K)."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dscores):
        """Propagasi mundur gradien dari skor ke seluruh lapisan."""
        grad = dscores
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict_proba(self, x, batch_size=64):
        """Peluang softmax (N, K), dihitung per-batch agar hemat memori."""
        outs = []
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size]
            outs.append(softmax(self.forward(xb)))
        return np.concatenate(outs, axis=0)

    def predict(self, x, batch_size=64):
        """Prediksi kelas (argmax peluang)."""
        return np.argmax(self.predict_proba(x, batch_size), axis=1)

    # ----- penyimpanan bobot -----
    def save(self, path):
        params = {}
        for li, layer in enumerate(self.layers):
            for name, p in layer.params.items():
                params[f"layer{li}_{name}"] = p
        np.savez(path, **params)

    def load(self, path):
        data = np.load(path)
        for li, layer in enumerate(self.layers):
            for name in layer.params:
                key = f"layer{li}_{name}"
                layer.params[name] = data[key]
        return self
