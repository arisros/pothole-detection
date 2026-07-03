"""Model LeNet-5 (varian modern) untuk citra RGB 48x48, 2 kelas.

Aliran dimensi tiap lapisan (rumus ukuran: (W - N + 2P)/S + 1), untuk masukan
RGB 48x48 yang dipakai pipeline:

    Input            3 x 48 x 48
    Conv1 6@5x5  ->  6 x 44 x 44   (48 - 5 + 0)/1 + 1 = 44
    MaxPool 2x2  ->  6 x 22 x 22   (44 - 2)/2 + 1     = 22
    Conv2 16@5x5 -> 16 x 18 x 18   (22 - 5)/1 + 1     = 18
    MaxPool 2x2  -> 16 x  9 x  9   (18 - 2)/2 + 1     = 9
    Flatten      -> 1296
    FC-120       -> 120
    FC-84        ->  84
    FC-2         ->   2  (softmax: normal / pothole)

Ukuran & jumlah kanal masukan dapat diatur lewat argumen (in_channels, img_size);
flat_dim dihitung dinamis. LeNet asli memakai average pooling + sigmoid; di sini
dipakai varian modern ReLU + max pooling yang lebih stabil dilatih, namun tetap
"minim magic": seluruh forward & backward ditulis manual.
"""
from __future__ import annotations

import numpy as np

from .layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, ReLU
from .losses import softmax


class LeNet5:
    def __init__(self, num_classes=2, conv1=6, conv2=16, fc1=120, fc2=84,
                 kernel=5, pool=2, seed=42, dropout_p=0.0,
                 in_channels=1, img_size=32):
        rng = np.random.default_rng(seed)

        self.conv1 = Conv2D(in_channels, conv1, kernel, stride=1, pad=0, rng=rng)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool)
        self.conv2 = Conv2D(conv1, conv2, kernel, stride=1, pad=0, rng=rng)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool)
        self.flatten = Flatten()

        # Aliran dimensi spasial: conv (k) mengurangi (k-1), pool membagi 2.
        # Dihitung dinamis agar mendukung IMG_SIZE selain 32 (mis. 48, 64).
        s = img_size
        s = (s - (kernel - 1)) // pool          # setelah conv1 + pool1
        s = (s - (kernel - 1)) // pool          # setelah conv2 + pool2
        self.feat_size = s
        flat_dim = conv2 * s * s
        self.fc1 = Dense(flat_dim, fc1, rng=rng, init="he")
        self.relu3 = ReLU()
        # Dropout pada lapisan terhubung penuh (regularisasi). p=0 -> identitas.
        self.drop1 = Dropout(dropout_p, seed=seed + 1)
        self.fc2 = Dense(fc1, fc2, rng=rng, init="he")
        self.relu4 = ReLU()
        self.drop2 = Dropout(dropout_p, seed=seed + 2)
        self.fc3 = Dense(fc2, num_classes, rng=rng, init="xavier")

        # Urutan lapisan untuk forward/backward dan iterasi parameter.
        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten,
            self.fc1, self.relu3, self.drop1,
            self.fc2, self.relu4, self.drop2,
            self.fc3,
        ]

    def train(self):
        """Aktifkan mode latih (dropout hidup) pada semua lapisan."""
        for layer in self.layers:
            layer.training = True
        return self

    def eval(self):
        """Aktifkan mode evaluasi (dropout mati / identitas) pada semua lapisan."""
        for layer in self.layers:
            layer.training = False
        return self

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
        self.eval()  # inferensi deterministik: dropout mati
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
