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

from .layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, ReLU
from .losses import softmax


class LeNet5:
    """CNN dengan tumpukan konvolusi yang dapat dikonfigurasi.

    `conv_specs` = daftar (out_channels, kernel) per blok konv; tiap blok =
    Conv2D -> ReLU -> MaxPool. Default [(6,5),(16,5)] = LeNet-5 klasik (dua blok).
    Tumpukan lebih dalam (mis. tiga blok) berguna untuk masukan beresolusi lebih
    besar (64px) agar bidang reseptif memadai dan dimensi flatten tetap ramping.
    """

    def __init__(self, num_classes=2, conv1=6, conv2=16, fc1=120, fc2=84,
                 kernel=5, pool=2, seed=42, dropout_p=0.0,
                 in_channels=1, img_size=32, conv_specs=None):
        rng = np.random.default_rng(seed)
        if conv_specs is None:
            conv_specs = [(conv1, kernel), (conv2, kernel)]

        # ----- Tumpukan konvolusi (dinamis) -----
        self.conv_stages = []   # [(conv, relu, pool), ...] untuk introspeksi
        layers = []
        s = img_size
        prev = in_channels
        for i, (ch, k) in enumerate(conv_specs, 1):
            conv = Conv2D(prev, ch, k, stride=1, pad=0, rng=rng)
            relu = ReLU()
            mp = MaxPool2D(pool)
            self.conv_stages.append((conv, relu, mp))
            layers += [conv, relu, mp]
            # Alias bernomor untuk kompatibilitas (mis. visualize.py).
            setattr(self, f"conv{i}", conv)
            setattr(self, f"relu{i}", relu)
            setattr(self, f"pool{i}", mp)
            # Aliran dimensi spasial: conv (k) mengurangi (k-1), pool membagi.
            s = (s - (k - 1)) // pool
            prev = ch
        if s <= 0:
            raise ValueError(
                f"img_size={img_size} terlalu kecil untuk {len(conv_specs)} "
                f"blok konv (feat_size={s}). Kurangi blok atau perbesar input.")
        self.feat_size = s

        # ----- Kepala terhubung penuh -----
        flat_dim = prev * s * s
        self.flatten = Flatten()
        self.fc1 = Dense(flat_dim, fc1, rng=rng, init="he")
        self.relu_fc1 = ReLU()
        # Dropout pada lapisan FC (regularisasi). p=0 -> identitas.
        self.drop1 = Dropout(dropout_p, seed=seed + 1)
        self.fc2 = Dense(fc1, fc2, rng=rng, init="he")
        self.relu_fc2 = ReLU()
        self.drop2 = Dropout(dropout_p, seed=seed + 2)
        self.fc3 = Dense(fc2, num_classes, rng=rng, init="xavier")

        layers += [
            self.flatten,
            self.fc1, self.relu_fc1, self.drop1,
            self.fc2, self.relu_fc2, self.drop2,
            self.fc3,
        ]
        # Urutan lapisan untuk forward/backward dan iterasi parameter.
        self.layers = layers

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
