"""Lapisan-lapisan CNN — forward & backward diturunkan dan ditulis manual.

Konvensi tensor citra: (N, C, H, W)
    N = ukuran batch, C = jumlah kanal, H = tinggi, W = lebar.

Setiap lapisan menyediakan:
    forward(x)      -> keluaran lapisan; menyimpan cache untuk backward
    backward(dout)  -> gradien terhadap masukan (dx); mengisi self.grads

Lapisan berparameter menyimpan:
    self.params = {"W": ..., "b": ...}
    self.grads  = {"W": ..., "b": ...}
Lapisan tanpa parameter memiliki dict kosong.
"""
from __future__ import annotations

import numpy as np

from . import init as winit
from .tensor_utils import col2im, conv_output_size, im2col


class Layer:
    """Antarmuka dasar sebuah lapisan."""

    def __init__(self):
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x):  # pragma: no cover - antarmuka
        raise NotImplementedError

    def backward(self, dout):  # pragma: no cover - antarmuka
        raise NotImplementedError


class Conv2D(Layer):
    """Lapisan konvolusi 2D.

    Forward (untuk tiap filter f dan posisi (i, j)):

        O[f,i,j] = b[f] + sum_c sum_m sum_n X[c, i*S+m, j*S+n] * W[f,c,m,n]

    Dihitung efisien lewat im2col: O = W_col @ X_col + b.

    Backward:
        dW   = dout_col @ X_col^T
        db   = sum dout pada sumbu (N, out_h, out_w)
        dX   = col2im(W_col^T @ dout_col)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, pad=0, rng=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh = self.kw = kernel_size
        self.stride = stride
        self.pad = pad

        rng = rng if rng is not None else np.random.default_rng()
        fan_in = in_channels * self.kh * self.kw
        # W: (F, C, KH, KW) — He init karena diikuti ReLU.
        self.params["W"] = winit.he_normal(
            (out_channels, in_channels, self.kh, self.kw), fan_in, rng
        )
        self.params["b"] = np.zeros(out_channels, dtype=np.float64)
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])

        self._cache = None

    def forward(self, x):
        n, c, h, w = x.shape
        out_h = conv_output_size(h, self.kh, self.stride, self.pad)
        out_w = conv_output_size(w, self.kw, self.stride, self.pad)

        x_col = im2col(x, self.kh, self.kw, self.stride, self.pad)  # (C*KH*KW, N*oh*ow)
        w_col = self.params["W"].reshape(self.out_channels, -1)     # (F, C*KH*KW)

        out = w_col @ x_col + self.params["b"][:, None]             # (F, N*oh*ow)
        out = out.reshape(self.out_channels, out_h, out_w, n)
        out = out.transpose(3, 0, 1, 2)                            # (N, F, oh, ow)

        self._cache = (x.shape, x_col, w_col, out_h, out_w)
        return out

    def backward(self, dout):
        x_shape, x_col, w_col, out_h, out_w = self._cache
        n = x_shape[0]

        # Susun dout agar selaras dengan tata letak forward: (F, N*oh*ow).
        dout_r = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

        # Gradien bias: jumlah seluruh posisi & batch.
        self.grads["b"] = dout_r.sum(axis=1)
        # Gradien bobot.
        dw_col = dout_r @ x_col.T                                   # (F, C*KH*KW)
        self.grads["W"] = dw_col.reshape(self.params["W"].shape)
        # Gradien terhadap masukan.
        dx_col = w_col.T @ dout_r                                   # (C*KH*KW, N*oh*ow)
        dx = col2im(dx_col, x_shape, self.kh, self.kw, self.stride, self.pad)
        return dx


class MaxPool2D(Layer):
    """Max pooling 2D (tanpa parameter).

    Forward: ambil nilai maksimum pada tiap jendela pool_size x pool_size.
    Backward: gradien hanya diteruskan ke posisi pemenang (argmax); posisi
    lain mendapat gradien 0.
    """

    def __init__(self, pool_size=2, stride=None):
        super().__init__()
        self.pool = pool_size
        self.stride = stride if stride is not None else pool_size
        self._cache = None

    def forward(self, x):
        n, c, h, w = x.shape
        out_h = conv_output_size(h, self.pool, self.stride, 0)
        out_w = conv_output_size(w, self.pool, self.stride, 0)

        # Perlakukan tiap kanal sebagai citra terpisah agar pooling per-kanal.
        x_reshaped = x.reshape(n * c, 1, h, w)
        x_col = im2col(x_reshaped, self.pool, self.pool, self.stride, 0)
        # x_col: (pool*pool, (N*C)*out_h*out_w)

        max_idx = np.argmax(x_col, axis=0)
        out = x_col[max_idx, np.arange(x_col.shape[1])]
        out = out.reshape(out_h, out_w, n, c).transpose(2, 3, 0, 1)

        self._cache = (x.shape, x_col.shape, max_idx, x_reshaped.shape)
        return out

    def backward(self, dout):
        x_shape, x_col_shape, max_idx, x_reshaped_shape = self._cache
        n, c, h, w = x_shape

        # Susun dout mengikuti urutan kolom forward.
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()
        dx_col = np.zeros(x_col_shape, dtype=dout.dtype)
        dx_col[max_idx, np.arange(dx_col.shape[1])] = dout_flat

        dx = col2im(dx_col, x_reshaped_shape, self.pool, self.pool, self.stride, 0)
        dx = dx.reshape(x_shape)
        return dx


class ReLU(Layer):
    """Rectified Linear Unit: f(u) = max(0, u), f'(u) = 1[u > 0]."""

    def __init__(self):
        super().__init__()
        self._mask = None

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask


class Flatten(Layer):
    """Ratakan (N, C, H, W) -> (N, C*H*W) sebelum lapisan terhubung penuh."""

    def __init__(self):
        super().__init__()
        self._shape = None

    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._shape)


class Dense(Layer):
    """Lapisan terhubung penuh (fully-connected / MLP).

    Forward:  y = x W + b
    Backward: dW = x^T dy,  db = sum dy,  dx = dy W^T
    """

    def __init__(self, in_features, out_features, rng=None, init="he"):
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        if init == "he":
            self.params["W"] = winit.he_normal(
                (in_features, out_features), in_features, rng
            )
        else:
            self.params["W"] = winit.xavier_normal(
                (in_features, out_features), in_features, rng
            )
        self.params["b"] = np.zeros(out_features, dtype=np.float64)
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads["W"] = self._x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T
