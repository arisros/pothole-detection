"""Numerical gradient checking — bukti bahwa backward (turunan analitik) benar.

Ide: untuk fungsi skalar L(theta), gradien numerik tiap elemen dihampiri dengan
beda hingga terpusat (central difference):

    dL/dtheta ~= ( L(theta + eps) - L(theta - eps) ) / (2 * eps)

Lalu dibandingkan dengan gradien analitik dari backward(). Galat relatif:

    err = |num - ana| / max(eps_kecil, |num| + |ana|)

Jika err < 1e-5 untuk semua elemen, implementasi backward dianggap benar.
Karena seluruh perhitungan memakai float64, galat tipikal jauh lebih kecil (~1e-9).

Jalankan: python -m src.cnn.gradcheck
"""
from __future__ import annotations

import numpy as np

from .layers import Conv2D, Dense, Dropout, MaxPool2D, ReLU
from .losses import SoftmaxCrossEntropy

EPS = 1e-5
THRESHOLD = 1e-5


def _rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def _numerical_grad(f, x, dout):
    """Gradien numerik dari L(x) = sum(f(x) * dout) terhadap x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + EPS
        plus = np.sum(f(x) * dout)
        x[idx] = old - EPS
        minus = np.sum(f(x) * dout)
        x[idx] = old
        grad[idx] = (plus - minus) / (2 * EPS)
        it.iternext()
    return grad


def check_layer(name, layer, x, has_params=True):
    """Periksa gradien input dan parameter sebuah lapisan."""
    out = layer.forward(x)
    dout = np.random.default_rng(0).normal(size=out.shape)
    dx_analytic = layer.backward(dout)

    # Gradien input numerik.
    dx_numeric = _numerical_grad(lambda z: layer.forward(z), x.copy(), dout)
    err_x = _rel_error(dx_numeric, dx_analytic)
    results = [(f"{name}.dx", err_x)]

    # Gradien parameter numerik.
    if has_params:
        for pname, p in layer.params.items():
            def f_param(_):
                return layer.forward(x)
            # Perturbasi langsung pada array parameter.
            g_num = np.zeros_like(p)
            it = np.nditer(p, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                idx = it.multi_index
                old = p[idx]
                p[idx] = old + EPS
                plus = np.sum(layer.forward(x) * dout)
                p[idx] = old - EPS
                minus = np.sum(layer.forward(x) * dout)
                p[idx] = old
                g_num[idx] = (plus - minus) / (2 * EPS)
                it.iternext()
            # backward harus dijalankan ulang agar grads konsisten dengan dout.
            layer.forward(x)
            layer.backward(dout)
            err_p = _rel_error(g_num, layer.grads[pname])
            results.append((f"{name}.d{pname}", err_p))
    return results


def check_softmax_ce():
    """Periksa gradien gabungan softmax + cross-entropy (p - y)/N."""
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(8, 3))
    y = rng.integers(0, 3, size=8)
    loss_fn = SoftmaxCrossEntropy()

    loss_fn.forward(scores, y)
    grad_analytic = loss_fn.backward()

    grad_numeric = np.zeros_like(scores)
    it = np.nditer(scores, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = scores[idx]
        scores[idx] = old + EPS
        plus = loss_fn.forward(scores, y)
        scores[idx] = old - EPS
        minus = loss_fn.forward(scores, y)
        scores[idx] = old
        grad_numeric[idx] = (plus - minus) / (2 * EPS)
        it.iternext()
    return [("SoftmaxCE.dscores", _rel_error(grad_numeric, grad_analytic))]


def main():
    rng = np.random.default_rng(7)
    all_results = []

    # Conv2D
    conv = Conv2D(2, 3, 3, stride=1, pad=1, rng=rng)
    x_conv = rng.normal(size=(2, 2, 6, 6))
    all_results += check_layer("Conv2D", conv, x_conv, has_params=True)

    # MaxPool2D
    pool = MaxPool2D(2)
    x_pool = rng.normal(size=(2, 3, 4, 4))
    all_results += check_layer("MaxPool2D", pool, x_pool, has_params=False)

    # ReLU
    relu = ReLU()
    x_relu = rng.normal(size=(4, 5))
    all_results += check_layer("ReLU", relu, x_relu, has_params=False)

    # Dense
    dense = Dense(5, 4, rng=rng)
    x_dense = rng.normal(size=(6, 5))
    all_results += check_layer("Dense", dense, x_dense, has_params=True)

    # Dropout — mask dibekukan agar deterministik. Dengan mask tetap, dropout
    # adalah penskalaan elemen linear, sehingga gradien numerik & analitik cocok.
    drop = Dropout(0.5, seed=3)
    x_drop = rng.normal(size=(4, 5))
    drop.forward(x_drop)                       # tetapkan _mask sekali
    frozen = drop._mask
    drop.forward = lambda z, m=frozen: z * m   # bekukan forward = z * mask tetap
    all_results += check_layer("Dropout", drop, x_drop, has_params=False)

    # Softmax + Cross-Entropy
    all_results += check_softmax_ce()

    print("=" * 52)
    print(f"{'Komponen':<22}{'galat relatif':>18}{'status':>10}")
    print("-" * 52)
    ok = True
    for name, err in all_results:
        status = "LULUS" if err < THRESHOLD else "GAGAL"
        if err >= THRESHOLD:
            ok = False
        print(f"{name:<22}{err:>18.2e}{status:>10}")
    print("=" * 52)
    print("SEMUA LULUS — backprop terbukti benar." if ok
          else "ADA YANG GAGAL — periksa implementasi backward.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
