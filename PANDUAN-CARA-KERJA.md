# Membangun CNN Deteksi Jalan Berlubang dari Nol — Panduan Step-by-Step

> Dokumen ini menjelaskan **arsitektur** dan **implementasi kode inti** sistem
> klasifikasi citra jalan berlubang (pothole) vs normal menggunakan **CNN LeNet-5
> yang seluruh matematikanya ditulis manual dengan NumPy** — tanpa PyTorch/TF/Keras,
> tanpa autograd. Ditujukan untuk software engineer yang ingin memahami/merekonstruksi.

---

## 0. Prinsip & Tujuan

- **Tugas:** klasifikasi biner citra jalan → `normal` (0) / `pothole` (1).
- **"Tanpa magic":** konvolusi, pooling, ReLU, softmax, cross-entropy, dan
  **backpropagation** diturunkan matematis lalu dikodekan sendiri. Kebenarannya
  dibuktikan dengan *numerical gradient checking* (galat 1e-9–1e-11).
- **Arsitektur acuan:** LeNet-5 (paling sederhana, ideal dipelajari).
- **Helper (bukan inti):** Pillow (baca/resize citra), scikit-learn (split),
  matplotlib (plot). Inti pembelajaran = 100% NumPy.
- **Skala:** citra 48×48 RGB + dataset kecil, agar pelatihan murni-CPU tetap selesai.

Konvensi tensor citra di seluruh kode: **`(N, C, H, W)`** = (batch, channel, height, width).

---

## 1. Arsitektur Sistem (peta besar)

```
Dataset publik ─► Unduh ─► Pelabelan ─► Preprocessing ─► dataset.npz
                                                             │
                                                             ▼
                                        ┌────────── CORE: CNN manual (NumPy) ──────────┐
                                        │ layers (Conv/Pool/ReLU/Dense) → losses →      │
                                        │ optim (Adam / SGD-momentum) → model (LeNet5) →│
                                        │ trainer (loop) → gradcheck (bukti benar)      │
                                        └───────────────────────────────────────────────┘
                                                             │
                              weights.npz ◄──────────────────┤
                                    │                        ▼
                              evaluate.py            visualize.py (feature map)
```

Struktur folder:

```
config.py            # hyperparameter & path terpusat (seed=42, reproducible)
src/
  download_data.py   # unduh dataset pothole/normal dari repo GitHub publik
  label_dataset.py   # susun folder kelas -> labels.csv
  preprocess.py      # RGB, resize 48x48, normalisasi, augmentasi daring, split -> dataset.npz
  cnn/               # ======== INTI MANUAL ========
    tensor_utils.py  # im2col / col2im
    init.py          # inisialisasi He / Xavier
    layers.py        # Conv2D, MaxPool2D, ReLU, Flatten, Dense (forward+backward)
    losses.py        # softmax + cross-entropy
    optim.py         # SGD-momentum & Adam (default) + cosine LR, weight decay
    metrics.py       # akurasi, presisi, recall, F1, confusion matrix
    model.py         # LeNet5 (rakit lapisan)
    trainer.py       # training loop mini-batch + best-val checkpoint
    gradcheck.py     # numerical gradient checking
  train.py           # latih -> weights.npz + history.csv
  evaluate.py        # metrik test + figur
  visualize.py       # feature map antar lapisan
```

---

## 2. Pipeline Data (Langkah 1–3)

### Langkah 1 — Dataset
Unduh dataset klasifikasi biner dari repo GitHub publik (folder `Pothole/` & `Plain/`),
petakan `Pothole → pothole (1)`, `Plain → normal (0)`. Hasil: 712 citra (348 normal,
364 pothole), disimpan ke `data/raw/{pothole,normal}/`.

### Langkah 2 — Pelabelan
Baca folder kelas → tulis `labels.csv` (`path,label`) + verifikasi tiap gambar bisa dibuka.

### Langkah 3 — Preprocessing
Lima langkah (tiap langkah eksplisit):

1. **Kanal warna:** citra dipertahankan **RGB (3 kanal)** — warna menambah sinyal.
   (Opsi grayscale via luminance BT.601 `Y = 0.299R + 0.587G + 0.114B` tetap tersedia.)
2. **Resize** → 48×48 (input LeNet-5 adaptasi).
3. **Normalisasi:** skala `[0,1]` lalu standardisasi `(x - μ)/σ` (μ,σ skalar dari data latih).
4. **Augmentasi daring** (train saja, acak per-batch tiap epoch): flip horizontal, geser ±3px,
   kontras/kecerahan, noise Gaussian → model tak pernah lihat citra sama persis dua kali.
5. **Split stratified** 70/15/15 (pakai `sklearn.train_test_split`).

Output: `data/processed/dataset.npz` berisi `x_train/y_train/x_val/y_val/x_test/y_test`.

---

## 3. CORE — CNN Manual (Langkah 4)  ◄── JANTUNG "TANPA MAGIC"

Semua lapisan punya antarmuka seragam:
```python
out = layer.forward(x)       # simpan cache utk backward
dx  = layer.backward(dout)   # kembalikan grad thd input; isi self.grads utk parameter
```

### 3.1 im2col / col2im — kunci efisiensi konvolusi

Konvolusi naif = 4 loop bersarang (lambat di Python). Trik **im2col** mengubah tiap
jendela konvolusi menjadi satu kolom matriks → konvolusi jadi **satu perkalian matriks**.

Rumus ukuran output (dipakai di mana-mana):
```
output = (W - N + 2P) / S + 1     # W=ukuran input, N=kernel, P=padding, S=stride
```

```python
import numpy as np

def conv_output_size(in_size, kernel, stride, pad):
    return (in_size - kernel + 2*pad) // stride + 1

def _get_im2col_indices(x_shape, kh, kw, stride, pad):
    n, c, h, w = x_shape
    out_h = conv_output_size(h, kh, stride, pad)
    out_w = conv_output_size(w, kw, stride, pad)
    i0 = np.repeat(np.arange(kh), kw); i0 = np.tile(i0, c)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(kw), kh * c)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    d = np.repeat(np.arange(c), kh * kw).reshape(-1, 1)
    return i, j, d, out_h, out_w

def im2col(x, kh, kw, stride, pad):
    x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode="constant")
    i, j, d, _, _ = _get_im2col_indices(x.shape, kh, kw, stride, pad)
    cols = x_padded[:, d, i, j]                    # (N, C*KH*KW, out_h*out_w)
    c = x.shape[1]
    return cols.transpose(1, 2, 0).reshape(kh*kw*c, -1)   # (C*KH*KW, N*out_h*out_w)

def col2im(cols, x_shape, kh, kw, stride, pad):
    n, c, h, w = x_shape
    h_pad, w_pad = h + 2*pad, w + 2*pad
    x_padded = np.zeros((n, c, h_pad, w_pad), dtype=cols.dtype)
    i, j, d, _, _ = _get_im2col_indices(x_shape, kh, kw, stride, pad)
    cols_reshaped = cols.reshape(c*kh*kw, -1, n).transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), d, i, j), cols_reshaped)  # jumlahkan kontribusi
    return x_padded if pad == 0 else x_padded[:, :, pad:-pad, pad:-pad]
```

### 3.2 Conv2D — forward & backward

**Forward:** `O[f,i,j] = b[f] + Σ_c Σ_m Σ_n X[c, iS+m, jS+n] · W[f,c,m,n]`
→ via im2col: `O = W_col @ X_col + b`.

**Backward (turunan aturan rantai):**
```
dW = dout_col @ X_col^T        db = Σ dout        dX = col2im(W_col^T @ dout_col)
```

```python
from . import init as winit  # He/Xavier

class Conv2D:
    def __init__(self, in_ch, out_ch, k, stride=1, pad=0, rng=None):
        rng = rng or np.random.default_rng()
        self.out_ch, self.kh, self.kw, self.stride, self.pad = out_ch, k, k, stride, pad
        fan_in = in_ch * k * k
        self.params = {"W": winit.he_normal((out_ch, in_ch, k, k), fan_in, rng),
                       "b": np.zeros(out_ch)}
        self.grads = {"W": np.zeros_like(self.params["W"]),
                      "b": np.zeros_like(self.params["b"])}

    def forward(self, x):
        n, c, h, w = x.shape
        oh = conv_output_size(h, self.kh, self.stride, self.pad)
        ow = conv_output_size(w, self.kw, self.stride, self.pad)
        x_col = im2col(x, self.kh, self.kw, self.stride, self.pad)   # (C*KH*KW, N*oh*ow)
        w_col = self.params["W"].reshape(self.out_ch, -1)           # (F, C*KH*KW)
        out = (w_col @ x_col + self.params["b"][:, None])           # (F, N*oh*ow)
        out = out.reshape(self.out_ch, oh, ow, n).transpose(3, 0, 1, 2)  # (N,F,oh,ow)
        self._cache = (x.shape, x_col, w_col)
        return out

    def backward(self, dout):
        x_shape, x_col, w_col = self._cache
        dout_r = dout.transpose(1, 2, 3, 0).reshape(self.out_ch, -1)   # (F, N*oh*ow)
        self.grads["b"] = dout_r.sum(axis=1)
        self.grads["W"] = (dout_r @ x_col.T).reshape(self.params["W"].shape)
        dx_col = w_col.T @ dout_r
        return col2im(dx_col, x_shape, self.kh, self.kw, self.stride, self.pad)
```

### 3.3 MaxPool2D — forward & backward
Forward: maksimum tiap jendela + simpan posisi argmax. Backward: gradien **hanya**
mengalir ke posisi pemenang.

```python
class MaxPool2D:
    def __init__(self, pool=2, stride=None):
        self.pool, self.stride = pool, stride or pool
        self.params, self.grads = {}, {}

    def forward(self, x):
        n, c, h, w = x.shape
        oh = conv_output_size(h, self.pool, self.stride, 0)
        ow = conv_output_size(w, self.pool, self.stride, 0)
        x_r = x.reshape(n*c, 1, h, w)                     # pooling per-kanal
        x_col = im2col(x_r, self.pool, self.pool, self.stride, 0)  # (pool*pool, N*C*oh*ow)
        idx = np.argmax(x_col, axis=0)
        out = x_col[idx, np.arange(x_col.shape[1])]
        out = out.reshape(oh, ow, n, c).transpose(2, 3, 0, 1)
        self._cache = (x.shape, x_col.shape, idx, x_r.shape)
        return out

    def backward(self, dout):
        x_shape, x_col_shape, idx, x_r_shape = self._cache
        dflat = dout.transpose(2, 3, 0, 1).ravel()
        dcol = np.zeros(x_col_shape, dtype=dout.dtype)
        dcol[idx, np.arange(dcol.shape[1])] = dflat        # rute ke argmax
        dx = col2im(dcol, x_r_shape, self.pool, self.pool, self.stride, 0)
        return dx.reshape(x_shape)
```

### 3.4 ReLU & Flatten
```python
class ReLU:
    def __init__(self): self.params, self.grads = {}, {}
    def forward(self, x):  self._mask = x > 0; return x * self._mask
    def backward(self, d): return d * self._mask            # f'(u) = 1[u>0]

class Flatten:
    def __init__(self): self.params, self.grads = {}, {}
    def forward(self, x): self._shape = x.shape; return x.reshape(x.shape[0], -1)
    def backward(self, d): return d.reshape(self._shape)
```

### 3.5 Dense (fully-connected)
`y = xW + b` → `dW = xᵀδ`, `db = Σδ`, `dx = δWᵀ`.
```python
class Dense:
    def __init__(self, n_in, n_out, rng=None, init="he"):
        rng = rng or np.random.default_rng()
        W = (winit.he_normal if init == "he" else winit.xavier_normal)((n_in, n_out), n_in, rng)
        self.params = {"W": W, "b": np.zeros(n_out)}
        self.grads = {"W": np.zeros_like(W), "b": np.zeros(n_out)}
    def forward(self, x): self._x = x; return x @ self.params["W"] + self.params["b"]
    def backward(self, d):
        self.grads["W"] = self._x.T @ d
        self.grads["b"] = d.sum(axis=0)
        return d @ self.params["W"].T
```

### 3.6 Softmax + Cross-Entropy — kenapa digabung
Softmax: `p_i = e^{z_i} / Σ_j e^{z_j}`. Cross-entropy: `L = -Σ y_i ln p_i`.
Diturunkan **bersama**, gradiennya menyederhana jadi bentuk ringkas & stabil:
```
∂L/∂z = p - y          # inilah titik awal aliran gradien mundur
```
```python
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)   # stabil numerik
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

class SoftmaxCrossEntropy:
    def forward(self, scores, y):
        self.p = softmax(scores); self.y = y; self.n = scores.shape[0]
        return float(np.mean(-np.log(self.p[np.arange(self.n), y] + 1e-12)))
    def backward(self):
        g = self.p.copy(); g[np.arange(self.n), self.y] -= 1.0
        return g / self.n
```

### 3.7 Inisialisasi & Optimizer
```python
# init.py — jaga varians aktivasi stabil
def he_normal(shape, fan_in, rng):     return rng.normal(0, np.sqrt(2/fan_in), shape)   # utk ReLU
def xavier_normal(shape, fan_in, rng): return rng.normal(0, np.sqrt(1/fan_in), shape)   # utk output

# optim.py — SGD + momentum:  v = μv - η∇ ;  θ = θ + v
class SGDMomentum:
    def __init__(self, layers, lr=0.01, momentum=0.9):
        self.layers, self.lr, self.mu = layers, lr, momentum
        self.v = {(i, k): np.zeros_like(p)
                  for i, l in enumerate(layers) for k, p in l.params.items()}
    def step(self):
        for i, l in enumerate(self.layers):
            for k in l.params:
                v = self.v[(i, k)]
                v *= self.mu; v -= self.lr * l.grads[k]
                l.params[k] += v            # update in-place
```

### 3.8 Merakit LeNet-5
Aliran dimensi (pakai `(W-N+2P)/S+1`): `48 → 44 → 22 → 18 → 9`.
```python
class LeNet5:
    def __init__(self, num_classes=2, seed=42):
        rng = np.random.default_rng(seed)
        self.layers = [
            Conv2D(3, 6, 5, rng=rng),  ReLU(), MaxPool2D(2),   # 3x48x48 -> 6x44x44 -> 6x22x22
            Conv2D(6, 16, 5, rng=rng), ReLU(), MaxPool2D(2),   # -> 16x18x18 -> 16x9x9
            Flatten(),                                          # -> 1296
            Dense(16*9*9, 120, rng=rng), ReLU(), Dropout(0.3),
            Dense(120, 84, rng=rng),     ReLU(), Dropout(0.3),
            Dense(84, num_classes, rng=rng, init="xavier"),     # -> logit (N,2)
        ]
    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x
    def backward(self, dscores):
        g = dscores
        for l in reversed(self.layers): g = l.backward(g)
```

### 3.9 Training loop (satu iterasi = forward → loss → backward → step)
```python
loss_fn = SoftmaxCrossEntropy()
opt = Adam([l for l in model.layers if l.params], lr=1e-3, weight_decay=1e-4)  # default; SGDMomentum juga tersedia

for epoch in range(epochs):
    opt.lr = cosine_decay(1e-3, epoch, epochs)   # jadwal cosine
    for xb, yb in minibatches(x_train, y_train, batch=32, shuffle=True):
        xb = augment(xb)                    # augmentasi daring acak per-batch
        scores = model.forward(xb)          # 1. maju (dropout aktif)
        loss   = loss_fn.forward(scores, yb)# 2. loss
        model.backward(loss_fn.backward())  # 3. mundur (∂L/∂z = p - y) -> isi grads
        opt.step()                          # 4. update bobot (+ L2 weight decay)
    # + evaluasi val, simpan snapshot bobot val terbaik (early stopping ringan)
    # Ulangi utuh untuk seed 42/7/123 -> ensembel; saat uji rata-ratakan softmax + TTA.
```

### 3.10 Gradient checking — BUKTI backprop benar (wajib!)
Bandingkan gradien analitik vs numerik (beda hingga terpusat):
```
∂L/∂θ ≈ ( L(θ+ε) - L(θ-ε) ) / (2ε)
```
```python
def numerical_grad(f, x, dout, eps=1e-5):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index; old = x[idx]
        x[idx] = old + eps; plus  = np.sum(f(x) * dout)
        x[idx] = old - eps; minus = np.sum(f(x) * dout)
        x[idx] = old; g[idx] = (plus - minus) / (2*eps)
        it.iternext()
    return g
# galat relatif |num - ana| / (|num|+|ana|) harus < 1e-5.
# Hasil nyata: 1e-9 .. 1e-11 untuk Conv/Pool/Dense/Softmax -> backprop terbukti benar.
```

---

## 4. Latih, Uji, Visualisasi (Langkah 5–7)

- **Sanity check:** overfit 32 sampel → akurasi ~100% (memastikan model & backprop belajar).
- **Latih penuh:** 40 epoch, Adam lr=1e-3 + cosine, batch 32, weight decay 1e-4, dropout 0.3;
  diulang untuk 3 seed (42/7/123) → ensembel. Simpan `weights.npz` + `history.csv`.
- **Uji:** akurasi/presisi/recall/F1 + confusion matrix di test set.
- **Visualisasi:** render feature map tiap lapisan (Conv1→Pool1→Conv2→Pool2) untuk melihat
  "pergerakan hidden layer".

**Hasil nyata (test set, ensembel + TTA):** Akurasi **94,4%**, Presisi **91,5%**, Recall **98,2%**, F1 **94,7%**.
Dicapai dengan RGB 48×48 + regularisasi (weight decay/dropout) + augmentasi
daring + ensembel 3 seed; overfitting terkendali (val-acc terbaik ~89,7% @ epoch 25).

---

## 5. Arsitektur Deployment (opsional — cara publish di homelab k8s)

Situs live di `https://potholes.arisjirat.com`, di-deploy **k8s-native + GitOps (ArgoCD)**:

```
Browser ─► Cloudflare ─► VPS Caddy (TLS) ─► [WireGuard] ─► caddy-homelab (:30100)
        ─► k3s-node:30100 (NodePort) ─► Pod (container 1 proses: situs statis + /api/predict)
```

- **Container:** `python:3.11-slim` + numpy/pillow menjalankan `web/server.py` (stdlib HTTP;
  serve statis + endpoint `/api/predict` yang memuat `weights.npz` & menjalankan LeNet5).
- **Image:** `docker build` → `k3d image import potholes:v1 -c homelab` (side-load, tanpa registry).
- **k8s:** Deployment + Service NodePort (`apps/potholes/base` + `overlays/production`, Kustomize).
  Penting: `enableServiceLinks: false` + env eksplisit (karena service bernama `potholes` bikin
  k8s inject `POTHOLES_PORT=tcp://...` yang bentrok dgn env aplikasi).
- **GitOps:** ArgoCD Application (`argocd/applications/apps/potholes.yaml`) auto-sync dari `master`.
- **Routing:** caddy-homelab `:30100 { reverse_proxy k3s-node:30100 }` + VPS Caddy
  `potholes.arisjirat.com { reverse_proxy 10.10.0.1:30100 }`.

Update rilis: `bash web/build-image.sh v2` → bump `newTag: v2` di overlay → commit+push master → ArgoCD sync.

---

## 6. Reproduksi Ringkas

```bash
pip install numpy pillow scikit-learn matplotlib
python src/download_data.py     # dataset -> data/raw/
python src/label_dataset.py     # -> labels.csv
python src/preprocess.py        # -> dataset.npz
python -m src.cnn.gradcheck     # BUKTI backprop benar (galat < 1e-5)
python src/train.py             # -> weights.npz + history.csv
python src/evaluate.py          # metrik test + confusion matrix
python src/visualize.py         # feature map antar lapisan
```

**Intinya:** semua pembelajaran (conv, pool, relu, softmax, cross-entropy, backprop,
SGD) ada di `src/cnn/` dan ditulis tangan dengan NumPy; kebenarannya dibuktikan dengan
gradient checking. LeNet-5 hanya kerangka yang merangkai lapisan-lapisan itu.
```
