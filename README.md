# Deteksi Jalan Berlubang dengan CNN (Implementasi dari Nol)

Klasifikasi citra **jalan berlubang (pothole)** vs **jalan normal** menggunakan
**Convolutional Neural Network yang inti algoritmanya ditulis manual dengan NumPy** —
konvolusi, pooling, ReLU, softmax, cross-entropy, dan *backpropagation* diturunkan
secara matematis lalu dikodekan sendiri **tanpa** framework deep learning (PyTorch/TF/Keras).
Arsitektur acuan: **LeNet-5**.

> Library pihak ketiga hanya dipakai sebagai *helper* pinggiran — Pillow (I/O citra),
> scikit-learn (split data), matplotlib (plot). Inti pembelajaran 100% NumPy manual.

## Struktur

```
config.py            Hyperparameter & path terpusat
src/
  download_data.py   Unduh dataset publik pothole/normal
  label_dataset.py   Susun kelas -> labels.csv
  preprocess.py      Grayscale, resize 32x32, normalisasi, augmentasi, split
  cnn/               INTI MANUAL (tanpa autograd)
    tensor_utils.py  im2col / col2im
    init.py          Inisialisasi He / Xavier
    layers.py        Conv2D, MaxPool2D, ReLU, Flatten, Dense (forward+backward)
    losses.py        Softmax + Cross-Entropy
    optim.py         SGD + momentum
    metrics.py       Akurasi, presisi, recall, F1, confusion matrix
    model.py         LeNet-5
    trainer.py       Training loop mini-batch
    gradcheck.py     Numerical gradient checking (bukti backprop benar)
  train.py           Latih model
  evaluate.py        Evaluasi + figures
  visualize.py       Feature map antar lapisan
docs/                Dokumentasi teknis + diagram Mermaid (aliran hidden layer)
makalah/             Makalah ilmiah Bahasa Indonesia (>=20 halaman)
experiments/         weights.npz, history.csv, figures/
```

## Cara menjalankan

```bash
pip install -r requirements.txt

make data        # unduh dataset
make label       # susun kelas
make preprocess  # buat dataset.npz
make gradcheck   # verifikasi kebenaran backprop (galat < 1e-5)
make train       # latih LeNet-5 manual
make eval        # metrik test + confusion matrix
make visualize   # peta fitur antar lapisan
# atau: make all
```

## Arsitektur (aliran dimensi tiap lapisan)

```
Input 1x32x32
  -> Conv1 6@5x5 + ReLU   -> 6x28x28
  -> MaxPool 2x2          -> 6x14x14
  -> Conv2 16@5x5 + ReLU  -> 16x10x10
  -> MaxPool 2x2          -> 16x5x5
  -> Flatten              -> 400
  -> FC-120 + ReLU
  -> FC-84  + ReLU
  -> FC-2   + Softmax     -> {normal, pothole}
```

Ukuran tiap *feature map* dihitung dengan rumus `output = (W - N + 2P) / S + 1`.

## Referensi notasi

R. Munir, *"21 — Convolutional Neural Network"*, IF4073 Pemrosesan Citra Digital,
Teknik Informatika ITB, 2024.
