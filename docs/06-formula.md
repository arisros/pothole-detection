# 06, Kumpulan Formula (Forward & Backward)

Rangkuman seluruh formula yang diimplementasikan manual. Notasi mengikuti slide
kuliah *Convolutional Neural Network* (Rinaldi Munir, ITB, 2024).

## Notasi

- $X$ : tensor masukan $(N, C, H, W)$
- $W, b$ : bobot dan bias lapisan
- $\delta = \partial L / \partial(\text{keluaran})$ : gradien dari lapisan berikutnya
- $S$ : stride, $P$ : padding, $N$ (pada rumus ukuran) : ukuran kernel

## 1. Ukuran feature map

$$\text{output} = \frac{W - N + 2P}{S} + 1$$

## 2. Konvolusi

**Forward:**
$$O[f,i,j] = b_f + \sum_{c}\sum_{m}\sum_{n} X[c,\,iS+m,\,jS+n]\cdot W[f,c,m,n]$$

**Backward:**
$$\frac{\partial L}{\partial W[f,c,m,n]} = \sum_{i,j} \delta[f,i,j]\, X[c,\,iS+m,\,jS+n]$$
$$\frac{\partial L}{\partial b_f} = \sum_{i,j}\delta[f,i,j], \qquad
\frac{\partial L}{\partial X} = \text{col2im}(W_{\text{col}}^{\top}\,\delta_{\text{col}})$$

## 3. ReLU

$$f(u) = \max(0, u), \qquad \frac{\partial L}{\partial u} = \delta \cdot \mathbb{1}[u > 0]$$

## 4. Max Pooling

**Forward:** $\; O[i,j] = \max_{(m,n)\in \text{jendela}} X[\,iS+m,\, jS+n]$

**Backward:** gradien $\delta$ diteruskan hanya ke posisi pemenang (argmax);
posisi lain bernilai 0.

## 5. Fully-Connected (Dense)

**Forward:** $\; y = xW + b$

**Backward:**
$$\frac{\partial L}{\partial W} = x^{\top}\delta, \qquad
\frac{\partial L}{\partial b} = \sum \delta, \qquad
\frac{\partial L}{\partial x} = \delta W^{\top}$$

## 6. Softmax

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

(Implementasi mengurangi $\max(z)$ sebelum eksponen agar stabil secara numerik.)

## 7. Cross-Entropy + gradien gabungan

$$L = -\sum_{i} y_i \ln p_i, \qquad \frac{\partial L}{\partial z_i} = p_i - y_i$$

## 8. Inisialisasi bobot

$$\text{He (ReLU)}: \; W \sim \mathcal{N}\!\left(0, \tfrac{2}{n_{in}}\right), \qquad
\text{Xavier}: \; W \sim \mathcal{N}\!\left(0, \tfrac{1}{n_{in}}\right)$$

## 9. SGD + Momentum

$$v \leftarrow \mu v - \eta\,\nabla, \qquad \theta \leftarrow \theta + v$$

## 10. Metrik evaluasi

$$\text{Akurasi} = \frac{TP+TN}{TP+TN+FP+FN}, \qquad
\text{Presisi} = \frac{TP}{TP+FP}$$
$$\text{Recall} = \frac{TP}{TP+FN}, \qquad
F1 = \frac{2\cdot\text{Presisi}\cdot\text{Recall}}{\text{Presisi}+\text{Recall}}$$

## Verifikasi numerik (gradient checking)

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta+\varepsilon) - L(\theta-\varepsilon)}{2\varepsilon}$$

Galat relatif terhadap gradien analitik < $10^{-5}$ untuk semua lapisan
(hasil aktual: $10^{-9}$-$10^{-11}$).
