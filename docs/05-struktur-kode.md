# 05 — Struktur Kode (Class Diagram)

Modul inti `src/cnn/` dirancang modular: tiap lapisan adalah objek dengan
antarmuka `forward`/`backward` yang seragam, sehingga model hanya merangkai
daftar lapisan.

## Diagram kelas

```mermaid
classDiagram
    class Layer {
        +dict params
        +dict grads
        +forward(x) out
        +backward(dout) dx
    }
    class Conv2D {
        +int in_channels
        +int out_channels
        +int kh, kw, stride, pad
        +forward(x)
        +backward(dout)
    }
    class MaxPool2D {
        +int pool, stride
        +forward(x)
        +backward(dout)
    }
    class ReLU {
        +forward(x)
        +backward(dout)
    }
    class Flatten {
        +forward(x)
        +backward(dout)
    }
    class Dense {
        +forward(x)
        +backward(dout)
    }
    Layer <|-- Conv2D
    Layer <|-- MaxPool2D
    Layer <|-- ReLU
    Layer <|-- Flatten
    Layer <|-- Dense

    class LeNet5 {
        +list layers
        +forward(x)
        +backward(dscores)
        +predict(x)
        +save(path)
        +load(path)
    }
    class SoftmaxCrossEntropy {
        +forward(scores, y) loss
        +backward() dscores
    }
    class SGDMomentum {
        +float lr, momentum
        +dict velocities
        +step()
        +zero_grad()
    }
    class Trainer {
        +fit(x_train, y_train, x_val, y_val)
        -_snapshot()
        -_restore()
    }

    LeNet5 o-- Layer : merangkai
    Trainer --> LeNet5 : melatih
    Trainer --> SoftmaxCrossEntropy : menghitung loss
    Trainer --> SGDMomentum : memperbarui bobot
```

## Tanggung jawab tiap berkas

| Berkas | Isi |
|--------|-----|
| `tensor_utils.py` | `im2col`, `col2im`, `conv_output_size` |
| `init.py` | inisialisasi He & Xavier |
| `layers.py` | `Conv2D`, `MaxPool2D`, `ReLU`, `Flatten`, `Dense` |
| `losses.py` | `softmax`, `SoftmaxCrossEntropy` |
| `optim.py` | `SGDMomentum` |
| `metrics.py` | confusion matrix, akurasi, presisi, recall, F1 |
| `model.py` | `LeNet5` (rakit lapisan, simpan/muat bobot) |
| `trainer.py` | `Trainer` (training loop + best-val checkpoint) |
| `gradcheck.py` | numerical gradient checking |

## Alur data antar objek

```mermaid
flowchart LR
    DS["dataset.npz"] --> TR["Trainer.fit"]
    TR --> M["LeNet5"]
    M --> WT["weights.npz"]
    WT --> EV["evaluate.py"]
    WT --> VI["visualize.py"]
    EV --> FG["figures/*.png"]
    VI --> FG
```
