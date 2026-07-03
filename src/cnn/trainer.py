"""Training loop mini-batch — ditulis manual.

Satu epoch = satu kali lewat seluruh data latih. Untuk tiap mini-batch:

    1. forward  : skor = model(x)
    2. loss     : L = SoftmaxCrossEntropy(skor, y)
    3. backward : model.backward(dL/dskor)
    4. update   : optimizer.step()  (SGD + momentum)

Loss dan akurasi train/val dicatat tiap epoch ke history.
"""
from __future__ import annotations

import copy

import numpy as np

from .augment import random_augment
from .losses import SoftmaxCrossEntropy
from .optim import Adam, SGDMomentum


class Trainer:
    def __init__(self, model, lr=0.01, momentum=0.9, batch_size=32, seed=42,
                 weight_decay=0.0, augment=False, aug_params=None,
                 optimizer="sgd", cosine_lr=False):
        self.model = model
        self.batch_size = batch_size
        self.augment = augment
        # Kekuatan augmentasi daring (None -> pakai default random_augment).
        self.aug_params = aug_params or {}
        self.loss_fn = SoftmaxCrossEntropy()
        # Hanya lapisan berparameter yang diberi optimizer state.
        param_layers = [l for l in model.layers if l.params]
        if optimizer == "adam":
            self.optimizer = Adam(param_layers, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = SGDMomentum(param_layers, lr=lr, momentum=momentum,
                                         weight_decay=weight_decay)
        self.base_lr = lr
        self.cosine_lr = cosine_lr
        self.rng = np.random.default_rng(seed)
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

    def _iterate_minibatches(self, x, y, shuffle=True):
        idx = np.arange(x.shape[0])
        if shuffle:
            self.rng.shuffle(idx)
        for start in range(0, x.shape[0], self.batch_size):
            batch = idx[start:start + self.batch_size]
            yield x[batch], y[batch]

    def _evaluate(self, x, y):
        """Loss & akurasi rata-rata pada satu himpunan (tanpa update)."""
        self.model.eval()  # dropout mati saat evaluasi
        total_loss, total_correct, n = 0.0, 0, x.shape[0]
        for start in range(0, n, self.batch_size):
            xb = x[start:start + self.batch_size]
            yb = y[start:start + self.batch_size]
            scores = self.model.forward(xb)
            total_loss += self.loss_fn.forward(scores, yb) * xb.shape[0]
            total_correct += int(np.sum(np.argmax(scores, axis=1) == yb))
        return total_loss / n, total_correct / n

    def _snapshot(self):
        """Salin seluruh parameter model (untuk menyimpan bobot val terbaik)."""
        return [{k: v.copy() for k, v in l.params.items()} for l in self.model.layers]

    def _restore(self, snapshot):
        for layer, snap in zip(self.model.layers, snapshot):
            for k in layer.params:
                layer.params[k] = snap[k].copy()

    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=15,
            verbose=True, restore_best=True):
        best_val_acc, best_snapshot, best_epoch = -1.0, None, 0
        for epoch in range(1, epochs + 1):
            # Jadwal cosine: lr meluruh mulus dari base_lr -> ~0 sepanjang latih,
            # menstabilkan epoch akhir (mengurangi val bouncing).
            if self.cosine_lr:
                self.optimizer.lr = 0.5 * self.base_lr * (
                    1.0 + np.cos(np.pi * (epoch - 1) / max(1, epochs - 1)))
            running_loss, seen = 0.0, 0
            self.model.train()  # dropout hidup saat latih
            for xb, yb in self._iterate_minibatches(x_train, y_train):
                if self.augment:
                    xb = random_augment(xb, self.rng, **self.aug_params)
                scores = self.model.forward(xb)
                loss = self.loss_fn.forward(scores, yb)
                dscores = self.loss_fn.backward()
                self.model.backward(dscores)
                self.optimizer.step()

                running_loss += loss * xb.shape[0]
                seen += xb.shape[0]

            train_loss, train_acc = self._evaluate(x_train, y_train)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            if x_val is not None:
                val_loss, val_acc = self._evaluate(x_val, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                if val_acc > best_val_acc:
                    best_val_acc, best_epoch = val_acc, epoch
                    best_snapshot = self._snapshot()
                if verbose:
                    print(f"Epoch {epoch:2d}/{epochs} | "
                          f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                          f"val_loss={val_loss:.4f} acc={val_acc:.4f}")
            elif verbose:
                print(f"Epoch {epoch:2d}/{epochs} | "
                      f"train_loss={train_loss:.4f} acc={train_acc:.4f}")

        if restore_best and best_snapshot is not None:
            self._restore(best_snapshot)
            if verbose:
                print(f"-> memulihkan bobot val terbaik "
                      f"(epoch {best_epoch}, val_acc={best_val_acc:.4f})")
        self.best_epoch = best_epoch
        self.best_val_acc = best_val_acc
        return self.history
