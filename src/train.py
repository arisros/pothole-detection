"""Latih LeNet-5 (implementasi manual) pada dataset jalan berlubang.

Alur:
    1. Muat dataset.npz (sudah dipreprocess).
    2. (Sanity check) overfit satu batch kecil -> akurasi ~1.0 sebagai bukti
       model & backprop dapat belajar.
    3. Latih penuh dengan SGD + momentum; catat history per epoch.
    4. Simpan weights.npz dan history.csv.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.cnn.model import LeNet5  # noqa: E402
from src.cnn.trainer import Trainer  # noqa: E402


def load_dataset():
    data = np.load(config.DATASET_NPZ)
    return (data["x_train"], data["y_train"],
            data["x_val"], data["y_val"],
            data["x_test"], data["y_test"])


def sanity_overfit(x_train, y_train):
    """Latih pada 32 sampel saja; harus mencapai akurasi train tinggi."""
    print("\n[Sanity check] overfit batch kecil ...")
    idx = np.arange(min(32, len(y_train)))
    m = LeNet5(seed=config.SEED)
    t = Trainer(m, lr=config.LEARNING_RATE, momentum=config.MOMENTUM,
                batch_size=16, seed=config.SEED)
    t.fit(x_train[idx], y_train[idx], epochs=40, verbose=False)
    acc = float(np.mean(m.predict(x_train[idx]) == y_train[idx]))
    print(f"  akurasi overfit = {acc:.3f}  "
          f"({'OK, model belajar' if acc > 0.95 else 'PERIKSA implementasi'})")


def save_history(history, path):
    epochs = len(history["train_loss"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for e in range(epochs):
            w.writerow([
                e + 1,
                f"{history['train_loss'][e]:.6f}",
                f"{history['train_acc'][e]:.6f}",
                f"{history['val_loss'][e]:.6f}" if history["val_loss"] else "",
                f"{history['val_acc'][e]:.6f}" if history["val_acc"] else "",
            ])


def main():
    np.random.seed(config.SEED)
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset()
    print(f"train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    sanity_overfit(x_train, y_train)

    print("\n[Pelatihan penuh]")
    model = LeNet5(seed=config.SEED)
    trainer = Trainer(model, lr=config.LEARNING_RATE, momentum=config.MOMENTUM,
                      batch_size=config.BATCH_SIZE, seed=config.SEED)
    history = trainer.fit(x_train, y_train, x_val, y_val,
                          epochs=config.EPOCHS, verbose=True)

    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    model.save(config.WEIGHTS_NPZ)
    save_history(history, config.HISTORY_CSV)
    print(f"\nBobot  -> {config.WEIGHTS_NPZ}")
    print(f"History-> {config.HISTORY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
