"""Evaluasi model pada test set + render figur untuk makalah.

Menghasilkan:
    - metrik test: akurasi, presisi, recall, F1 + confusion matrix (dicetak)
    - experiments/figures/training_curves.png   (loss & akurasi train/val)
    - experiments/figures/confusion_matrix.png
    - experiments/figures/sample_predictions.png
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.cnn.metrics import binary_metrics, confusion_matrix  # noqa: E402
from src.cnn.model import LeNet5  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_history(path):
    epochs, tl, ta, vl, va = [], [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            epochs.append(int(r["epoch"]))
            tl.append(float(r["train_loss"]))
            ta.append(float(r["train_acc"]))
            vl.append(float(r["val_loss"]) if r["val_loss"] else np.nan)
            va.append(float(r["val_acc"]) if r["val_acc"] else np.nan)
    return epochs, tl, ta, vl, va


def plot_curves(path_in, path_out):
    epochs, tl, ta, vl, va = load_history(path_in)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, tl, "-o", label="train", ms=3)
    ax1.plot(epochs, vl, "-s", label="val", ms=3)
    ax1.set_title("Loss (cross-entropy)"); ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, ta, "-o", label="train", ms=3)
    ax2.plot(epochs, va, "-s", label="val", ms=3)
    ax2.set_title("Akurasi"); ax2.set_xlabel("epoch")
    ax2.set_ylabel("akurasi"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path_out, dpi=120); plt.close(fig)


def plot_confusion(cm, path_out):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(config.CLASS_NAMES); ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Asli")
    ax.set_title("Confusion Matrix (test)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path_out, dpi=120); plt.close(fig)


def plot_samples(model, x_test, y_test, path_out, n=8):
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(y_test), size=min(n, len(y_test)), replace=False)
    probs = model.predict_proba(x_test[idxs])
    preds = np.argmax(probs, axis=1)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    for ax, idx, pred, prob in zip(axes.ravel(), idxs, preds, probs):
        ax.imshow(x_test[idx, 0], cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
        true = y_test[idx]
        ok = pred == true
        ax.set_title(
            f"asli={config.CLASS_NAMES[true]}\n"
            f"pred={config.CLASS_NAMES[pred]} ({prob[pred]:.2f})",
            color="green" if ok else "red", fontsize=9)
    fig.suptitle("Contoh prediksi pada test set (hijau=benar, merah=salah)")
    fig.tight_layout(); fig.savefig(path_out, dpi=120); plt.close(fig)


def main():
    data = np.load(config.DATASET_NPZ)
    x_test, y_test = data["x_test"], data["y_test"]

    model = LeNet5(seed=config.SEED).load(config.WEIGHTS_NPZ)
    y_pred = model.predict(x_test)

    m = binary_metrics(y_test, y_pred, positive=1)
    cm = confusion_matrix(y_test, y_pred, config.NUM_CLASSES)

    print("Evaluasi Test Set")
    print("=================")
    print(f"  Akurasi : {m['accuracy']:.4f}")
    print(f"  Presisi : {m['precision']:.4f}")
    print(f"  Recall  : {m['recall']:.4f}")
    print(f"  F1-score: {m['f1']:.4f}")
    print(f"  TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']}")
    print("\nConfusion matrix (baris=asli, kolom=prediksi):")
    print(f"           pred:normal  pred:pothole")
    print(f"  normal      {cm[0,0]:5d}        {cm[0,1]:5d}")
    print(f"  pothole     {cm[1,0]:5d}        {cm[1,1]:5d}")

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    if os.path.exists(config.HISTORY_CSV):
        plot_curves(config.HISTORY_CSV,
                    os.path.join(config.FIGURES_DIR, "training_curves.png"))
    plot_confusion(cm, os.path.join(config.FIGURES_DIR, "confusion_matrix.png"))
    plot_samples(model, x_test, y_test,
                 os.path.join(config.FIGURES_DIR, "sample_predictions.png"))
    print(f"\nFigur tersimpan di {config.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
