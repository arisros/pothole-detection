"""Visualisasi *feature map* antar lapisan ,  memperlihatkan pergerakan data
melalui hidden layer CNN.

Untuk satu citra contoh tiap kelas, dijalankan forward pass dan keluaran tiap
tahap (Conv1+ReLU, Pool1, Conv2+ReLU, Pool2) dirender sebagai grid peta fitur.
Ini memperlihatkan bagaimana representasi berubah dari tepi/tekstur kasar di
lapisan awal menjadi fitur lebih abstrak di lapisan dalam.

Keluaran: experiments/figures/feature_maps_<kelas>.png
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.cnn.model import LeNet5  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def forward_capture(model, x):
    """Jalankan forward sambil menangkap keluaran tahap-tahap kunci."""
    stages = {}
    out = model.conv1.forward(x)
    out = model.relu1.forward(out)
    stages["Conv1+ReLU (6x28x28)"] = out
    out = model.pool1.forward(out)
    stages["Pool1 (6x14x14)"] = out
    out = model.conv2.forward(out)
    out = model.relu2.forward(out)
    stages["Conv2+ReLU (16x10x10)"] = out
    out = model.pool2.forward(out)
    stages["Pool2 (16x5x5)"] = out
    return stages


def plot_feature_maps(image, stages, title, path_out):
    n_stages = len(stages)
    max_maps = 8  # tampilkan hingga 8 kanal per tahap
    fig, axes = plt.subplots(n_stages + 1, max_maps,
                             figsize=(1.4 * max_maps, 1.4 * (n_stages + 1)))

    # Baris 0: citra masukan (hanya kolom pertama).
    for col in range(max_maps):
        ax = axes[0, col]; ax.axis("off")
    axes[0, 0].imshow(image[0], cmap="gray")
    axes[0, 0].set_title("Input 1x32x32", fontsize=8)

    for row, (name, fmap) in enumerate(stages.items(), start=1):
        maps = fmap[0]  # (C, H, W)
        for col in range(max_maps):
            ax = axes[row, col]; ax.axis("off")
            if col < maps.shape[0]:
                ax.imshow(maps[col], cmap="viridis")
        axes[row, 0].set_ylabel(name, fontsize=8, rotation=0,
                                ha="right", va="center")
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path_out, dpi=120)
    plt.close(fig)


def main():
    data = np.load(config.DATASET_NPZ)
    x_test, y_test = data["x_test"], data["y_test"]
    model = LeNet5(seed=config.SEED).load(config.WEIGHTS_NPZ)

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    for label, name in enumerate(config.CLASS_NAMES):
        idxs = np.where(y_test == label)[0]
        if len(idxs) == 0:
            continue
        idx = idxs[0]
        x = x_test[idx:idx + 1]
        stages = forward_capture(model, x)
        out_path = os.path.join(config.FIGURES_DIR, f"feature_maps_{name}.png")
        plot_feature_maps(x_test[idx], stages,
                          f"Pergerakan hidden layer ,  kelas '{name}'", out_path)
        print(f"  {out_path}")
    print("Selesai render feature map.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
