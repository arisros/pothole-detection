"""Metrik evaluasi klasifikasi ,  dihitung manual dari confusion matrix.

Untuk klasifikasi biner (kelas positif = 1 = pothole):

    TP = benar diprediksi positif      FP = negatif diprediksi positif
    TN = benar diprediksi negatif      FN = positif diprediksi negatif

    Akurasi  = (TP + TN) / (TP + TN + FP + FN)
    Presisi  = TP / (TP + FP)
    Recall   = TP / (TP + FN)
    F1       = 2 * Presisi * Recall / (Presisi + Recall)
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes=2):
    """Confusion matrix (baris = kelas asli, kolom = kelas prediksi)."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _safe_div(a, b):
    return a / b if b > 0 else 0.0


def binary_metrics(y_true, y_pred, positive=1):
    """Hitung akurasi, presisi, recall, F1 untuk kelas positif tertentu."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_pred == positive) & (y_true == positive)))
    tn = int(np.sum((y_pred != positive) & (y_true != positive)))
    fp = int(np.sum((y_pred == positive) & (y_true != positive)))
    fn = int(np.sum((y_pred != positive) & (y_true == positive)))

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))
