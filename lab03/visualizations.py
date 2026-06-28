from __future__ import annotations

import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from wordcloud import WordCloud

from lab03.preprocessing import clean_texts


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str], path: str) -> str:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Confusion matrix")

    _ensure_dir(os.path.dirname(path) or ".")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_train_history_plot(history: dict[str, list[float]], path: str) -> str:
    fig, (acc_axis, loss_axis) = plt.subplots(1, 2, figsize=(11, 4))

    if "accuracy" in history:
        acc_axis.plot(history["accuracy"], label="train")
    if "val_accuracy" in history:
        acc_axis.plot(history["val_accuracy"], label="val")
    acc_axis.set_title("Accuracy")
    acc_axis.set_xlabel("Epoch")
    acc_axis.legend()

    if "loss" in history:
        loss_axis.plot(history["loss"], label="train")
    if "val_loss" in history:
        loss_axis.plot(history["val_loss"], label="val")
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.legend()

    _ensure_dir(os.path.dirname(path) or ".")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_compare_methods_plot(rows: list[dict[str, object]], path: str) -> str:
    methods = [row["method"] for row in rows]
    accuracies = [row["accuracy"] for row in rows]

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(methods, accuracies, color="#4e79a7")
    axis.set_ylabel("Accuracy")
    axis.set_title("Method comparison")
    axis.tick_params(axis="x", rotation=30)

    _ensure_dir(os.path.dirname(path) or ".")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_wordcloud_for_class(texts: list[str], class_name: str, path: str) -> str:
    tokens: list[str] = []
    for text in clean_texts(texts):
        tokens.extend(text.split())

    cloud = WordCloud(width=1000, height=500, background_color="white", collocations=False)
    cloud = cloud.generate(" ".join(tokens) if tokens else "no_data")

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.imshow(cloud, interpolation="bilinear")
    axis.axis("off")
    axis.set_title(f"Word cloud - {class_name}")

    _ensure_dir(os.path.dirname(path) or ".")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_class_distribution_plot(labels: list[str], path: str) -> str:
    counts = Counter(labels)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(list(counts.keys()), list(counts.values()), color="#59a14f")
    axis.set_ylabel("Count")
    axis.set_title("Class distribution")

    _ensure_dir(os.path.dirname(path) or ".")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
