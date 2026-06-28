from __future__ import annotations

import os
from collections import Counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from wordcloud import WordCloud

from lab2_embeddings import tokenize_for_embedding


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_wordcloud_corpus(texts: list[str], plots_dir: str) -> str:
    _ensure_dir(plots_dir)
    tokens: list[str] = []
    for text in texts:
        tokens.extend(tokenize_for_embedding(text))

    cloud = WordCloud(width=1000, height=500, background_color="white", collocations=False)
    cloud = cloud.generate(" ".join(tokens) if tokens else "no_data")

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.imshow(cloud, interpolation="bilinear")
    axis.axis("off")
    axis.set_title("Word cloud - corpus")

    path = os.path.join(plots_dir, "wordcloud_corpus.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_wordcloud_per_class(texts: list[str], labels: list[str], plots_dir: str) -> list[str]:
    _ensure_dir(plots_dir)
    paths: list[str] = []
    by_class: dict[str, list[str]] = {}
    for text, label in zip(texts, labels):
        by_class.setdefault(label, []).append(text)

    for class_name, class_texts in by_class.items():
        tokens: list[str] = []
        for text in class_texts:
            tokens.extend(tokenize_for_embedding(text))

        cloud = WordCloud(width=1000, height=500, background_color="white", collocations=False)
        cloud = cloud.generate(" ".join(tokens) if tokens else "no_data")

        fig, axis = plt.subplots(figsize=(10, 5))
        axis.imshow(cloud, interpolation="bilinear")
        axis.axis("off")
        axis.set_title(f"Word cloud - class {class_name}")

        safe_class = "".join(char if char.isalnum() else "_" for char in class_name)
        path = os.path.join(plots_dir, f"wordcloud_class_{safe_class}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


def _to_dense(X: Any) -> np.ndarray:
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def save_embedding_reduction(
    X: Any,
    labels: list[str],
    method: str,
    path: str,
    seed: int = 42,
) -> str:
    dense = _to_dense(X)
    normalized_method = method.lower()

    if normalized_method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
    elif normalized_method == "tsne":
        reducer = TSNE(n_components=2, random_state=seed, init="random", perplexity=min(30, max(5, len(labels) // 3)))
    elif normalized_method == "svd":
        reducer = TruncatedSVD(n_components=2, random_state=seed)
    else:
        raise ValueError(f"Unknown reduction method '{method}'. Allowed: pca, tsne, svd")

    coords = reducer.fit_transform(dense)

    fig, axis = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    colormap = matplotlib.colormaps["tab10"]

    for index, class_name in enumerate(unique_labels):
        mask = [label == class_name for label in labels]
        axis.scatter(
            coords[mask, 0], coords[mask, 1], label=class_name, color=colormap(index), alpha=0.7, s=20
        )

    axis.legend()
    axis.set_title(f"Embedding visualization ({normalized_method.upper()})")

    _ensure_dir(os.path.dirname(path) or ".")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


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


def save_feature_importance(
    model: Any,
    feature_names: list[str] | None,
    path: str,
    top_n: int = 10,
) -> str | None:
    importances: np.ndarray | None = None

    estimator = getattr(model, "best_estimator_", model)

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

    if importances is None or feature_names is None:
        return None

    top_indices = np.argsort(importances)[::-1][:top_n]
    lines = [f"{feature_names[index]}: {importances[index]:.6f}" for index in top_indices]

    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return path


def save_word_embedding_plot(words: list[str], keyed_vectors: Any, method: str, path: str, seed: int = 42) -> str | None:
    available_words = [word for word in words if word in keyed_vectors]
    if len(available_words) < 2:
        return None

    vectors = np.array([keyed_vectors[word] for word in available_words])
    normalized_method = method.lower()

    if normalized_method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
    elif normalized_method == "tsne":
        reducer = TSNE(n_components=2, random_state=seed, init="random", perplexity=min(5, len(available_words) - 1))
    else:
        raise ValueError(f"Unknown reduction method '{method}'. Allowed: pca, tsne")

    coords = reducer.fit_transform(vectors)

    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(coords[:, 0], coords[:, 1], color="#4e79a7")
    for index, word in enumerate(available_words):
        axis.annotate(word, (coords[index, 0], coords[index, 1]))

    axis.set_title(f"Word embedding visualization ({normalized_method.upper()})")

    _ensure_dir(os.path.dirname(path) or ".")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
