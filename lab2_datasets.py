from __future__ import annotations

import csv
import os
from typing import Any

DEFAULT_20NEWS_CATEGORIES = ["sci.space", "comp.graphics", "rec.autos", "sci.med"]
MAX_SAMPLES_PER_CLASS = int(os.getenv("LAB2_MAX_SAMPLES_PER_CLASS", "120"))
DATASETS_DIR = os.getenv("LAB2_DATASETS_DIR", "datasets")

BUILTIN_DATASETS = {"20news_group", "ag_news"}


def _cap_per_class(texts: list[str], labels: list[str], max_per_class: int) -> tuple[list[str], list[str]]:
    if max_per_class <= 0:
        return texts, labels

    counts: dict[str, int] = {}
    out_texts: list[str] = []
    out_labels: list[str] = []
    for text, label in zip(texts, labels):
        counts.setdefault(label, 0)
        if counts[label] >= max_per_class:
            continue
        counts[label] += 1
        out_texts.append(text)
        out_labels.append(label)

    return out_texts, out_labels


def _load_20news_group() -> tuple[list[str], list[str]]:
    from sklearn.datasets import fetch_20newsgroups

    bunch = fetch_20newsgroups(
        subset="train",
        categories=DEFAULT_20NEWS_CATEGORIES,
        remove=("headers", "footers", "quotes"),
    )
    texts = [text for text in bunch.data]
    labels = [bunch.target_names[target] for target in bunch.target]
    return texts, labels


def _load_ag_news() -> tuple[list[str], list[str]]:
    csv_path = os.path.join(DATASETS_DIR, "ag_news.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "ag_news requires a local file at "
            f"'{csv_path}' with columns 'text' and 'label'. "
            "AG News cannot be auto-downloaded without Kaggle credentials."
        )
    return _load_csv_dataset(csv_path)


def _load_csv_dataset(csv_path: str) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"CSV dataset '{csv_path}' must have 'text' and 'label' columns.")

        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)

    return texts, labels


def load_dataset(name: str, max_samples_per_class: int | None = None) -> tuple[list[str], list[str]]:
    normalized = (name or "").strip().lower()
    cap = MAX_SAMPLES_PER_CLASS if max_samples_per_class is None else max_samples_per_class

    if normalized == "20news_group":
        texts, labels = _load_20news_group()
    elif normalized == "ag_news":
        texts, labels = _load_ag_news()
    else:
        csv_path = os.path.join(DATASETS_DIR, f"{normalized}.csv")
        if os.path.exists(csv_path):
            texts, labels = _load_csv_dataset(csv_path)
        else:
            raise ValueError(
                f"Unknown dataset '{name}'. Built-in datasets: {sorted(BUILTIN_DATASETS)}. "
                f"For other datasets, place a CSV with 'text' and 'label' columns at '{csv_path}'."
            )

    if not texts:
        raise ValueError(f"Dataset '{name}' produced no usable records.")

    texts, labels = _cap_per_class(texts, labels, cap)

    if len({label for label in labels}) < 2:
        raise ValueError(f"Dataset '{name}' must contain at least two classes after loading.")

    return texts, labels
