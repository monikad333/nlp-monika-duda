from __future__ import annotations

import csv
import os

from lab03.config import AMAZON_DATASET_PATH, CUSTOM_DATASET_PATH, MAX_SAMPLES_PER_CLASS, VALID_LABELS


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


def load_custom_dataset(csv_path: str = CUSTOM_DATASET_PATH) -> tuple[list[str], list[str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Custom dataset not found at '{csv_path}'.")

    texts: list[str] = []
    labels: list[str] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"Custom dataset '{csv_path}' must have 'text' and 'label' columns.")

        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if text and label in VALID_LABELS:
                texts.append(text)
                labels.append(label)

    return texts, labels


def append_to_custom_dataset(text: str, label: str, csv_path: str = CUSTOM_DATASET_PATH) -> None:
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["text", "label"])
        writer.writerow([text, label])


def _decode_imdb_sequence(sequence: list[int], reverse_word_index: dict[int, str]) -> str:
    words = [reverse_word_index.get(index - 3, "?") for index in sequence if index >= 3]
    return " ".join(words)


def load_imdb_dataset() -> tuple[list[str], list[str]]:
    from tensorflow.keras.datasets import imdb

    (train_data, train_labels), _ = imdb.load_data(num_words=10000)
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}

    texts = [_decode_imdb_sequence(sequence, reverse_word_index) for sequence in train_data]
    labels = ["pozytywny" if label == 1 else "negatywny" for label in train_labels]
    return texts, labels


def load_amazon_dataset(csv_path: str = AMAZON_DATASET_PATH) -> tuple[list[str], list[str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Dataset 'amazon' requires a local file at "
            f"'{csv_path}' with columns 'text' and 'label' "
            "(pozytywny/neutralny/negatywny). Amazon reviews cannot be auto-downloaded from Kaggle."
        )
    texts: list[str] = []
    labels: list[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if text and label in VALID_LABELS:
                texts.append(text)
                labels.append(label)
    return texts, labels


def load_dataset(name: str, max_samples_per_class: int | None = None) -> tuple[list[str], list[str]]:
    normalized = (name or "").strip().lower()
    cap = MAX_SAMPLES_PER_CLASS if max_samples_per_class is None else max_samples_per_class

    if normalized == "custom":
        texts, labels = load_custom_dataset()
    elif normalized == "imdb":
        texts, labels = load_imdb_dataset()
    elif normalized == "amazon":
        texts, labels = load_amazon_dataset()
    else:
        raise ValueError(f"Unknown dataset '{name}'. Allowed: amazon, imdb, custom.")

    if not texts:
        raise ValueError(f"Dataset '{name}' produced no usable records.")

    texts, labels = _cap_per_class(texts, labels, cap)

    if len(set(labels)) < 2:
        raise ValueError(f"Dataset '{name}' must contain at least two classes after loading.")

    return texts, labels
