from __future__ import annotations

import json
import os
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_task import preprocess_for_model

CLASS_TO_ID = {
    "pozytywny": 1,
    "neutralny": 0,
    "negatywny": -1,
}

ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}
VALID_CLASSES = set(CLASS_TO_ID.keys())


def normalize_class_name(label: str) -> str:
    return (label or "").strip().lower()


def is_valid_class(label: str) -> bool:
    return normalize_class_name(label) in VALID_CLASSES


def _sanitize_record(record: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(record, dict):
        return None

    text = (record.get("text") or "").strip()
    class_name = normalize_class_name(record.get("class") or "")

    if not text or class_name not in VALID_CLASSES:
        return None

    # Keep only the required two fields.
    return {"text": text, "class": class_name}


def load_sentences(data_path: str = "sentences.json") -> list[dict[str, str]]:
    if not os.path.exists(data_path):
        return []

    with open(data_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("JSON data must be a list of records.")

    records: list[dict[str, str]] = []
    for raw_record in payload:
        sanitized = _sanitize_record(raw_record)
        if sanitized:
            records.append(sanitized)

    return records


def append_records(records: list[dict[str, str]], data_path: str = "sentences.json") -> list[dict[str, str]]:
    existing_records = load_sentences(data_path)
    sanitized_new = []

    for record in records:
        sanitized = _sanitize_record(record)
        if sanitized:
            sanitized_new.append(sanitized)

    merged = existing_records + sanitized_new
    with open(data_path, "w", encoding="utf-8") as file:
        json.dump(merged, file, ensure_ascii=False, indent=2)

    return merged


def prepare_training_data(
    records: list[dict[str, str]],
    use_preprocessing: bool = True,
) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []

    for record in records:
        sanitized = _sanitize_record(record)
        if not sanitized:
            continue

        text = sanitized["text"]
        class_name = sanitized["class"]
        if use_preprocessing:
            processed = preprocess_for_model(text)
            text = processed if processed else text

        texts.append(text)
        labels.append(CLASS_TO_ID[class_name])

    return texts, labels


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
            (
                "classifier",
                LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42),
            ),
        ]
    )


def predict_class(
    records: list[dict[str, str]],
    new_text: str,
    use_preprocessing: bool = True,
) -> str:
    text = (new_text or "").strip()
    if not text:
        raise ValueError("Input text for classification cannot be empty.")

    train_texts, train_labels = prepare_training_data(records, use_preprocessing=use_preprocessing)

    if len(train_texts) < 2:
        raise ValueError("Not enough training samples in sentences.json.")

    if len(set(train_labels)) < 2:
        raise ValueError("Classifier needs at least two different classes in training data.")

    model = build_model()
    model.fit(train_texts, train_labels)

    if use_preprocessing:
        processed = preprocess_for_model(text)
        text = processed if processed else text

    predicted_id = int(model.predict([text])[0])
    return ID_TO_CLASS.get(predicted_id, "neutralny")
