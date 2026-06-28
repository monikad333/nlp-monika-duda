from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from lab03.config import PLOTS_DIR, RESULTS_PATH, SENTIMENT_METHODS, VALID_LABELS
from lab03.datasets import load_dataset
from lab03.sentiment_methods import SentimentMethodError, run_sentiment_method
from lab03.visualizations import (
    save_class_distribution_plot,
    save_compare_methods_plot,
    save_confusion_matrix,
    save_wordcloud_for_class,
)
from lab03.utils import CommandArgsError


def resolve_methods(methods_arg: str) -> list[str]:
    normalized = (methods_arg or "").strip().lower()
    methods = [item.strip() for item in normalized.split(",") if item.strip()]
    invalid = [item for item in methods if item not in SENTIMENT_METHODS]
    if invalid:
        raise CommandArgsError(f"Unknown method(s) {invalid}. Allowed: {SENTIMENT_METHODS}")
    return methods


def run_compare(dataset_name: str, methods_arg: str, sample_limit: int = 30) -> dict[str, Any]:
    methods = resolve_methods(methods_arg)
    texts, labels = load_dataset(dataset_name, max_samples_per_class=sample_limit)

    by_class: dict[str, list[str]] = {}
    for text, label in zip(texts, labels):
        by_class.setdefault(label, []).append(text)
    for class_name, class_texts in by_class.items():
        save_wordcloud_for_class(class_texts, class_name, f"{PLOTS_DIR}/wordcloud_{class_name}.png")
    save_class_distribution_plot(labels, f"{PLOTS_DIR}/class_distribution_{dataset_name}.png")

    rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}

    for method in methods:
        predictions: list[str] = []
        true_labels: list[str] = []

        for text, true_label in zip(texts, labels):
            try:
                result = run_sentiment_method(method, text)
            except SentimentMethodError as exc:
                skipped[method] = str(exc)
                predictions = []
                break
            predictions.append(result["label"])
            true_labels.append(true_label)

        if not predictions:
            continue

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average="macro", zero_division=0)
        recall = recall_score(true_labels, predictions, average="macro", zero_division=0)
        macro_f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)

        confusion_path = f"{PLOTS_DIR}/confusion_{method}_{dataset_name}.png"
        present_labels = sorted(set(true_labels) | set(predictions))
        save_confusion_matrix(true_labels, predictions, present_labels, confusion_path)

        rows.append(
            {
                "dataset": dataset_name,
                "method": method,
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "macro_f1": round(macro_f1, 4),
                "model_path": confusion_path,
            }
        )

    if rows:
        _append_results_csv(rows)
        compare_plot_path = save_compare_methods_plot(rows, f"{PLOTS_DIR}/compare_methods_{dataset_name}.png")
    else:
        compare_plot_path = None

    return {"rows": rows, "skipped": skipped, "compare_plot_path": compare_plot_path}


def _append_results_csv(rows: list[dict[str, Any]], results_path: str = RESULTS_PATH) -> None:
    import csv
    import os

    fields = ["dataset", "method", "accuracy", "precision", "recall", "macro_f1", "model_path"]
    file_exists = os.path.exists(results_path)
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    with open(results_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
