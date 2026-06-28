from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Any

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from lab02.lab2_datasets import load_dataset
from lab02.lab2_embeddings import EMBEDDING_NAMES, build_embedding, similar_words
from lab02.lab2_models import build_model, resolve_methods
from lab02.lab2_visualize import (
    save_confusion_matrix,
    save_embedding_reduction,
    save_feature_importance,
    save_word_embedding_plot,
    save_wordcloud_corpus,
    save_wordcloud_per_class,
)

SEEDS = [42, 1337, 2024]
SIMILARITY_QUERY_WORDS = ["space", "computer", "science", "music", "car"]
RESULTS_FIELDS = ["dataset", "embedding", "model", "accuracy", "macro_f1", "seed"]


class ClassifyArgsError(ValueError):
    pass


def parse_classify_args(args_text: str) -> dict[str, str]:
    tokens = (args_text or "").strip().split()
    params: dict[str, str] = {}

    for token in tokens:
        if "=" not in token:
            raise ClassifyArgsError(f"Invalid argument '{token}'. Expected key=value.")
        key, value = token.split("=", 1)
        params[key.strip().lower()] = value.strip()

    required = {"dataset", "method", "gridsearch", "run"}
    missing = required - params.keys()
    if missing:
        raise ClassifyArgsError(
            "Missing parameters: "
            f"{sorted(missing)}. Usage: /classify dataset=<name> method=<model|all> gridsearch=<true/false> run=<n>"
        )

    return params


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def _write_results_csv(rows: list[dict[str, Any]], results_path: str) -> None:
    file_exists = os.path.exists(results_path)
    with open(results_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULTS_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_classify_experiment(
    dataset_name: str,
    method_arg: str,
    use_gridsearch: bool,
    run_count: int,
    plots_dir: str = "lab2plots",
    results_path: str = "lab2results.csv",
) -> dict[str, Any]:
    if run_count < 1:
        raise ClassifyArgsError("run must be >= 1.")

    methods = resolve_methods(method_arg)
    seeds = SEEDS[:run_count] if run_count <= len(SEEDS) else SEEDS + [SEEDS[-1] + i for i in range(run_count - len(SEEDS))]

    texts, labels = load_dataset(dataset_name)
    os.makedirs(plots_dir, exist_ok=True)

    generated_files: list[str] = []
    generated_files.append(save_wordcloud_corpus(texts, plots_dir))
    generated_files.extend(save_wordcloud_per_class(texts, labels, plots_dir))

    all_rows: list[dict[str, Any]] = []
    similar_words_by_embedding: dict[str, dict[str, Any]] = {}

    for run_index, seed in enumerate(seeds):
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.25, random_state=seed, stratify=labels
        )

        for embedding_name in EMBEDDING_NAMES:
            embedding = build_embedding(embedding_name, train_texts, test_texts, seed=seed)

            if run_index == 0:
                reduction_path_base = f"{_safe_name(dataset_name)}_{embedding_name}"
                for reduction_method, suffix in (("pca", "pca"), ("tsne", "tsne"), ("svd", "svd")):
                    path = os.path.join(plots_dir, f"{reduction_path_base}_{suffix}_embedding.png")
                    save_embedding_reduction(embedding.X_train, train_labels, reduction_method, path, seed=seed)
                    generated_files.append(path)

            if (
                run_index == 0
                and embedding_name in {"word2vec", "glove"}
                and embedding.keyed_vectors is not None
                and embedding_name not in similar_words_by_embedding
            ):
                matches = similar_words(embedding.keyed_vectors, SIMILARITY_QUERY_WORDS, topn=5)
                similar_words_by_embedding[embedding_name] = matches

                pca_path = save_word_embedding_plot(
                    SIMILARITY_QUERY_WORDS,
                    embedding.keyed_vectors,
                    "pca",
                    os.path.join(plots_dir, f"word_embedding_{embedding_name}_pca.png"),
                )
                tsne_path = save_word_embedding_plot(
                    SIMILARITY_QUERY_WORDS,
                    embedding.keyed_vectors,
                    "tsne",
                    os.path.join(plots_dir, f"word_embedding_{embedding_name}_tsne.png"),
                )
                for path in (pca_path, tsne_path):
                    if path:
                        generated_files.append(path)

            for method_name in methods:
                model = build_model(method_name, embedding.is_dense, seed, use_gridsearch)
                model.fit(embedding.X_train, train_labels)
                predictions = model.predict(embedding.X_test)

                accuracy = accuracy_score(test_labels, predictions)
                macro_f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)

                all_rows.append(
                    {
                        "dataset": dataset_name,
                        "embedding": embedding_name,
                        "model": method_name,
                        "accuracy": round(accuracy, 4),
                        "macro_f1": round(macro_f1, 4),
                        "seed": seed,
                    }
                )

                if run_index == 0:
                    class_labels = sorted(set(labels))
                    confusion_path = os.path.join(plots_dir, f"confusion_{embedding_name}_{method_name}.png")
                    save_confusion_matrix(test_labels, list(predictions), class_labels, confusion_path)
                    generated_files.append(confusion_path)

                    importance_path = os.path.join(
                        plots_dir, f"feature_importance_{_safe_name(dataset_name)}_{embedding_name}_{method_name}.txt"
                    )
                    saved_path = save_feature_importance(model, embedding.feature_names, importance_path)
                    if saved_path:
                        generated_files.append(saved_path)

    if similar_words_by_embedding:
        similar_words_path = os.path.join(os.path.dirname(results_path) or ".", "lab2_similar_words.txt")
        with open(similar_words_path, "w", encoding="utf-8") as file:
            for embedding_name, matches in similar_words_by_embedding.items():
                file.write(f"[{embedding_name}]\n")
                for word, neighbours in matches.items():
                    file.write(f"{word}: {neighbours}\n")
                file.write("\n")
        generated_files.append(similar_words_path)

    _write_results_csv(all_rows, results_path)

    averages: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"accuracy": 0.0, "macro_f1": 0.0, "count": 0})
    for row in all_rows:
        key = (row["embedding"], row["model"])
        averages[key]["accuracy"] += row["accuracy"]
        averages[key]["macro_f1"] += row["macro_f1"]
        averages[key]["count"] += 1

    averaged_summary = [
        {
            "embedding": embedding_name,
            "model": method_name,
            "accuracy": round(stats["accuracy"] / stats["count"], 4),
            "macro_f1": round(stats["macro_f1"] / stats["count"], 4),
        }
        for (embedding_name, method_name), stats in averages.items()
    ]

    return {
        "rows": all_rows,
        "averaged": averaged_summary,
        "generated_files": generated_files,
        "results_path": results_path,
    }
