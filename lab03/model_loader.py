from __future__ import annotations

import os
import pickle
from typing import Any

from lab03.config import MODELS_DIR, SEQUENTIAL_MODELS


def model_paths(model_name: str, dataset_name: str, models_dir: str = MODELS_DIR) -> dict[str, str]:
    prefix = f"{model_name}_{dataset_name}"
    return {
        "model": os.path.join(models_dir, f"{prefix}.h5"),
        "tokenizer": os.path.join(models_dir, f"{prefix}_tokenizer.h5"),
        "label_encoder": os.path.join(models_dir, f"{prefix}_label_encoder.h5"),
    }


def save_artifacts(model: Any, tokenizer: Any, label_encoder: Any, model_name: str, dataset_name: str) -> dict[str, str]:
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = model_paths(model_name, dataset_name)

    model.save(paths["model"])

    with open(paths["tokenizer"], "wb") as file:
        pickle.dump(tokenizer, file)

    with open(paths["label_encoder"], "wb") as file:
        pickle.dump(label_encoder, file)

    return paths


def load_artifacts(model_name: str, dataset_name: str) -> tuple[Any, Any, Any]:
    from tensorflow.keras.models import load_model

    paths = model_paths(model_name, dataset_name)

    if not os.path.exists(paths["model"]):
        raise FileNotFoundError(
            f"Model '{model_name}' for dataset '{dataset_name}' not found at '{paths['model']}'. "
            f"Train it first with /train model={model_name} dataset={dataset_name}."
        )

    model = load_model(paths["model"])

    with open(paths["tokenizer"], "rb") as file:
        tokenizer = pickle.load(file)

    with open(paths["label_encoder"], "rb") as file:
        label_encoder = pickle.load(file)

    return model, tokenizer, label_encoder


def list_available_models(models_dir: str = MODELS_DIR) -> list[dict[str, Any]]:
    if not os.path.isdir(models_dir):
        return []

    entries = []
    seen = set()
    for filename in sorted(os.listdir(models_dir)):
        if not filename.endswith(".h5") or "_tokenizer" in filename or "_label_encoder" in filename:
            continue

        stem = filename[: -len(".h5")]
        if stem in seen:
            continue
        seen.add(stem)

        model_name = next((candidate for candidate in SEQUENTIAL_MODELS if stem.startswith(f"{candidate}_")), None)
        if not model_name:
            continue
        dataset_name = stem[len(model_name) + 1 :]

        paths = model_paths(model_name, dataset_name, models_dir)
        entries.append(
            {
                "model": model_name,
                "dataset": dataset_name,
                "has_tokenizer": os.path.exists(paths["tokenizer"]),
                "has_label_encoder": os.path.exists(paths["label_encoder"]),
            }
        )

    return entries
