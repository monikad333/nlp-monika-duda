from __future__ import annotations

import time
from typing import Any

from lab03.config import BATCH_SIZE, EMBEDDING_DIM, EPOCHS, MAX_LEN, MAX_VOCAB_SIZE, PLOTS_DIR
from lab03.datasets import load_dataset
from lab03.model_loader import save_artifacts
from lab03.preprocessing import build_tokenizer, clean_texts, texts_to_padded_sequences
from lab03.visualizations import save_train_history_plot


def _build_sequential_model(model_type: str, vocab_size: int, num_classes: int, max_len: int, units: int = 32):
    from tensorflow.keras.layers import GRU, LSTM, Dense, Embedding, SimpleRNN
    from tensorflow.keras.models import Sequential

    recurrent_layer = {
        "simplernn": SimpleRNN(units),
        "lstm": LSTM(units),
        "gru": GRU(units),
    }[model_type]

    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len),
            recurrent_layer,
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _effective_max_len(texts: list[str], configured_max_len: int) -> int:
    token_counts = [len(text.split()) for text in texts]
    if not token_counts:
        return configured_max_len
    token_counts.sort()
    index = min(len(token_counts) - 1, int(len(token_counts) * 0.95))
    suggested = token_counts[index] + 2
    return max(10, min(configured_max_len, suggested))


def _effective_batch_size(train_size: int, configured_batch_size: int) -> int:
    return max(2, min(configured_batch_size, train_size // 4))


def train_sequential_model(
    model_type: str,
    dataset_name: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.callbacks import EarlyStopping

    normalized_model = (model_type or "").strip().lower()
    if normalized_model not in {"simplernn", "lstm", "gru"}:
        raise ValueError("model must be one of: simplernn, lstm, gru")

    texts, labels = load_dataset(dataset_name)
    cleaned_texts = clean_texts(texts)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        cleaned_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    effective_max_len = _effective_max_len(train_texts, max_len)
    effective_batch_size = _effective_batch_size(len(train_texts), batch_size)

    tokenizer = build_tokenizer(train_texts, max_vocab_size=MAX_VOCAB_SIZE)
    X_train = texts_to_padded_sequences(tokenizer, train_texts, max_len=effective_max_len)
    X_val = texts_to_padded_sequences(tokenizer, val_texts, max_len=effective_max_len)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_val = label_encoder.transform(val_labels)

    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
    model = _build_sequential_model(normalized_model, vocab_size, len(label_encoder.classes_), effective_max_len)

    patience = max(3, int(epochs * 0.1))
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    started_at = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=effective_batch_size,
        callbacks=[early_stopping],
        verbose=0,
    )
    elapsed_seconds = round(time.time() - started_at, 1)

    artifact_paths = save_artifacts(model, tokenizer, label_encoder, normalized_model, dataset_name, effective_max_len)

    history_plot_path = save_train_history_plot(
        history.history, f"{PLOTS_DIR}/train_history_{normalized_model}_{dataset_name}.png"
    )

    final_val_accuracy = round(float(history.history["val_accuracy"][-1]), 4)
    final_val_loss = round(float(history.history["val_loss"][-1]), 4)
    epochs_ran = len(history.history["loss"])

    return {
        "model": normalized_model,
        "dataset": dataset_name,
        "epochs_ran": epochs_ran,
        "epochs_requested": epochs,
        "elapsed_seconds": elapsed_seconds,
        "final_val_accuracy": final_val_accuracy,
        "final_val_loss": final_val_loss,
        "artifact_paths": artifact_paths,
        "history_plot_path": history_plot_path,
    }
