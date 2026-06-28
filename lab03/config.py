from __future__ import annotations

import os

VALID_LABELS = ["pozytywny", "neutralny", "negatywny"]

CUSTOM_DATASET_PATH = os.getenv("LAB3_CUSTOM_DATASET_PATH", "lab03/sentiment_dataset.csv")
AMAZON_DATASET_PATH = os.getenv("LAB3_AMAZON_DATASET_PATH", "datasets/amazon_sentiment.csv")
MODELS_DIR = os.getenv("LAB3_MODELS_DIR", "lab03/models")
PLOTS_DIR = os.getenv("LAB3_PLOTS_DIR", "lab03/lab3plots")
RESULTS_PATH = os.getenv("LAB3_RESULTS_PATH", "lab03/lab3results.csv")

SEQUENTIAL_MODELS = ["simplernn", "lstm", "gru"]
DATASET_NAMES = ["amazon", "imdb", "custom"]

SENTIMENT_METHODS = ["rule", "nb", "rf", "transformer", "textblob", "stanza", "simplernn", "lstm", "gru"]

EMBEDDING_DIM = int(os.getenv("LAB3_EMBEDDING_DIM", "100"))
MAX_LEN = int(os.getenv("LAB3_MAX_LEN", "200"))
BATCH_SIZE = int(os.getenv("LAB3_BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("LAB3_EPOCHS", "10"))
MAX_VOCAB_SIZE = int(os.getenv("LAB3_MAX_VOCAB_SIZE", "10000"))
MAX_SAMPLES_PER_CLASS = int(os.getenv("LAB3_MAX_SAMPLES_PER_CLASS", "300"))

TRANSFORMER_MODEL_NAME = os.getenv(
    "LAB3_TRANSFORMER_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment"
)
