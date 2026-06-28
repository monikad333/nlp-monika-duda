from __future__ import annotations

import os
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from common.nlp_task import clean_text, tokenize_text

EMBEDDING_NAMES = ["bow", "tfidf", "word2vec", "glove"]
GLOVE_MODEL_NAME = os.getenv("LAB2_GLOVE_MODEL", "glove-wiki-gigaword-50")
_GLOVE_CACHE: Any = None


def tokenize_for_embedding(text: str) -> list[str]:
    return tokenize_text(clean_text(text))


def _doc_vector(tokens: list[str], keyed_vectors: Any) -> np.ndarray:
    vectors = [keyed_vectors[token] for token in tokens if token in keyed_vectors]
    if not vectors:
        return np.zeros(keyed_vectors.vector_size, dtype="float32")
    return np.mean(vectors, axis=0)


def _load_glove() -> Any:
    global _GLOVE_CACHE
    if _GLOVE_CACHE is not None:
        return _GLOVE_CACHE

    import gensim.downloader as gensim_downloader

    _GLOVE_CACHE = gensim_downloader.load(GLOVE_MODEL_NAME)
    return _GLOVE_CACHE


class EmbeddingResult:
    def __init__(
        self,
        name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: list[str] | None,
        keyed_vectors: Any | None,
        is_dense: bool,
    ) -> None:
        self.name = name
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.keyed_vectors = keyed_vectors
        self.is_dense = is_dense


def build_embedding(
    name: str,
    train_texts: list[str],
    test_texts: list[str],
    seed: int = 42,
) -> EmbeddingResult:
    normalized = (name or "").strip().lower()

    if normalized == "bow":
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        return EmbeddingResult(
            normalized, X_train, X_test, list(vectorizer.get_feature_names_out()), None, is_dense=False
        )

    if normalized == "tfidf":
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        return EmbeddingResult(
            normalized, X_train, X_test, list(vectorizer.get_feature_names_out()), None, is_dense=False
        )

    if normalized == "word2vec":
        from gensim.models import Word2Vec

        train_tokens = [tokenize_for_embedding(text) for text in train_texts]
        test_tokens = [tokenize_for_embedding(text) for text in test_texts]

        model = Word2Vec(
            sentences=train_tokens,
            vector_size=100,
            window=5,
            min_count=1,
            seed=seed,
            workers=1,
        )

        X_train = np.array([_doc_vector(tokens, model.wv) for tokens in train_tokens])
        X_test = np.array([_doc_vector(tokens, model.wv) for tokens in test_tokens])
        return EmbeddingResult(normalized, X_train, X_test, None, model.wv, is_dense=True)

    if normalized == "glove":
        keyed_vectors = _load_glove()
        train_tokens = [tokenize_for_embedding(text) for text in train_texts]
        test_tokens = [tokenize_for_embedding(text) for text in test_texts]

        X_train = np.array([_doc_vector(tokens, keyed_vectors) for tokens in train_tokens])
        X_test = np.array([_doc_vector(tokens, keyed_vectors) for tokens in test_tokens])
        return EmbeddingResult(normalized, X_train, X_test, None, keyed_vectors, is_dense=True)

    raise ValueError(f"Unknown embedding '{name}'. Allowed: {EMBEDDING_NAMES}")


def similar_words(keyed_vectors: Any, words: list[str], topn: int = 5) -> dict[str, list[tuple[str, float]]]:
    results: dict[str, list[tuple[str, float]]] = {}
    for word in words:
        if word in keyed_vectors:
            results[word] = keyed_vectors.most_similar(word, topn=topn)
        else:
            results[word] = []
    return results
