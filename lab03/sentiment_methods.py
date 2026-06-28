from __future__ import annotations

from typing import Any

import numpy as np

from common.nlp_task import tokenize_text
from lab03.config import MAX_LEN
from lab03.datasets import load_custom_dataset
from lab03.model_loader import list_available_models, load_artifacts
from lab03.preprocessing import clean_texts, texts_to_padded_sequences

_DIACRITICS_MAP = str.maketrans("ąćęłńóśźż", "acelnoszz")


def _strip_diacritics(text: str) -> str:
    return text.lower().translate(_DIACRITICS_MAP)


POSITIVE_WORDS = {
    "dobry", "swietny", "super", "wspanialy", "uwielbiam", "zadowolony", "polecam",
    "fantastyczny", "pomocna", "szybka", "wzruszajacy", "najlepszy", "milo", "pozytywnie",
}
NEGATIVE_WORDS = {
    "zly", "okropny", "fatalny", "najgorszy", "rozczarowal", "rozczarowany", "nie polecam",
    "uszkodzony", "niesmaczne", "zimne", "slaba", "zawiesza", "traci", "wadliwy", "okropnie",
}

_NB_RF_CACHE: dict[str, Any] = {}


class SentimentMethodError(ValueError):
    pass


def rule_based(text: str) -> dict[str, Any]:
    tokens = {_strip_diacritics(token) for token in tokenize_text(text)}
    positive_hits = len(tokens & POSITIVE_WORDS)
    negative_hits = len(tokens & NEGATIVE_WORDS)

    if positive_hits > negative_hits:
        label = "pozytywny"
    elif negative_hits > positive_hits:
        label = "negatywny"
    else:
        label = "neutralny"

    total = positive_hits + negative_hits
    score = 0.5 if total == 0 else max(positive_hits, negative_hits) / total
    return {"label": label, "score": round(score, 3), "model": "rule-based lexicon"}


def _get_nb_rf_pipeline(method: str):
    if method in _NB_RF_CACHE:
        return _NB_RF_CACHE[method]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    texts, labels = load_custom_dataset()
    if len(set(labels)) < 2:
        raise SentimentMethodError("Custom dataset needs at least two classes to train nb/rf.")

    classifier = MultinomialNB() if method == "nb" else RandomForestClassifier(n_estimators=200, random_state=42)
    pipeline = Pipeline([("vectorizer", TfidfVectorizer()), ("classifier", classifier)])
    pipeline.fit(texts, labels)

    _NB_RF_CACHE[method] = pipeline
    return pipeline


def ml_based(text: str, method: str) -> dict[str, Any]:
    pipeline = _get_nb_rf_pipeline(method)
    label = pipeline.predict([text])[0]

    score = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([text])[0]
        score = round(float(max(probabilities)), 3)

    model_name = "Multinomial Naive Bayes" if method == "nb" else "Random Forest"
    return {"label": label, "score": score, "model": model_name}


_TRANSFORMER_CACHE: Any = None


def transformer_based(text: str) -> dict[str, Any]:
    global _TRANSFORMER_CACHE

    if _TRANSFORMER_CACHE is None:
        from transformers import pipeline as hf_pipeline
        from lab03.config import TRANSFORMER_MODEL_NAME

        _TRANSFORMER_CACHE = hf_pipeline(
            "sentiment-analysis", model=TRANSFORMER_MODEL_NAME, framework="tf"
        )

    result = _TRANSFORMER_CACHE(text[:512])[0]
    stars = int(result["label"][0])

    if stars <= 2:
        label = "negatywny"
    elif stars == 3:
        label = "neutralny"
    else:
        label = "pozytywny"

    return {"label": label, "score": round(float(result["score"]), 3), "model": "transformer (multilingual BERT)"}


def textblob_based(text: str) -> dict[str, Any]:
    from textblob import TextBlob

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        label = "pozytywny"
    elif polarity < -0.1:
        label = "negatywny"
    else:
        label = "neutralny"

    return {"label": label, "score": round(float(polarity), 3), "model": "TextBlob (en-tuned, limited for Polish)"}


_STANZA_CACHE: Any = None


def stanza_based(text: str) -> dict[str, Any]:
    global _STANZA_CACHE

    if _STANZA_CACHE is None:
        import stanza

        try:
            stanza.download("en", processors="tokenize,sentiment", verbose=False)
        except Exception:
            pass
        _STANZA_CACHE = stanza.Pipeline(lang="en", processors="tokenize,sentiment", verbose=False)

    document = _STANZA_CACHE(text)
    sentiments = [sentence.sentiment for sentence in document.sentences]
    average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 1

    label = {0: "negatywny", 1: "neutralny", 2: "pozytywny"}.get(round(average_sentiment), "neutralny")
    return {"label": label, "score": round(float(average_sentiment) / 2, 3), "model": "Stanza (en sentiment model)"}


def sequential_based(text: str, method: str, dataset_name: str | None = None) -> dict[str, Any]:
    default_search_order = ["custom", "imdb", "amazon"]
    candidate_datasets = [dataset_name] if dataset_name else default_search_order

    last_error: Exception | None = None
    for candidate in candidate_datasets:
        try:
            model, tokenizer, label_encoder = load_artifacts(method, candidate)
            cleaned = clean_texts([text])
            padded = texts_to_padded_sequences(tokenizer, cleaned, max_len=MAX_LEN)
            probabilities = model.predict(padded, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            label = label_encoder.inverse_transform([predicted_index])[0]
            return {
                "label": label,
                "score": round(float(probabilities[predicted_index]), 3),
                "model": f"{method.upper()} (trained on {candidate})",
            }
        except FileNotFoundError as exc:
            last_error = exc
            continue

    available = list_available_models()
    raise SentimentMethodError(
        f"No trained '{method}' model found. Train one first with /train model={method} dataset=<amazon|imdb|custom>. "
        f"Available models: {available}"
    ) from last_error


def run_sentiment_method(method: str, text: str) -> dict[str, Any]:
    normalized = (method or "").strip().lower()

    if not text or not text.strip():
        raise SentimentMethodError("Text cannot be empty.")

    if normalized == "rule":
        return rule_based(text)
    if normalized in {"nb", "rf"}:
        return ml_based(text, normalized)
    if normalized == "transformer":
        return transformer_based(text)
    if normalized == "textblob":
        return textblob_based(text)
    if normalized == "stanza":
        return stanza_based(text)
    if normalized in {"simplernn", "lstm", "gru"}:
        return sequential_based(text, normalized)

    raise SentimentMethodError(
        "Unknown method. Allowed: rule, nb, rf, transformer, textblob, stanza, simplernn, lstm, gru"
    )
