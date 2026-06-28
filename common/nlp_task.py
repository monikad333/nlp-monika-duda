from __future__ import annotations

import os
import re
from collections import Counter
from datetime import datetime, timedelta
from statistics import mean
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

_FALLBACK_STOPWORDS = {
    "a",
    "ale",
    "bo",
    "byl",
    "byla",
    "bylo",
    "czy",
    "dla",
    "do",
    "i",
    "jak",
    "jest",
    "na",
    "nie",
    "o",
    "oraz",
    "po",
    "sie",
    "to",
    "w",
    "z",
    "za",
    "ze",
}

_POLISH_NON_WORDS_PATTERN = r"[^\w\sąćęłńóśźżĄĆĘŁŃÓŚŹŻ]"
_NLTK_READY = False
_POLISH_STOPWORDS_REPO_PRIMARY_URL = (
    "https://raw.githubusercontent.com/stopwords-iso/stopwords-pl/master/raw/"
    "gh-stopwords-json-pl.txt"
)
_POLISH_STOPWORDS_REPO_FALLBACK_URL = (
    "https://raw.githubusercontent.com/stopwords-iso/stopwords-pl/master/stopwords-pl.txt"
)
_POLISH_STOPWORDS_REPO_CACHE: set[str] | None = None


def ensure_nltk_resources() -> None:
    """Download missing NLTK resources if possible."""
    global _NLTK_READY
    if _NLTK_READY:
        return

    for lookup_key, resource_name in _NLTK_RESOURCES:
        try:
            nltk.data.find(lookup_key)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                # Keep graceful fallback behavior if downloads are not possible.
                continue

    _NLTK_READY = True


def _parse_stopwords_text(raw_text: str) -> set[str]:
    cleaned = (raw_text or "").strip()
    if not cleaned:
        return set()

    words = [token.strip().lower() for token in re.split(r"\s+", cleaned) if token.strip()]
    return {word for word in words if word}


def _load_polish_stopwords_from_repo() -> set[str]:
    global _POLISH_STOPWORDS_REPO_CACHE
    if _POLISH_STOPWORDS_REPO_CACHE is not None:
        return _POLISH_STOPWORDS_REPO_CACHE

    source_url = os.getenv("POLISH_STOPWORDS_URL", _POLISH_STOPWORDS_REPO_PRIMARY_URL)
    urls_to_try = [source_url]
    if source_url != _POLISH_STOPWORDS_REPO_FALLBACK_URL:
        urls_to_try.append(_POLISH_STOPWORDS_REPO_FALLBACK_URL)

    for url in urls_to_try:
        try:
            with urlopen(url, timeout=8) as response:
                payload = response.read().decode("utf-8", errors="replace")
                parsed = _parse_stopwords_text(payload)
                if parsed:
                    _POLISH_STOPWORDS_REPO_CACHE = parsed
                    return parsed
        except (URLError, TimeoutError, ValueError):
            continue

    _POLISH_STOPWORDS_REPO_CACHE = set()
    return _POLISH_STOPWORDS_REPO_CACHE


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    ensure_nltk_resources()
    try:
        sentences = [sentence.strip() for sentence in sent_tokenize(text, language="polish") if sentence.strip()]
    except LookupError:
        sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]

    return sentences if sentences else [text]


def clean_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(_POLISH_NON_WORDS_PATTERN, " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_text(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    ensure_nltk_resources()
    try:
        tokens = word_tokenize(text, language="polish")
    except LookupError:
        tokens = re.findall(r"\b[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+\b", text)

    return [token for token in tokens if token.strip()]


def get_stopwords(language: str = "polish") -> set[str]:
    if language == "polish":
        repo_stopwords = _load_polish_stopwords_from_repo()
        if repo_stopwords:
            return repo_stopwords

    ensure_nltk_resources()
    try:
        return set(stopwords.words(language))
    except LookupError:
        return set(_FALLBACK_STOPWORDS)


def remove_stopwords(tokens: list[str], language: str = "polish") -> list[str]:
    stop_words = get_stopwords(language)
    return [token for token in tokens if token.lower() not in stop_words]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    lemmatized: list[str] = []

    for token in tokens:
        try:
            lemmatized.append(lemmatizer.lemmatize(token))
        except LookupError:
            return tokens

    return lemmatized


def stem_tokens(tokens: list[str], language: str = "polish") -> list[str]:
    try:
        stemmer = SnowballStemmer(language)
    except ValueError:
        return tokens

    return [stemmer.stem(token) for token in tokens]


def generate_ngrams(tokens: list[str], n_value: int) -> list[tuple[str, ...]]:
    if n_value <= 0 or len(tokens) < n_value:
        return []
    return list(ngrams(tokens, n_value))


def text_statistics(tokens: list[str]) -> dict[str, float | int]:
    lengths = [len(token) for token in tokens]
    if not tokens:
        return {
            "token_count": 0,
            "unique_token_count": 0,
            "avg_token_length": 0.0,
            "min_token_length": 0,
            "max_token_length": 0,
        }

    return {
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
        "avg_token_length": round(mean(lengths), 3),
        "min_token_length": min(lengths),
        "max_token_length": max(lengths),
    }


def bag_of_words(tokens: list[str]) -> dict[str, int]:
    if not tokens:
        return {}

    text_value = " ".join(tokens)
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform([text_value])
    counts = matrix.toarray()[0]

    return {
        token: int(count)
        for token, count in zip(vectorizer.get_feature_names_out(), counts)
        if count > 0
    }


def tfidf_representation(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}

    text_value = " ".join(tokens)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([text_value])
    values = matrix.toarray()[0]

    return {
        token: round(float(value), 6)
        for token, value in zip(vectorizer.get_feature_names_out(), values)
        if value > 0
    }


def preprocess_for_model(text: str) -> str:
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    without_stopwords = remove_stopwords(tokens)
    lemmatized = lemmatize_tokens(without_stopwords)
    stemmed = stem_tokens(lemmatized)
    return " ".join(stemmed).strip()


def _next_plot_path(plots_dir: str = "plots") -> str:
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now()

    while True:
        filename = timestamp.strftime("Sentence_%Y-%m-%d_%H-%M-%S.png")
        plot_path = os.path.join(plots_dir, filename)
        if not os.path.exists(plot_path):
            return plot_path
        timestamp += timedelta(seconds=1)


def _save_figure(fig: Any, plots_dir: str = "plots") -> str:
    path = _next_plot_path(plots_dir)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_top_tokens(tokens: list[str], plots_dir: str = "plots", top_n: int = 10) -> str:
    counts = Counter(tokens)
    top_tokens = counts.most_common(top_n)

    fig, axis = plt.subplots(figsize=(10, 4))
    if top_tokens:
        labels, values = zip(*top_tokens)
        axis.bar(labels, values, color="#4e79a7")
        axis.set_ylabel("Frequency")
        axis.tick_params(axis="x", rotation=35)
    else:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.set_axis_off()

    axis.set_title("Most frequent tokens")
    return _save_figure(fig, plots_dir)


def plot_token_length_histogram(tokens: list[str], plots_dir: str = "plots") -> str:
    lengths = [len(token) for token in tokens]

    fig, axis = plt.subplots(figsize=(10, 4))
    if lengths:
        bins = range(1, max(lengths) + 2)
        axis.hist(lengths, bins=bins, color="#f28e2b", edgecolor="black")
        axis.set_xlabel("Token length")
        axis.set_ylabel("Count")
    else:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.set_axis_off()

    axis.set_title("Token length histogram")
    return _save_figure(fig, plots_dir)


def plot_wordcloud(tokens: list[str], plots_dir: str = "plots") -> str:
    token_text = " ".join(tokens).strip()
    if not token_text:
        token_text = "no_data"

    cloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        collocations=False,
    ).generate(token_text)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.imshow(cloud, interpolation="bilinear")
    axis.axis("off")
    axis.set_title("Word cloud")
    return _save_figure(fig, plots_dir)


def plot_class_distribution(class_counts: Counter[str], plots_dir: str = "plots") -> str:
    fig, axis = plt.subplots(figsize=(7, 4))
    if class_counts:
        labels = list(class_counts.keys())
        values = list(class_counts.values())
        axis.bar(labels, values, color="#59a14f")
        axis.set_xlabel("Class")
        axis.set_ylabel("Count")
    else:
        axis.text(0.5, 0.5, "No class data", ha="center", va="center")
        axis.set_axis_off()

    axis.set_title("Class distribution")
    return _save_figure(fig, plots_dir)


def run_single_task(task_name: str, text: str, plots_dir: str = "plots") -> dict[str, Any]:
    normalized_task = (task_name or "").strip().lower()
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    without_stopwords = remove_stopwords(tokens)
    lemmatized = lemmatize_tokens(without_stopwords)
    stemmed = stem_tokens(lemmatized)

    result: dict[str, Any] = {
        "task": normalized_task,
        "cleaned_text": cleaned,
        "tokens": tokens,
        "plots": [],
    }

    if normalized_task == "tokenize":
        result["result"] = tokens
    elif normalized_task == "remove_stopwords":
        result["result"] = without_stopwords
    elif normalized_task == "lemmatize":
        result["result"] = lemmatized
    elif normalized_task == "stemming":
        result["result"] = stemmed
    elif normalized_task == "stats":
        result["result"] = text_statistics(stemmed)
    elif normalized_task in {"n-grams", "ngrams"}:
        result["result"] = {
            "2-grams": [" ".join(item) for item in generate_ngrams(stemmed, 2)],
            "3-grams": [" ".join(item) for item in generate_ngrams(stemmed, 3)],
        }
    elif normalized_task == "plot_histogram":
        result["result"] = "Histogram generated."
        result["plots"].append(plot_token_length_histogram(stemmed, plots_dir))
    elif normalized_task == "plot_wordcloud":
        result["result"] = "Word cloud generated."
        result["plots"].append(plot_wordcloud(stemmed, plots_dir))
    elif normalized_task in {"plot_barchart", "plot_bar", "plot_top_tokens"}:
        result["result"] = "Bar chart generated."
        result["plots"].append(plot_top_tokens(stemmed, plots_dir))
    else:
        raise ValueError(
            "Unknown task. Allowed: tokenize, remove_stopwords, lemmatize, stemming, "
            "stats, n-grams, plot_histogram, plot_wordcloud, plot_barchart"
        )

    return result


def run_full_pipeline(text: str, plots_dir: str = "plots") -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    without_stopwords = remove_stopwords(tokens)
    lemmatized = lemmatize_tokens(without_stopwords)
    stemmed = stem_tokens(lemmatized)

    bow = bag_of_words(stemmed)
    tfidf = tfidf_representation(stemmed)
    stats = text_statistics(stemmed)

    plots = [
        plot_top_tokens(stemmed, plots_dir),
        plot_token_length_histogram(stemmed, plots_dir),
        plot_wordcloud(stemmed, plots_dir),
    ]

    return {
        "cleaned_text": cleaned,
        "tokens": tokens,
        "tokens_without_stopwords": without_stopwords,
        "lemmatized_tokens": lemmatized,
        "stemmed_tokens": stemmed,
        "bag_of_words": bow,
        "tfidf": tfidf,
        "stats": stats,
        "plots": plots,
    }


def run_corpus_stats(records: list[dict[str, str]], plots_dir: str = "plots") -> dict[str, Any]:
    all_tokens: list[str] = []
    class_counts: Counter[str] = Counter()

    for record in records:
        text = (record.get("text") or "").strip()
        class_name = (record.get("class") or "").strip().lower()
        if not text:
            continue

        cleaned = clean_text(text)
        tokens = tokenize_text(cleaned)
        tokens = remove_stopwords(tokens)
        tokens = stem_tokens(lemmatize_tokens(tokens))

        all_tokens.extend(tokens)
        if class_name:
            class_counts[class_name] += 1

    unique_tokens = sorted(set(all_tokens))
    unique_bigrams = sorted({" ".join(item) for item in generate_ngrams(all_tokens, 2)})
    unique_trigrams = sorted({" ".join(item) for item in generate_ngrams(all_tokens, 3)})

    plots = [
        plot_top_tokens(all_tokens, plots_dir),
        plot_token_length_histogram(all_tokens, plots_dir),
        plot_wordcloud(all_tokens, plots_dir),
        plot_class_distribution(class_counts, plots_dir),
    ]

    return {
        "token_count": len(all_tokens),
        "unique_tokens": unique_tokens,
        "unique_bigrams": unique_bigrams,
        "unique_trigrams": unique_trigrams,
        "top_tokens": Counter(all_tokens).most_common(20),
        "class_counts": dict(class_counts),
        "plots": plots,
    }
