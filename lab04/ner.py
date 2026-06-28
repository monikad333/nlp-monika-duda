from __future__ import annotations

from typing import Any

from lab04.config import SPACY_MODELS

_SPACY_CACHE: dict[str, Any] = {}
_STANZA_CACHE: dict[str, Any] = {}


class NERError(ValueError):
    pass


def detect_spacy_language(text: str) -> str:
    from lab04.language import detect_language

    language = detect_language(text)
    return language if language in SPACY_MODELS else "en"


def _get_spacy_pipeline(language: str):
    if language not in SPACY_MODELS:
        raise NERError(f"Spacy model not configured for language '{language}'. Allowed: {list(SPACY_MODELS)}")

    if language in _SPACY_CACHE:
        return _SPACY_CACHE[language]

    import spacy

    pipeline = spacy.load(SPACY_MODELS[language])
    _SPACY_CACHE[language] = pipeline
    return pipeline


def run_spacy_ner(text: str, language: str | None = None) -> list[dict[str, Any]]:
    resolved_language = language or detect_spacy_language(text)
    pipeline = _get_spacy_pipeline(resolved_language)
    document = pipeline(text)

    return [
        {"text": entity.text, "label": entity.label_, "start": entity.start_char, "end": entity.end_char}
        for entity in document.ents
    ]


def _get_stanza_pipeline(language: str):
    if language in _STANZA_CACHE:
        return _STANZA_CACHE[language]

    import stanza

    try:
        stanza.download(language, processors="tokenize,ner", verbose=False)
    except Exception:
        pass

    pipeline = stanza.Pipeline(lang=language, processors="tokenize,ner", verbose=False)
    _STANZA_CACHE[language] = pipeline
    return pipeline


def run_stanza_ner(text: str, language: str | None = None) -> list[dict[str, Any]]:
    from lab04.language import detect_language

    resolved_language = language or detect_language(text)
    pipeline = _get_stanza_pipeline(resolved_language)
    document = pipeline(text)

    return [
        {"text": entity.text, "label": entity.type, "start": entity.start_char, "end": entity.end_char}
        for entity in document.ents
    ]


def run_ner(method: str, text: str, language: str | None = None) -> list[dict[str, Any]]:
    if not text or not text.strip():
        raise NERError("Text cannot be empty.")

    normalized_method = (method or "").strip().lower()
    if normalized_method == "spacy":
        return run_spacy_ner(text, language)
    if normalized_method == "stanza":
        return run_stanza_ner(text, language)

    raise NERError("Unknown NER method. Allowed: spacy, stanza")


def format_entities(entities: list[dict[str, Any]]) -> str:
    if not entities:
        return "No entities found."
    return "\n".join(f"- {entity['text']} ({entity['label']}) [{entity['start']}:{entity['end']}]" for entity in entities)
