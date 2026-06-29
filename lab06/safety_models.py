from __future__ import annotations

import json
import re
from typing import Any

from lab06.config import (
    BIELIK_GUARD_MODEL_NAME,
    GUARD_CATEGORIES,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT_SECONDS,
    PII_MODEL_NAME,
    QWEN_GUARD_MODEL,
)

_PII_PIPELINE: Any = None
_BIELIK_GUARD_PIPELINE: Any = None
_BIELIK_GUARD_UNAVAILABLE = False

MANDATORY_PII_TYPES = {"private_email", "private_phone", "private_credit_card", "private_ssn", "private_address"}

_REGEX_PATTERNS = {
    "private_email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "private_phone": re.compile(r"(\+?\d{1,3}[\s-]?)?(\d{2,3}[\s-]?){2,4}\d{2,4}"),
    "private_credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}

_POLISH_TOXIC_LEXICON = {
    "hate_speech": {"glupszy", "idiot", "debil", "kretyn", "wszyscy zlodzieje", "zlodzieje", "bezwartosciowy"},
    "self_harm": {
        "zabij sie", "powinienes sie zabic", "samobojstwo", "zniknij z tego swiata",
        "zniknac z tego swiata", "nie zasluzysz zyc", "nikomu nie potrzebny", "lepiej by cie nie bylo",
    },
    "violence": {"zabije", "pobije", "zniszcze cie", "znajde cie i pobije", "rozjebie", "skrzywdze cie"},
    "sexual": {"erotyczny", "porno"},
    "spam": {"kliknij tutaj", "darmowe pieniadze", "zarabiaj z domu", "promocja link"},
}


def _strip_diacritics(text: str) -> str:
    mapping = str.maketrans("ąćęłńóśźż", "acelnoszz")
    return text.lower().translate(mapping)


def detect_private_info(text: str) -> dict[str, Any]:
    """Detect personally identifiable information (PII) using openai/privacy-filter, with a regex safety net."""
    global _PII_PIPELINE

    entities: list[dict[str, Any]] = []
    regex_spans: list[tuple[int, int]] = []

    for label, pattern in _REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group().strip()
            if len(value) < 6:
                continue
            entities.append({"type": label, "text": value, "score": 1.0, "source": "regex"})
            regex_spans.append((match.start(), match.end()))

    if _PII_PIPELINE is None:
        from transformers import pipeline as hf_pipeline

        _PII_PIPELINE = hf_pipeline("token-classification", model=PII_MODEL_NAME, aggregation_strategy="simple")

    for item in _PII_PIPELINE(text):
        label = item["entity_group"]
        word = item["word"].strip()
        if not label.startswith("private_") or len(word) <= 2:
            continue

        covered_by_regex = any(
            item["start"] < end and item["end"] > start for start, end in regex_spans
        )
        if covered_by_regex:
            continue

        entities.append({"type": label, "text": word, "score": round(float(item["score"]), 3), "source": "model"})

    actionable_entities = [entity for entity in entities if entity["type"] in MANDATORY_PII_TYPES]

    return {"has_pii": bool(actionable_entities), "entities": entities}


def _combined_lexicon() -> dict[str, set[str]]:
    from lab06.feedback import get_learned_keywords

    combined = {category: set(words) for category, words in _POLISH_TOXIC_LEXICON.items()}
    for category, words in get_learned_keywords().items():
        combined.setdefault(category, set()).update(words)
    return combined


def _classify_bielik_guard_lexicon(text: str) -> dict[str, Any]:
    normalized = _strip_diacritics(text)
    lexicon = _combined_lexicon()

    matched_categories = [
        category for category, words in lexicon.items() if any(word in normalized for word in words)
    ]

    if not matched_categories:
        return {"label": "clean", "score": 0.9, "severity": "low"}

    label = "+".join(matched_categories)
    severity = "critical" if "self_harm" in matched_categories or "violence" in matched_categories else "high"
    return {"label": label, "score": 0.85, "severity": severity}


def classify_bielik_guard(text: str) -> dict[str, Any]:
    """Classify text using Bielik Guard. Falls back to a lexicon heuristic if the gated model is unavailable."""
    global _BIELIK_GUARD_PIPELINE, _BIELIK_GUARD_UNAVAILABLE

    if _BIELIK_GUARD_UNAVAILABLE:
        return _classify_bielik_guard_lexicon(text)

    if _BIELIK_GUARD_PIPELINE is None:
        try:
            from transformers import pipeline as hf_pipeline

            _BIELIK_GUARD_PIPELINE = hf_pipeline("text-classification", model=BIELIK_GUARD_MODEL_NAME)
        except Exception:
            _BIELIK_GUARD_UNAVAILABLE = True
            return _classify_bielik_guard_lexicon(text)

    result = _BIELIK_GUARD_PIPELINE(text)[0]
    label = result["label"]
    score = round(float(result["score"]), 3)

    if label == "clean" or score < 0.5:
        severity = "low"
    elif score < 0.7:
        severity = "medium"
    elif score < 0.9:
        severity = "high"
    else:
        severity = "critical"

    return {"label": label, "score": score, "severity": severity}


_QWEN_GUARD_PROMPT = """You are a content safety classifier. Analyze the text and respond with ONLY a JSON object \
(no extra text) in this exact format:
{{"risk_level": "safe|low|medium|high|critical", "categories": ["..."], "confidence": 0.0, "recommended_action": "approve|review|reject"}}

Text: {text}
JSON:"""


def _translate_to_english_for_classification(text: str) -> str:
    """Small Qwen guard models judge safety far more reliably in English than Polish; translate first."""
    from lab04.language import detect_language
    from lab04.translation import translate_text

    try:
        if detect_language(text) == "en":
            return text
        return translate_text(text, target_lang="en")["translation"]
    except Exception:
        return text


def classify_qwen_guard(text: str) -> dict[str, Any]:
    """Classify text using a small Qwen model in Ollama, prompted to act as a safety guard."""
    import requests

    classification_text = _translate_to_english_for_classification(text)

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": QWEN_GUARD_MODEL,
                "prompt": _QWEN_GUARD_PROMPT.format(text=classification_text),
                "stream": False,
                "format": "json",
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        raw_output = response.json().get("response", "")
        parsed = json.loads(raw_output)
    except Exception:
        return {"risk_level": "medium", "categories": ["unparseable_response"], "confidence": 0.3, "recommended_action": "review"}

    return {
        "risk_level": parsed.get("risk_level", "medium"),
        "categories": parsed.get("categories", []),
        "confidence": round(float(parsed.get("confidence", 0.5)), 3),
        "recommended_action": parsed.get("recommended_action", "review"),
    }
