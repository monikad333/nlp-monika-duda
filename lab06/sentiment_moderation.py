from __future__ import annotations

from typing import Any

ANGER_WORDS = {"glupszy", "idiot", "debil", "kretyn", "zlodzieje", "wszyscy", "nienawidze", "wsciekly"}
SADNESS_WORDS = {"rozczarowany", "smutny", "przykro", "zal"}
FEAR_WORDS = {"strach", "boje", "przerazony"}
JOY_WORDS = {"uwielbiam", "super", "swietny", "najlepszy", "zadowolony", "szczesliwy"}
SURPRISE_WORDS = {"niesamowite", "zdumiewajace", "wow"}


def _strip_diacritics(text: str) -> str:
    mapping = str.maketrans("ąćęłńóśźż", "acelnoszz")
    return text.lower().translate(mapping)


def _detect_emotion(normalized_text: str) -> str:
    tokens = set(normalized_text.split())

    emotion_word_sets = [
        ("anger", ANGER_WORDS),
        ("sadness", SADNESS_WORDS),
        ("fear", FEAR_WORDS),
        ("joy", JOY_WORDS),
        ("surprise", SURPRISE_WORDS),
    ]

    for emotion, words in emotion_word_sets:
        if tokens & words:
            return emotion

    return "neutral"


def _detect_sarcasm(normalized_text: str) -> bool:
    sarcasm_markers = ["super... nie", "no super", "jasne, jasne", "/s", "wow, super"]
    return any(marker in normalized_text for marker in sarcasm_markers)


def analyze_sentiment_for_moderation(text: str) -> dict[str, Any]:
    """Analyze sentiment to provide context for moderation (not a moderation verdict by itself)."""
    from lab03.sentiment_methods import rule_based

    rule_result = rule_based(text)
    label_map = {"pozytywny": "positive", "neutralny": "neutral", "negatywny": "negative"}
    sentiment = label_map.get(rule_result["label"], "neutral")

    normalized = _strip_diacritics(text)

    return {
        "sentiment": sentiment,
        "confidence": rule_result["score"],
        "emotion": _detect_emotion(normalized),
        "sarcasm_detected": _detect_sarcasm(normalized),
    }
