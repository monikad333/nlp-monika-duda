from __future__ import annotations

from lab04.config import SUPPORTED_LANGUAGES


def detect_language(text: str) -> str:
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException

    if not text or not text.strip():
        return "en"

    try:
        detected = detect(text)
    except LangDetectException:
        return "en"

    return detected if detected in SUPPORTED_LANGUAGES else detected
