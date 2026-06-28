from __future__ import annotations

from typing import Any

from lab04.config import SUPPORTED_LANGUAGES, TRANSLATION_MODEL_OVERRIDES, TRANSLATION_MODEL_TEMPLATE

_TRANSLATOR_CACHE: dict[str, Any] = {}


class TranslationError(ValueError):
    pass


def _get_translator(source_lang: str, target_lang: str) -> tuple[Any, Any, str]:
    cache_key = f"{source_lang}-{target_lang}"
    if cache_key in _TRANSLATOR_CACHE:
        return _TRANSLATOR_CACHE[cache_key]

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    override = TRANSLATION_MODEL_OVERRIDES.get((source_lang, target_lang))
    model_name = override["model"] if override else TRANSLATION_MODEL_TEMPLATE.format(source=source_lang, target=target_lang)
    target_prefix = override["target_prefix"] if override else ""

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as exc:
        raise TranslationError(
            f"Translation model '{model_name}' is not available for {source_lang} -> {target_lang}."
        ) from exc

    _TRANSLATOR_CACHE[cache_key] = (tokenizer, model, target_prefix)
    return tokenizer, model, target_prefix


def translate_text(text: str, target_lang: str, source_lang: str | None = None) -> dict[str, Any]:
    from lab04.language import detect_language

    if not text or not text.strip():
        raise TranslationError("Text cannot be empty.")

    target_lang = (target_lang or "").strip().lower()
    if target_lang not in SUPPORTED_LANGUAGES:
        raise TranslationError(f"Unsupported target_lang '{target_lang}'. Allowed: {SUPPORTED_LANGUAGES}")

    resolved_source = source_lang or detect_language(text)
    if resolved_source not in SUPPORTED_LANGUAGES:
        resolved_source = "en"

    if resolved_source == target_lang:
        raise TranslationError("Source and target languages are the same.")

    tokenizer, model, target_prefix = _get_translator(resolved_source, target_lang)
    inputs = tokenizer(f"{target_prefix}{text[:512]}", return_tensors="pt")
    generated = model.generate(**inputs, max_new_tokens=256)
    translation = tokenizer.decode(generated[0], skip_special_tokens=True)

    return {"source": resolved_source, "target": target_lang, "translation": translation}
