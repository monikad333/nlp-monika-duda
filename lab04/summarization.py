from __future__ import annotations

import os
import time
from typing import Any

import requests

from lab04.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    SUMMARY_LENGTHS,
    SUMMARY_TYPES,
    SUMMARIES_DIR,
)


class SummarizationError(ValueError):
    pass


def _build_prompt(text: str, summary_type: str, length: str) -> str:
    sentence_count = SUMMARY_LENGTHS.get(length, SUMMARY_LENGTHS["medium"])

    if summary_type == "extractive":
        instruction = (
            f"Extract the {sentence_count} most important sentences from the text below, verbatim, as a list."
        )
    elif summary_type == "bullets":
        instruction = f"Summarize the text below as {sentence_count} concise bullet points."
    else:
        instruction = f"Write an abstractive summary of the text below in about {sentence_count} sentences."

    return f"{instruction}\n\nText:\n{text}\n\nSummary:"


def generate_summary(
    text: str,
    summary_type: str = "abstractive",
    length: str = "medium",
    custom_prompt: str | None = None,
    model: str = OLLAMA_MODEL,
) -> dict[str, Any]:
    if not text or not text.strip():
        raise SummarizationError("Text cannot be empty.")

    normalized_type = (summary_type or "").strip().lower()
    if normalized_type not in SUMMARY_TYPES:
        raise SummarizationError(f"Unknown summary_type '{summary_type}'. Allowed: {SUMMARY_TYPES}")

    normalized_length = (length or "").strip().lower()
    if normalized_length not in SUMMARY_LENGTHS:
        raise SummarizationError(f"Unknown length '{length}'. Allowed: {list(SUMMARY_LENGTHS)}")

    prompt = custom_prompt or _build_prompt(text, normalized_type, normalized_length)

    started_at = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SummarizationError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Is it running ('ollama serve') and is model "
            f"'{model}' pulled ('ollama pull {model}')? Original error: {exc}"
        ) from exc

    elapsed_seconds = round(time.time() - started_at, 2)
    payload = response.json()
    summary_text = payload.get("response", "").strip()

    saved_path = _save_summary(summary_text, normalized_type, normalized_length)

    return {
        "model": model,
        "text_length_tokens": len(text.split()),
        "summary_type": normalized_type,
        "summary_length": normalized_length,
        "summary": summary_text,
        "generation_time_seconds": elapsed_seconds,
        "saved_path": saved_path,
    }


def _save_summary(summary_text: str, summary_type: str, length: str) -> str:
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(SUMMARIES_DIR, f"summary_{summary_type}_{length}_{timestamp}.txt")

    with open(path, "w", encoding="utf-8") as file:
        file.write(summary_text)

    return path
