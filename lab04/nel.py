from __future__ import annotations

from typing import Any

import requests

from lab04.config import NEL_CONFIDENCE_THRESHOLD, WIKIDATA_API_URL, WIKIPEDIA_API_URL_TEMPLATE

_REQUEST_HEADERS = {"User-Agent": "WSEI-NLP-Lab04-Bot/1.0 (educational project)"}


class NELError(ValueError):
    pass


def search_wikidata_candidates(entity_text: str, language: str = "en", limit: int = 5) -> list[dict[str, Any]]:
    params = {
        "action": "wbsearchentities",
        "search": entity_text,
        "language": language,
        "format": "json",
        "limit": limit,
    }

    response = requests.get(WIKIDATA_API_URL, params=params, headers=_REQUEST_HEADERS, timeout=10)
    response.raise_for_status()
    payload = response.json()

    candidates = []
    results = payload.get("search", [])
    total = max(len(results), 1)

    for rank, item in enumerate(results):
        confidence = round(max(0.05, 1.0 - rank / total), 3)
        candidates.append(
            {
                "id": item.get("id"),
                "label": item.get("label"),
                "description": item.get("description", ""),
                "confidence": confidence,
            }
        )

    return candidates


def get_wikipedia_summary(title: str, language: str = "en") -> dict[str, Any] | None:
    url = WIKIPEDIA_API_URL_TEMPLATE.format(language=language, title=title.replace(" ", "_"))

    try:
        response = requests.get(url, headers=_REQUEST_HEADERS, timeout=10)
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    payload = response.json()
    return {
        "title": payload.get("title"),
        "extract": payload.get("extract"),
        "url": payload.get("content_urls", {}).get("desktop", {}).get("page"),
    }


def link_entity(entity_text: str, language: str = "en") -> dict[str, Any]:
    if not entity_text or not entity_text.strip():
        raise NELError("Entity text cannot be empty.")

    candidates = search_wikidata_candidates(entity_text, language=language)

    for candidate in candidates:
        wikipedia_info = get_wikipedia_summary(candidate["label"] or entity_text, language=language)
        candidate["wikipedia_url"] = wikipedia_info["url"] if wikipedia_info else None

    return {"entity": entity_text, "candidates": candidates}


def _score_candidate_against_context(candidate: dict[str, Any], context: str) -> float:
    description = (candidate.get("description") or "").lower()
    context_lower = context.lower()

    description_tokens = set(description.split())
    context_tokens = set(context_lower.split())
    overlap = len(description_tokens & context_tokens)

    return candidate.get("confidence", 0.0) + overlap * 0.05


def disambiguate_entity(entity_text: str, context: str, language: str = "en") -> dict[str, Any]:
    if not entity_text or not entity_text.strip():
        raise NELError("Entity text cannot be empty.")
    if not context or not context.strip():
        raise NELError("Context text cannot be empty.")

    candidates = search_wikidata_candidates(entity_text, language=language)
    if not candidates:
        return {"entity": entity_text, "best_candidate": None, "candidates": []}

    scored_candidates = sorted(
        candidates, key=lambda candidate: _score_candidate_against_context(candidate, context), reverse=True
    )

    best_candidate = scored_candidates[0]
    filtered_candidates = [
        candidate for candidate in scored_candidates if candidate.get("confidence", 0) >= NEL_CONFIDENCE_THRESHOLD
    ]

    return {
        "entity": entity_text,
        "best_candidate": best_candidate,
        "candidates": filtered_candidates or scored_candidates[:1],
    }
