from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

from lab06.config import FEEDBACK_RETRAIN_THRESHOLD
from lab06.storage import load_feedback_log, load_moderation_log
from lab06.tools import add_feedback

LEARNED_KEYWORDS_PATH = os.getenv("LAB6_LEARNED_KEYWORDS_PATH", "lab06/learned_keywords.json")
_STOPWORDS = {"to", "jest", "byl", "byla", "sie", "nie", "tak", "ale", "oraz", "lub", "ten", "ta", "i", "w", "z", "na"}


def _strip_diacritics(text: str) -> str:
    mapping = str.maketrans("ąćęłńóśźż", "acelnoszz")
    return text.lower().translate(mapping)


def submit_feedback(content_id: str, moderator_decision: str, category: str = "unspecified") -> dict[str, Any]:
    """Record a moderator override against the bot's original decision for the given content_id."""
    log_entry = next((row for row in load_moderation_log() if row["content_id"] == content_id), None)
    if not log_entry:
        raise ValueError(f"No moderation log entry found for content_id '{content_id}'.")

    add_feedback(
        content_id=content_id,
        original_decision=log_entry["action"],
        moderator_override=moderator_decision,
        text_sample=log_entry["text"],
        category=category,
    )

    feedback_count = len(load_feedback_log())
    retrained = False
    if feedback_count > 0 and feedback_count % FEEDBACK_RETRAIN_THRESHOLD == 0:
        train_on_feedback()
        retrained = True

    return {
        "content_id": content_id,
        "original_decision": log_entry["action"],
        "moderator_override": moderator_decision,
        "feedback_count": feedback_count,
        "retrained": retrained,
    }


def _load_learned_keywords() -> dict[str, list[str]]:
    if not os.path.exists(LEARNED_KEYWORDS_PATH):
        return {}
    with open(LEARNED_KEYWORDS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_learned_keywords() -> dict[str, list[str]]:
    return _load_learned_keywords()


def train_on_feedback() -> dict[str, Any]:
    """Mine false negatives (bot APPROVE later overridden to REJECT) for new toxic keywords.

    This is a lightweight stand-in for fine-tuning a real model: it extends the lexicon-based
    Bielik Guard fallback with words that previously slipped through, closing the feedback loop
    without requiring GPU fine-tuning infrastructure that is out of scope for this lab.
    """
    feedback_rows = load_feedback_log()
    word_counts_by_category: dict[str, Counter] = {}

    for row in feedback_rows:
        bot_was_too_lenient = row["original_bot_decision"] != "REJECT" and row["moderator_override"] == "REJECT"
        if not bot_was_too_lenient:
            continue

        category = row.get("category") or "toxic"
        tokens = [token for token in _strip_diacritics(row["text_sample"]).split() if token not in _STOPWORDS and len(token) > 3]

        word_counts_by_category.setdefault(category, Counter()).update(tokens)

    learned = _load_learned_keywords()
    for category, counter in word_counts_by_category.items():
        top_words = [word for word, count in counter.most_common(10) if count >= 1]
        learned.setdefault(category, [])
        learned[category] = sorted(set(learned[category]) | set(top_words))

    os.makedirs(os.path.dirname(LEARNED_KEYWORDS_PATH) or ".", exist_ok=True)
    with open(LEARNED_KEYWORDS_PATH, "w", encoding="utf-8") as file:
        json.dump(learned, file, ensure_ascii=False, indent=2)

    return {"categories_updated": list(word_counts_by_category.keys()), "learned_keywords": learned}
