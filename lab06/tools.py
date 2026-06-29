from __future__ import annotations

import time
from typing import Any

from lab06.config import REPEAT_OFFENDER_THRESHOLD
from lab06.storage import (
    add_to_watchlist_storage,
    append_feedback_log,
    get_user_history_record,
    load_moderation_log,
    upsert_user_history,
)


class ModerationToolError(ValueError):
    pass


def approve_content(content_id: str, moderator_id: str = "bot") -> str:
    """Approve and publish flagged content."""
    return f"Content {content_id} approved by {moderator_id}."


def reject_content(content_id: str, reason: str, moderator_id: str = "bot") -> str:
    """Reject and remove content."""
    return f"Content {content_id} rejected by {moderator_id}. Reason: {reason}."


def flag_for_human_review(content_id: str, priority: str, reason: str) -> str:
    """Flag content for manual review by a human moderator. priority: low|medium|high|critical."""
    return f"Content {content_id} flagged for human review (priority={priority}). Reason: {reason}."


def shadow_ban_user(user_id: str, duration_hours: int, reason: str) -> str:
    """Limit user visibility (shadow ban) for a defined period."""
    upsert_user_history(user_id, username=user_id, violation_categories=None, shadow_banned=True)
    return f"User {user_id} shadow-banned for {duration_hours}h. Reason: {reason}."


def get_user_moderation_history(user_id: str) -> dict[str, Any]:
    """Return the user's moderation record."""
    record = get_user_history_record(user_id)
    if not record:
        return {
            "violations_count": 0,
            "last_violation": None,
            "categories": [],
            "risk_score": 0.0,
            "is_repeat_offender": False,
        }

    return {
        "violations_count": int(record["total_violations"]),
        "last_violation": record["last_violation_date"] or None,
        "categories": record["categories"].split(";") if record["categories"] else [],
        "risk_score": float(record["risk_score"]),
        "is_repeat_offender": record["is_repeat_offender"] == "True",
    }


def find_similar_violations(text: str, limit: int = 5) -> list[dict[str, Any]]:
    """Find similar previously moderated cases (simple keyword overlap over moderation_log.csv)."""
    query_tokens = set(text.lower().split())
    scored_cases = []

    for entry in load_moderation_log():
        entry_tokens = set(entry.get("text", "").lower().split())
        overlap = len(query_tokens & entry_tokens)
        if overlap > 0:
            scored_cases.append((overlap, entry))

    scored_cases.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in scored_cases[:limit]]


def add_to_watchlist(user_id: str, reason: str) -> str:
    """Add a user to the watchlist for increased monitoring."""
    add_to_watchlist_storage(user_id, reason)
    return f"User {user_id} added to watchlist. Reason: {reason}."


def add_feedback(content_id: str, original_decision: str, moderator_override: str, text_sample: str, category: str) -> str:
    """Record a human moderator override for the feedback loop."""
    append_feedback_log(
        {
            "content_id": content_id,
            "original_bot_decision": original_decision,
            "moderator_override": moderator_override,
            "text_sample": text_sample[:200],
            "category": category,
            "confidence_before": "",
            "confidence_after": "",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    return f"Feedback recorded for content {content_id}: {original_decision} -> {moderator_override}."


MODERATION_TOOL_FUNCTIONS = {
    "approve_content": approve_content,
    "reject_content": reject_content,
    "flag_for_human_review": flag_for_human_review,
    "shadow_ban_user": shadow_ban_user,
    "get_user_moderation_history": get_user_moderation_history,
    "find_similar_violations": find_similar_violations,
    "add_to_watchlist": add_to_watchlist,
}
