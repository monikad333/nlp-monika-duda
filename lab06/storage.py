from __future__ import annotations

import csv
import os
from typing import Any

from lab06.config import FEEDBACK_LOG_PATH, MODERATION_LOG_PATH, USER_HISTORY_PATH, WATCHLIST_PATH

MODERATION_LOG_FIELDS = [
    "timestamp",
    "content_id",
    "user_id",
    "text",
    "model_bielik_decision",
    "model_bielik_score",
    "model_qwen_decision",
    "model_qwen_score",
    "pii_detected",
    "sentiment",
    "action",
    "moderator_override",
    "reason",
    "appeal_filed",
]

USER_HISTORY_FIELDS = [
    "user_id",
    "username",
    "total_violations",
    "last_violation_date",
    "categories",
    "risk_score",
    "is_repeat_offender",
    "shadow_bans",
    "appeals_filed",
]

FEEDBACK_LOG_FIELDS = [
    "content_id",
    "original_bot_decision",
    "moderator_override",
    "text_sample",
    "category",
    "confidence_before",
    "confidence_after",
    "timestamp",
]

WATCHLIST_FIELDS = ["user_id", "reason", "added_at"]


def _read_csv(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: str, fields: list[str], rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_csv(path: str, fields: list[str], row: dict[str, Any]) -> None:
    file_exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_moderation_log(entry: dict[str, Any]) -> None:
    _append_csv(MODERATION_LOG_PATH, MODERATION_LOG_FIELDS, entry)


def load_moderation_log() -> list[dict[str, str]]:
    return _read_csv(MODERATION_LOG_PATH)


def get_user_history_record(user_id: str) -> dict[str, Any] | None:
    for record in _read_csv(USER_HISTORY_PATH):
        if record["user_id"] == user_id:
            return record
    return None


def upsert_user_history(
    user_id: str, username: str, violation_categories: list[str] | None, shadow_banned: bool = False
) -> dict[str, Any]:
    import time

    records = _read_csv(USER_HISTORY_PATH)
    existing = next((record for record in records if record["user_id"] == user_id), None)

    if existing is None:
        existing = {
            "user_id": user_id,
            "username": username,
            "total_violations": "0",
            "last_violation_date": "",
            "categories": "",
            "risk_score": "0.0",
            "is_repeat_offender": "False",
            "shadow_bans": "0",
            "appeals_filed": "0",
        }
        records.append(existing)

    if violation_categories:
        total_violations = int(existing["total_violations"]) + 1
        existing_categories = set(existing["categories"].split(";")) if existing["categories"] else set()
        existing_categories.update(violation_categories)

        existing["total_violations"] = str(total_violations)
        existing["last_violation_date"] = time.strftime("%Y-%m-%d")
        existing["categories"] = ";".join(sorted(existing_categories))
        existing["risk_score"] = str(round(min(1.0, total_violations / 10), 2))
        existing["is_repeat_offender"] = str(total_violations >= 3)

    if shadow_banned:
        existing["shadow_bans"] = str(int(existing["shadow_bans"]) + 1)

    _write_csv(USER_HISTORY_PATH, USER_HISTORY_FIELDS, records)
    return existing


def append_feedback_log(entry: dict[str, Any]) -> None:
    _append_csv(FEEDBACK_LOG_PATH, FEEDBACK_LOG_FIELDS, entry)


def load_feedback_log() -> list[dict[str, str]]:
    return _read_csv(FEEDBACK_LOG_PATH)


def add_to_watchlist_storage(user_id: str, reason: str) -> None:
    import time

    _append_csv(WATCHLIST_PATH, WATCHLIST_FIELDS, {"user_id": user_id, "reason": reason, "added_at": time.strftime("%Y-%m-%d %H:%M:%S")})


def load_watchlist() -> list[dict[str, str]]:
    return _read_csv(WATCHLIST_PATH)
