from __future__ import annotations

import time
from typing import Any

from lab06.entities_moderation import extract_moderation_entities
from lab06.safety_models import classify_bielik_guard, classify_qwen_guard, detect_private_info
from lab06.sentiment_moderation import analyze_sentiment_for_moderation
from lab06.storage import append_moderation_log, upsert_user_history
from lab06.tools import (
    approve_content,
    flag_for_human_review,
    reject_content,
    shadow_ban_user,
)

_NON_VIOLATION_BIELIK_LABELS = {"clean", "neutral", "safe"}
_HARMFUL_CATEGORY_KEYWORDS = {
    "toxic", "spam", "hate", "self_harm", "self-harm", "violence", "harm", "sexual", "harassment", "abuse",
}


def _bielik_flags_violation(bielik_result: dict[str, Any]) -> bool:
    return bielik_result["label"].lower() not in _NON_VIOLATION_BIELIK_LABELS and bielik_result["score"] > 0.8


def _qwen_flags_violation(qwen_result: dict[str, Any]) -> bool:
    if qwen_result["risk_level"] not in {"safe", "low"}:
        return True
    categories_text = " ".join(qwen_result.get("categories", [])).lower()
    return any(keyword in categories_text for keyword in _HARMFUL_CATEGORY_KEYWORDS)


def _next_content_id() -> str:
    return str(int(time.time() * 1000))


def moderate_text(text: str, user_id: str = "anonymous", content_id: str | None = None) -> dict[str, Any]:
    """Run the full moderation pipeline: PII + Bielik Guard + Qwen Guard + sentiment + entities, then decide."""
    content_id = content_id or _next_content_id()

    pii_result = detect_private_info(text)
    bielik_result = classify_bielik_guard(text)
    qwen_result = classify_qwen_guard(text)
    sentiment_result = analyze_sentiment_for_moderation(text)
    entities_result = extract_moderation_entities(text)

    bielik_violation = _bielik_flags_violation(bielik_result)
    qwen_violation = _qwen_flags_violation(qwen_result)

    reason = ""
    flag_account = False

    if pii_result["has_pii"]:
        action = "REJECT"
        reason = "personally_identifiable_information"
    elif qwen_result["risk_level"] == "critical":
        action = "REJECT"
        flag_account = True
        reason = "+".join(qwen_result["categories"]) or "critical_risk"
    else:
        votes_for_violation = sum([bielik_violation, qwen_violation])

        if votes_for_violation >= 2:
            action = "REJECT"
            reason = bielik_result["label"] if bielik_violation else "+".join(qwen_result["categories"])
            flag_account = bielik_result["severity"] == "critical" or qwen_result["risk_level"] == "critical"
        elif votes_for_violation == 1:
            action = "FLAG_FOR_REVIEW"
            reason = "model_disagreement"
        else:
            action = "APPROVE"

    tool_result = None
    if action == "APPROVE":
        tool_result = approve_content(content_id, moderator_id="bot")
    elif action == "REJECT":
        tool_result = reject_content(content_id, reason=reason, moderator_id="bot")
        if flag_account:
            tool_result += " " + shadow_ban_user(user_id, duration_hours=24, reason=reason)
    elif action == "FLAG_FOR_REVIEW":
        priority = "high" if (bielik_violation or qwen_violation) else "medium"
        tool_result = flag_for_human_review(content_id, priority=priority, reason=reason or "needs_human_judgement")

    user_history = None
    if action in {"REJECT", "FLAG_FOR_REVIEW"}:
        categories = [category for category in (bielik_result["label"].split("+") + qwen_result["categories"]) if category]
        user_history = upsert_user_history(user_id, username=user_id, violation_categories=categories or ["unspecified"])

    append_moderation_log(
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "content_id": content_id,
            "user_id": user_id,
            "text": text[:300],
            "model_bielik_decision": bielik_result["label"],
            "model_bielik_score": bielik_result["score"],
            "model_qwen_decision": qwen_result["risk_level"],
            "model_qwen_score": qwen_result["confidence"],
            "pii_detected": pii_result["has_pii"],
            "sentiment": sentiment_result["sentiment"],
            "action": action,
            "moderator_override": False,
            "reason": reason,
            "appeal_filed": False,
        }
    )

    return {
        "content_id": content_id,
        "action": action,
        "reason": reason,
        "tool_result": tool_result,
        "pii": pii_result,
        "bielik_guard": bielik_result,
        "qwen_guard": qwen_result,
        "sentiment": sentiment_result,
        "entities": entities_result,
        "is_repeat_offender": bool(user_history and user_history.get("is_repeat_offender") == "True"),
    }
