from __future__ import annotations

from collections import Counter
from typing import Any

from lab06.storage import load_feedback_log, load_moderation_log


def build_analytics_report() -> dict[str, Any]:
    log = load_moderation_log()
    total = len(log)

    action_counts = Counter(row["action"] for row in log)
    approved = action_counts.get("APPROVE", 0)
    rejected = action_counts.get("REJECT", 0)
    flagged = action_counts.get("FLAG_FOR_REVIEW", 0)

    violation_categories: Counter[str] = Counter()
    for row in log:
        if row["action"] in {"REJECT", "FLAG_FOR_REVIEW"}:
            for category in (row.get("reason") or "").split("+"):
                if category and category not in {"model_disagreement", "personally_identifiable_information"}:
                    violation_categories[category] += 1

    user_violation_counts: Counter[str] = Counter()
    for row in log:
        if row["action"] in {"REJECT", "FLAG_FOR_REVIEW"}:
            user_violation_counts[row["user_id"]] += 1

    consensus_counts = Counter()
    for row in log:
        bielik_flagged = row["model_bielik_decision"].lower() not in {"clean", "neutral", "safe"}
        qwen_flagged = row["model_qwen_decision"] in {"high", "critical"}
        pii_flagged = row["pii_detected"] in {"True", "true", True}

        votes = sum([bielik_flagged, qwen_flagged, pii_flagged])
        if votes == 0 or votes == 3:
            consensus_counts["all_agree"] += 1
        elif votes == 2:
            consensus_counts["two_thirds_agree"] += 1
        else:
            consensus_counts["conflicting"] += 1

    feedback_log = load_feedback_log()
    human_overrides = sum(1 for row in feedback_log if row["original_bot_decision"] != row["moderator_override"])

    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "flagged": flagged,
        "top_violations": violation_categories.most_common(5),
        "repeat_offenders": user_violation_counts.most_common(5),
        "consensus": consensus_counts,
        "human_overrides": human_overrides,
    }


def format_analytics_report() -> str:
    report = build_analytics_report()
    total = report["total"] or 1

    lines = [
        "MODERATION ANALYTICS",
        "=" * 40,
        f"Total posts reviewed:  {report['total']}",
        f"Approved:              {report['approved']} ({report['approved'] / total:.1%})",
        f"Rejected:              {report['rejected']} ({report['rejected'] / total:.1%})",
        f"Flagged for review:    {report['flagged']} ({report['flagged'] / total:.1%})",
        "",
        "TOP VIOLATIONS:",
    ]
    for category, count in report["top_violations"]:
        lines.append(f"- {category}: {count} cases")

    lines.append("")
    lines.append("REPEAT OFFENDERS:")
    for user_id, count in report["repeat_offenders"]:
        lines.append(f"- {user_id}: {count} violations")

    lines.append("")
    lines.append("MODEL CONSENSUS:")
    consensus = report["consensus"]
    lines.append(f"- All models agree: {consensus.get('all_agree', 0)}")
    lines.append(f"- 2/3 agree: {consensus.get('two_thirds_agree', 0)}")
    lines.append(f"- Conflicting: {consensus.get('conflicting', 0)}")
    lines.append(f"- Human overrides: {report['human_overrides']}")

    return "\n".join(lines)
