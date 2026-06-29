from __future__ import annotations

import asyncio
import logging
import shlex

from telegram import Update
from telegram.ext import ContextTypes

from lab06.analytics import format_analytics_report
from lab06.ensemble import moderate_text
from lab06.feedback import submit_feedback, train_on_feedback
from lab06.storage import load_moderation_log, load_watchlist
from lab06.tools import get_user_moderation_history

LOGGER = logging.getLogger(__name__)

HELP_TEXT = (
    "Lab06 - Content Moderation commands:\n"
    '/moderate "tekst do sprawdzenia"\n'
    "/mod_status <content_id>\n"
    "/mod_history <user_id>\n"
    "/mod_analytics\n"
    '/mod_add_feedback <content_id> "komentarz" "poprawna_decyzja"\n'
    "/mod_watchlist\n"
    "/mod_train_on_feedback\n"
    '/mod_policy_check "tekst" (dry-run, nie zapisuje do logow)\n\n'
    "Examples:\n"
    '/moderate "Uwielbiam ten produkt, najlepszy zakup!"\n'
    '/moderate "Jesteś głupszy niż cegła, powinieneś się zabić"\n'
    '/mod_add_feedback 1234567890 "to bylo zle ocenione" "APPROVE"'
)


def _format_moderation_result(result: dict) -> str:
    action_emoji = {"APPROVE": "✅", "REJECT": "❌", "FLAG_FOR_REVIEW": "⏳"}.get(result["action"], "ℹ️")

    lines = [
        f"{action_emoji} Action: {result['action']}",
        f"Content ID: {result['content_id']}",
    ]
    if result["reason"]:
        lines.append(f"Reason: {result['reason']}")

    lines.append("")
    lines.append(f"PII detected: {result['pii']['has_pii']}")
    lines.append(f"Bielik Guard: {result['bielik_guard']['label']} (severity={result['bielik_guard']['severity']})")
    lines.append(f"Qwen Guard: {result['qwen_guard']['risk_level']} (categories={result['qwen_guard']['categories']})")
    lines.append(
        f"Sentiment: {result['sentiment']['sentiment']} (emotion={result['sentiment']['emotion']}, "
        f"sarcasm={result['sentiment']['sarcasm_detected']})"
    )

    entities = result["entities"]
    entity_summary = ", ".join(
        f"{key}={values}" for key, values in entities.items() if values
    )
    if entity_summary:
        lines.append(f"Entities: {entity_summary}")

    if result["is_repeat_offender"]:
        lines.append("⚠️ User flagged as repeat offender.")

    if result["tool_result"]:
        lines.append("")
        lines.append(f"Tool: {result['tool_result']}")

    return "\n".join(lines)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.message:
        await update.message.reply_text(HELP_TEXT)


async def moderate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    try:
        args = shlex.split(update.message.text)
    except ValueError as exc:
        await update.message.reply_text(f"Could not parse arguments: {exc}")
        return

    if len(args) != 2:
        await update.message.reply_text('Usage: /moderate "tekst do sprawdzenia"')
        return

    text = args[1]
    user_id = str(update.effective_user.id) if update.effective_user else "anonymous"

    await update.message.reply_text("Moderating, this may take a while (3 safety models + sentiment + NER)...")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, moderate_text, text, user_id)
    except Exception:
        LOGGER.exception("/moderate failed")
        await update.message.reply_text("Internal error while moderating content.")
        return

    await update.message.reply_text(_format_moderation_result(result))


async def mod_policy_check_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    try:
        args = shlex.split(update.message.text)
    except ValueError as exc:
        await update.message.reply_text(f"Could not parse arguments: {exc}")
        return

    if len(args) != 2:
        await update.message.reply_text('Usage: /mod_policy_check "tekst"')
        return

    await update.message.reply_text("Checking policy (dry-run, no action taken or logged)...")

    loop = asyncio.get_event_loop()
    try:
        from lab06.safety_models import classify_bielik_guard, classify_qwen_guard, detect_private_info
        from lab06.sentiment_moderation import analyze_sentiment_for_moderation

        pii_result = await loop.run_in_executor(None, detect_private_info, args[1])
        bielik_result = await loop.run_in_executor(None, classify_bielik_guard, args[1])
        qwen_result = await loop.run_in_executor(None, classify_qwen_guard, args[1])
        sentiment_result = analyze_sentiment_for_moderation(args[1])
    except Exception:
        LOGGER.exception("/mod_policy_check failed")
        await update.message.reply_text("Internal error while checking policy.")
        return

    lines = [
        "POLICY CHECK (dry-run):",
        f"PII detected: {pii_result['has_pii']}",
        f"Bielik Guard: {bielik_result['label']} (score={bielik_result['score']}, severity={bielik_result['severity']})",
        f"Qwen Guard: {qwen_result['risk_level']} (categories={qwen_result['categories']})",
        f"Sentiment: {sentiment_result['sentiment']} (emotion={sentiment_result['emotion']})",
    ]
    await update.message.reply_text("\n".join(lines))


async def mod_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    parts = update.message.text.split()
    if len(parts) != 2:
        await update.message.reply_text("Usage: /mod_status <content_id>")
        return

    content_id = parts[1]
    entry = next((row for row in load_moderation_log() if row["content_id"] == content_id), None)

    if not entry:
        await update.message.reply_text(f"No moderation record found for content_id '{content_id}'.")
        return

    lines = [f"Content {content_id}:"]
    for key in ("action", "reason", "model_bielik_decision", "model_qwen_decision", "pii_detected", "sentiment", "timestamp"):
        lines.append(f"- {key}: {entry[key]}")

    await update.message.reply_text("\n".join(lines))


async def mod_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    parts = update.message.text.split()
    if len(parts) != 2:
        await update.message.reply_text("Usage: /mod_history <user_id>")
        return

    user_id = parts[1]
    history = get_user_moderation_history(user_id)

    lines = [
        f"Moderation history for {user_id}:",
        f"Violations: {history['violations_count']}",
        f"Last violation: {history['last_violation'] or 'none'}",
        f"Categories: {history['categories']}",
        f"Risk score: {history['risk_score']}",
        f"Repeat offender: {history['is_repeat_offender']}",
    ]
    await update.message.reply_text("\n".join(lines))


async def mod_analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message:
        return

    await update.message.reply_text(format_analytics_report())


async def mod_add_feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    try:
        args = shlex.split(update.message.text)
    except ValueError as exc:
        await update.message.reply_text(f"Could not parse arguments: {exc}")
        return

    if len(args) != 4:
        await update.message.reply_text('Usage: /mod_add_feedback <content_id> "komentarz" "poprawna_decyzja"')
        return

    _, content_id, comment, correct_decision = args

    try:
        result = submit_feedback(content_id, moderator_decision=correct_decision.upper(), comment=comment[:200])
    except ValueError as exc:
        await update.message.reply_text(f"Error: {exc}")
        return

    lines = [
        "Dziękujemy za feedback!",
        f"Original decision: {result['original_decision']} -> Override: {result['moderator_override']}",
        f"Total feedback entries: {result['feedback_count']}",
    ]
    if result["retrained"]:
        lines.append("Model lexicon retrained from accumulated feedback.")

    await update.message.reply_text("\n".join(lines))


async def mod_watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message:
        return

    watchlist = load_watchlist()
    if not watchlist:
        await update.message.reply_text("Watchlist is empty.")
        return

    lines = ["WATCHLIST:"]
    for entry in watchlist:
        lines.append(f"- {entry['user_id']}: {entry['reason']} (added {entry['added_at']})")

    await update.message.reply_text("\n".join(lines))


async def mod_train_on_feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message:
        return

    await update.message.reply_text("Training on accumulated feedback...")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, train_on_feedback)

    if not result["categories_updated"]:
        await update.message.reply_text("No new patterns learned (not enough false-negative feedback yet).")
        return

    await update.message.reply_text(f"Lexicon updated for categories: {result['categories_updated']}")
