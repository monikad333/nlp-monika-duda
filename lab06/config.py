from __future__ import annotations

import os

PII_MODEL_NAME = os.getenv("LAB6_PII_MODEL", "openai/privacy-filter")

BIELIK_GUARD_MODEL_NAME = os.getenv("LAB6_BIELIK_GUARD_MODEL", "speakleash/Bielik-Guard-0.1B-v1.0")
GUARD_CATEGORIES = ["toxic", "spam", "hate_speech", "self_harm", "violence", "sexual", "clean"]

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
QWEN_GUARD_MODEL = os.getenv("LAB6_QWEN_GUARD_MODEL", "qwen2.5:1.5b")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("LAB6_OLLAMA_TIMEOUT", "30"))

MODERATION_LOG_PATH = os.getenv("LAB6_MODERATION_LOG_PATH", "lab06/moderation_log.csv")
USER_HISTORY_PATH = os.getenv("LAB6_USER_HISTORY_PATH", "lab06/user_moderation_history.csv")
FEEDBACK_LOG_PATH = os.getenv("LAB6_FEEDBACK_LOG_PATH", "lab06/feedback_log.csv")
WATCHLIST_PATH = os.getenv("LAB6_WATCHLIST_PATH", "lab06/watchlist.csv")

REPEAT_OFFENDER_THRESHOLD = int(os.getenv("LAB6_REPEAT_OFFENDER_THRESHOLD", "3"))
FEEDBACK_RETRAIN_THRESHOLD = int(os.getenv("LAB6_FEEDBACK_RETRAIN_THRESHOLD", "5"))

VALID_ACTIONS = ["APPROVE", "REJECT", "FLAG_FOR_REVIEW", "SHADOW_BAN"]
