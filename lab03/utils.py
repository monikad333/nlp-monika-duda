from __future__ import annotations

from lab03.config import VALID_LABELS


class CommandArgsError(ValueError):
    pass


def parse_key_value_args(args_text: str) -> dict[str, str]:
    tokens = (args_text or "").strip().split()
    params: dict[str, str] = {}

    for token in tokens:
        if "=" not in token:
            raise CommandArgsError(f"Invalid argument '{token}'. Expected key=value.")
        key, value = token.split("=", 1)
        params[key.strip().lower()] = value.strip()

    return params


def require_params(params: dict[str, str], required: set[str], usage: str) -> None:
    missing = required - params.keys()
    if missing:
        raise CommandArgsError(f"Missing parameters: {sorted(missing)}. Usage: {usage}")


def normalize_label(label: str) -> str:
    return (label or "").strip().lower()


def is_valid_label(label: str) -> bool:
    return normalize_label(label) in VALID_LABELS
