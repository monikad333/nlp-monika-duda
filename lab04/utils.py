from __future__ import annotations

import shlex


class CommandArgsError(ValueError):
    pass


def parse_key_value_message(message_text: str) -> dict[str, str]:
    args_text = message_text.partition(" ")[2]

    try:
        tokens = shlex.split(args_text)
    except ValueError as exc:
        raise CommandArgsError(f"Could not parse arguments: {exc}") from exc

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
