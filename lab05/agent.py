from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

from lab05.config import HISTORY_PATH, MAX_TOOL_ROUNDS, OLLAMA_HOST, OLLAMA_TIMEOUT_SECONDS, TOOL_MODEL
from lab05.tool_schemas import TOOL_SCHEMAS
from lab05.tools import execute_tool


class AgentError(ValueError):
    pass


def _call_ollama_chat(messages: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={"model": TOOL_MODEL, "messages": messages, "tools": TOOL_SCHEMAS, "stream": False},
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise AgentError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Is it running ('ollama serve') and is model "
            f"'{TOOL_MODEL}' pulled ('ollama pull {TOOL_MODEL}')? Original error: {exc}"
        ) from exc

    return response.json()


def _append_history(entry: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(HISTORY_PATH) or ".", exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_agent(user_text: str) -> dict[str, Any]:
    if not user_text or not user_text.strip():
        raise AgentError("Text cannot be empty.")

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
    tool_call_log: list[dict[str, Any]] = []
    started_at = time.time()

    for _ in range(MAX_TOOL_ROUNDS):
        payload = _call_ollama_chat(messages)
        message = payload.get("message", {})
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            final_answer = message.get("content", "").strip()
            elapsed_seconds = round(time.time() - started_at, 2)

            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "user_text": user_text,
                "tool_calls": tool_call_log,
                "final_answer": final_answer,
                "elapsed_seconds": elapsed_seconds,
            }
            _append_history(entry)

            return {"answer": final_answer, "tool_calls": tool_call_log, "elapsed_seconds": elapsed_seconds}

        messages.append(message)

        for tool_call in tool_calls:
            function_info = tool_call.get("function", {})
            tool_name = function_info.get("name", "")
            raw_arguments = function_info.get("arguments", {})
            arguments = raw_arguments if isinstance(raw_arguments, dict) else json.loads(raw_arguments or "{}")

            result = execute_tool(tool_name, arguments)
            tool_call_log.append({"tool": tool_name, "arguments": arguments, "result": result})

            messages.append({"role": "tool", "content": result, "name": tool_name})

    raise AgentError(f"Reached max tool-call rounds ({MAX_TOOL_ROUNDS}) without a final answer.")
