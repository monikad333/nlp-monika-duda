from __future__ import annotations

import ast
import json
import operator
import os
from typing import Any

import requests

from lab05.config import (
    GEOCODING_API_URL,
    KNOWLEDGE_BASE_PATH,
    REQUEST_HEADERS,
    VISION_MODEL,
    WEATHER_API_URL,
    WIKIPEDIA_SEARCH_API_URL,
)


class ToolError(ValueError):
    pass


def web_search(query: str) -> str:
    """Search Wikipedia and return a short summary of the top result."""
    if not query or not query.strip():
        raise ToolError("query cannot be empty.")

    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1,
    }
    response = requests.get(WIKIPEDIA_SEARCH_API_URL, params=search_params, headers=REQUEST_HEADERS, timeout=10)
    response.raise_for_status()
    results = response.json().get("query", {}).get("search", [])

    if not results:
        return f"No web results found for '{query}'."

    title = results[0]["title"]
    summary_params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json",
    }
    summary_response = requests.get(WIKIPEDIA_SEARCH_API_URL, params=summary_params, headers=REQUEST_HEADERS, timeout=10)
    summary_response.raise_for_status()
    pages = summary_response.json().get("query", {}).get("pages", {})
    extract = next(iter(pages.values()), {}).get("extract", "")

    return f"{title}: {extract[:500]}" if extract else f"Found page '{title}' but no summary available."


def analyze_image(image_path: str) -> str:
    """Describe image content using a local Ollama vision model."""
    if not image_path or not os.path.exists(image_path):
        raise ToolError(f"Image not found at '{image_path}'.")

    import base64

    from lab05.config import OLLAMA_HOST, OLLAMA_TIMEOUT_SECONDS

    with open(image_path, "rb") as file:
        image_b64 = base64.b64encode(file.read()).decode("utf-8")

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": VISION_MODEL,
                "prompt": "Describe this image in detail, in English.",
                "images": [image_b64],
                "stream": False,
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ToolError(
            f"Could not reach Ollama vision model '{VISION_MODEL}'. Is it pulled ('ollama pull {VISION_MODEL}')? "
            f"Original error: {exc}"
        ) from exc

    return response.json().get("response", "").strip()


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval_node(node.operand))
    raise ToolError("Expression contains unsupported syntax.")


def simple_calculator(expression: str) -> str:
    """Evaluate a basic math expression (supports + - * / ** % and parentheses)."""
    if not expression or not expression.strip():
        raise ToolError("expression cannot be empty.")

    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval_node(parsed.body)
    except (SyntaxError, ToolError, ZeroDivisionError, TypeError) as exc:
        raise ToolError(f"Could not evaluate expression '{expression}': {exc}") from exc

    return f"{expression} = {result}"


def _load_knowledge_base(path: str = KNOWLEDGE_BASE_PATH) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def local_knowledge(query: str) -> str:
    """Search a local JSON knowledge base for matching facts."""
    if not query or not query.strip():
        raise ToolError("query cannot be empty.")

    records = _load_knowledge_base()
    if not records:
        return "Local knowledge base is empty."

    query_tokens = set(query.lower().split())
    best_match = None
    best_score = 0

    for record in records:
        haystack = f"{record.get('question', '')} {record.get('keywords', '')}".lower()
        score = sum(1 for token in query_tokens if token in haystack)
        if score > best_score:
            best_score = score
            best_match = record

    if not best_match or best_score == 0:
        return f"No local knowledge entry found for '{query}'."

    return best_match.get("answer", "")


def _geocode_city(city: str) -> tuple[float, float, str]:
    params = {"name": city, "count": 5, "language": "pl"}
    response = requests.get(GEOCODING_API_URL, params=params, headers=REQUEST_HEADERS, timeout=10)
    response.raise_for_status()
    results = response.json().get("results")

    if not results:
        raise ToolError(f"City '{city}' not found.")

    location = max(results, key=lambda item: item.get("population", 0))
    return location["latitude"], location["longitude"], location.get("name", city)


def get_weather(city: str) -> str:
    """Return current weather for a given city."""
    if not city or not city.strip():
        raise ToolError("city cannot be empty.")

    latitude, longitude, resolved_name = _geocode_city(city)

    params = {"latitude": latitude, "longitude": longitude, "current_weather": True}
    response = requests.get(WEATHER_API_URL, params=params, headers=REQUEST_HEADERS, timeout=10)
    response.raise_for_status()
    weather = response.json().get("current_weather", {})

    if not weather:
        raise ToolError(f"Weather data not available for '{city}'.")

    return f"{resolved_name}: {weather['temperature']}°C, wind {weather['windspeed']} km/h"


TOOL_FUNCTIONS = {
    "web_search": web_search,
    "analyze_image": analyze_image,
    "simple_calculator": simple_calculator,
    "local_knowledge": local_knowledge,
    "get_weather": get_weather,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    function = TOOL_FUNCTIONS.get(name)
    if not function:
        raise ToolError(f"Unknown tool '{name}'. Allowed: {list(TOOL_FUNCTIONS)}")

    try:
        return str(function(**arguments))
    except ToolError as exc:
        return f"Tool error: {exc}"
    except Exception as exc:
        return f"Tool '{name}' failed: {exc}"
