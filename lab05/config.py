from __future__ import annotations

import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TOOL_MODEL = os.getenv("LAB5_TOOL_MODEL", "llama3.2")
VISION_MODEL = os.getenv("LAB5_VISION_MODEL", "moondream")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("LAB5_OLLAMA_TIMEOUT", "60"))

MAX_TOOL_ROUNDS = int(os.getenv("LAB5_MAX_TOOL_ROUNDS", "5"))

KNOWLEDGE_BASE_PATH = os.getenv("LAB5_KNOWLEDGE_BASE_PATH", "lab05/knowledge_base.json")
HISTORY_PATH = os.getenv("LAB5_HISTORY_PATH", "lab05/history.jsonl")
IMAGES_DIR = os.getenv("LAB5_IMAGES_DIR", "lab05/images")

WIKIPEDIA_SEARCH_API_URL = "https://en.wikipedia.org/w/api.php"
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

REQUEST_HEADERS = {"User-Agent": "WSEI-NLP-Lab05-Bot/1.0 (educational project)"}
