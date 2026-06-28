from __future__ import annotations

import os

PLOTS_DIR = os.getenv("LAB4_PLOTS_DIR", "lab04/lab4plots")
SUMMARIES_DIR = os.getenv("LAB4_SUMMARIES_DIR", "lab04/summaries")
KNOWLEDGE_BASE_PATH = os.getenv("LAB4_KNOWLEDGE_BASE_PATH", "lab04/knowledge_base.json")

SUPPORTED_LANGUAGES = ["en", "pl", "de", "fr", "es"]
NER_METHODS = ["spacy", "stanza"]
SUMMARY_TYPES = ["extractive", "abstractive", "bullets"]
SUMMARY_LENGTHS = {"short": 2, "medium": 5, "long": 9}

SPACY_MODELS = {"en": "en_core_web_sm", "pl": "pl_core_news_sm"}

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("LAB4_OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("LAB4_OLLAMA_TIMEOUT", "60"))

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API_URL_TEMPLATE = "https://{language}.wikipedia.org/api/rest_v1/page/summary/{title}"

TRANSLATION_MODEL_TEMPLATE = "Helsinki-NLP/opus-mt-{source}-{target}"

# Helsinki-NLP has no direct en->pl checkpoint; use the multi-target Slavic model with a language token.
TRANSLATION_MODEL_OVERRIDES = {
    ("en", "pl"): {"model": "Helsinki-NLP/opus-mt-en-sla", "target_prefix": ">>pol<< "},
}
NEL_CONFIDENCE_THRESHOLD = float(os.getenv("LAB4_NEL_CONFIDENCE_THRESHOLD", "0.3"))
