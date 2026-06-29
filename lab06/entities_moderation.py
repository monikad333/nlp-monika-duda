from __future__ import annotations

import re
from typing import Any

_USERNAME_PATTERN = re.compile(r"@\w+")
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[\s-]?)?(\d{2,3}[\s-]?){2,4}\d{2,4}")


def extract_moderation_entities(text: str) -> dict[str, Any]:
    """Extract key entities for moderation context, reusing Lab04's NER plus regex for handles/contacts."""
    from lab04.ner import run_ner

    usernames = [match.group() for match in _USERNAME_PATTERN.finditer(text)]
    urls = [match.group() for match in _URL_PATTERN.finditer(text)]
    emails = [match.group() for match in _EMAIL_PATTERN.finditer(text)]
    phone_numbers = [match.group().strip() for match in _PHONE_PATTERN.finditer(text) if len(match.group().strip()) >= 7]

    try:
        entities = run_ner("spacy", text)
    except Exception:
        entities = []

    already_classified = set(usernames) | set(urls) | set(emails)

    organizations = [
        entity["text"] for entity in entities if entity["label"] in {"ORG", "orgName"} and entity["text"] not in already_classified
    ]
    locations = [
        entity["text"] for entity in entities if entity["label"] in {"GPE", "LOC", "placeName"} and entity["text"] not in already_classified
    ]
    persons = [
        entity["text"] for entity in entities if entity["label"] in {"PERSON", "persName"} and entity["text"] not in already_classified
    ]

    return {
        "usernames_mentioned": usernames,
        "urls": urls,
        "emails": emails,
        "phone_numbers": phone_numbers,
        "organizations": organizations,
        "locations": locations,
        "persons": persons,
    }
