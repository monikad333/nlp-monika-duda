from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import ContextTypes

from lab04.config import PLOTS_DIR
from lab04.knowledge_graph import build_entity_relations, save_knowledge_graph
from lab04.language import detect_language
from lab04.ner import NERError, format_entities, run_ner
from lab04.nel import NELError, disambiguate_entity, link_entity
from lab04.summarization import SummarizationError, generate_summary
from lab04.translation import TranslationError, translate_text
from lab04.utils import CommandArgsError, parse_key_value_message, require_params

LOGGER = logging.getLogger(__name__)

HELP_TEXT = (
    "Lab04 commands:\n"
    '/ner method=<spacy|stanza> text="tekst"\n'
    '/nel text="tekst" language=<en|pl>\n'
    '/ned entity="tekst" context="tekst"\n'
    '/translate text="tekst" target_lang=<en|pl|de|fr|es>\n'
    '/summarize text="tekst" summary_type=<extractive|abstractive|bullets> length=<short|medium|long>\n'
    '/analyze_entities text="tekst" link=<true|false>\n'
    '/knowledge_graph text="tekst"\n'
    '/language_detect text="tekst"\n\n'
    "Examples:\n"
    '/ner method=spacy text="Steve Jobs, założyciel Apple, urodził się w San Francisco."\n'
    '/translate text="The quick brown fox jumps over the lazy dog" target_lang=pl\n'
    '/summarize text="Dlugi tekst..." summary_type=abstractive length=medium\n'
    '/analyze_entities text="Elon Musk posiada firme Tesla w Austin." link=true'
)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.message:
        await update.message.reply_text(HELP_TEXT)


async def ner_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    try:
        params = parse_key_value_message(update.message.text)
        require_params(params, {"method", "text"}, '/ner method=<spacy|stanza> text="tekst"')
    except CommandArgsError as exc:
        await update.message.reply_text(str(exc))
        return

    try:
        loop = asyncio.get_event_loop()
        entities = await loop.run_in_executor(None, run_ner, params["method"], params["text"])
    except NERError as exc:
        await update.message.reply_text(f"Error: {exc}")
        return
    except Exception:
        LOGGER.exception("/ner failed")
        await update.message.reply_text("Internal error while running NER.")
        return

    response = (
        f"Metoda: {params['method'].capitalize()}\n"
        f"TEXT: {params['text']}\n\n"
        f"ENTITIES:\n{format_entities(entities)}"
    )
    await update.message.reply_text(response)


async def nel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    try:
        params = parse_key_value_message(update.message.text)
        require_params(params, {"text"}, '/nel text="tekst" language=<en|pl>')
    except CommandArgsError as exc:
        await update.message.reply_text(str(exc))
        return

    language = params.get("language") or detect_language(params["text"])

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, link_entity, params["text"], language)
    except NELError as exc:
        await update.message.reply_text(f"Error: {exc}")
        return
    except Exception:
        LOGGER.exception("/nel failed")
        await update.message.reply_text("Internal error while running NEL.")
        return

    if not result["candidates"]:
        await update.message.reply_text(f"Entity: {result['entity']}\nNo candidates found.")
        return

    lines = [f"Entity: {result['entity']}", "Candidates:"]
    for index, candidate in enumerate(result["candidates"], start=1):
        lines.append(f"{index}. {candidate['label']} ({candidate['id']}) - {candidate['description']}")
        if candidate.get("wikipedia_url"):
            lines.append(f"   - Wikipedia: {candidate['wikipedia_url']}")
        lines.append(f"   - Confidence: {candidate['confidence']}")

    await update.message.reply_text("\n".join(lines))


async def ned_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    try:
        params = parse_key_value_message(update.message.text)
        require_params(params, {"entity", "context"}, '/ned entity="tekst" context="tekst"')
    except CommandArgsError as exc:
        await update.message.reply_text(str(exc))
        return

    language = params.get("language") or detect_language(params["context"])

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, disambiguate_entity, params["entity"], params["context"], language)
    except NELError as exc:
        await update.message.reply_text(f"Error: {exc}")
        return
    except Exception:
        LOGGER.exception("/ned failed")
        await update.message.reply_text("Internal error while running NED.")
        return

    best = result["best_candidate"]
    if not best:
        await update.message.reply_text(f"Entity: {result['entity']}\nNo candidates found.")
        return

    lines = [
        f"Entity: {result['entity']}",
        f"Best match: {best['label']} ({best['id']}) - {best['description']}",
        f"Confidence: {best['confidence']}",
        "Other candidates:",
    ]
    for candidate in result["candidates"][1:]:
        lines.append(f"- {candidate['label']} ({candidate['id']}) - confidence {candidate['confidence']}")

    await update.message.reply_text("\n".join(lines))


async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    try:
        params = parse_key_value_message(update.message.text)
        require_params(params, {"text", "target_lang"}, '/translate text="tekst" target_lang=<en|pl|de|fr|es>')
    except CommandArgsError as exc:
        await update.message.reply_text(str(exc))
        return

    await update.message.reply_text("Translating, this may download a model on first use...")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, translate_text, params["text"], params["target_lang"], params.get("source_lang")
        )
    except TranslationError as exc:
        await update.message.reply_text(f"Error: {exc}")
        return
    except Exception:
        LOGGER.exception("/translate failed")
        await update.message.reply_text("Internal error while translating.")
        return

    response = f"Source: {result['source']}\nTarget: {result['target']}\nTranslation: {result['translation']}"
    await update.message.reply_text(response)


async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    try:
        params = parse_key_value_message(update.message.text)
        require_params(
            params,
            {"text"},
            '/summarize text="tekst" summary_type=<extractive|abstractive|bullets> length=<short|medium|long>',
        )
    except CommandArgsError as exc:
        await update.message.reply_text(str(exc))
        return

    summary_type = params.get("summary_type", "abstractive")
    length = params.get("length", "medium")

    await update.message.reply_text("Generating summary with Ollama, this may take a while...")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, generate_summary, params["text"], summary_type, length, params.get("prompt")
        )
    except SummarizationError as exc:
        await update.message.reply_text(f"Error: {exc}")
        return
    except Exception:
        LOGGER.exception("/summarize failed")
        await update.message.reply_text("Internal error while summarizing.")
        return

    response = (
        f"Model: {result['model']}\n"
        f"Text length: {result['text_length_tokens']} tokens\n"
        f"Summary type: {result['summary_type'].capitalize()}\n"
        f"Summary length: {result['summary_length'].capitalize()}\n\n"
        f"SUMMARY:\n{result['summary']}\n\n"
        f"Generation time: {result['generation_time_seconds']}s"
    )
    await update.message.reply_text(response)


async def analyze_entities_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    try:
        params = parse_key_value_message(update.message.text)
        require_params(params, {"text"}, '/analyze_entities text="tekst" link=<true|false>')
    except CommandArgsError as exc:
        await update.message.reply_text(str(exc))
        return

    text = params["text"]
    should_link = params.get("link", "false").strip().lower() == "true"
    language = detect_language(text)

    loop = asyncio.get_event_loop()
    try:
        entities = await loop.run_in_executor(None, run_ner, "spacy", text, language)
    except NERError as exc:
        await update.message.reply_text(f"Error: {exc}")
        return
    except Exception:
        LOGGER.exception("/analyze_entities failed")
        await update.message.reply_text("Internal error while analyzing entities.")
        return

    lines = ["ENTITIES FOUND:"]
    for entity in entities:
        lines.append(f"- {entity['text']} ({entity['label']}) [{entity['start']}:{entity['end']}]")
        if should_link:
            try:
                link_result = await loop.run_in_executor(None, link_entity, entity["text"], language)
            except NELError:
                link_result = {"candidates": []}
            if link_result["candidates"]:
                best = link_result["candidates"][0]
                lines.append(f"  Wikidata: {best['id']}")
                if best.get("wikipedia_url"):
                    lines.append(f"  Wikipedia: {best['wikipedia_url']}")
            else:
                lines.append("  Wikidata: Not found")
        lines.append("")

    await update.message.reply_text("\n".join(lines).strip())


async def knowledge_graph_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    try:
        params = parse_key_value_message(update.message.text)
        require_params(params, {"text"}, '/knowledge_graph text="tekst"')
    except CommandArgsError as exc:
        await update.message.reply_text(str(exc))
        return

    text = params["text"]
    language = detect_language(text)

    loop = asyncio.get_event_loop()
    try:
        entities = await loop.run_in_executor(None, run_ner, "spacy", text, language)
    except NERError as exc:
        await update.message.reply_text(f"Error: {exc}")
        return

    relations = build_entity_relations(text, entities)
    if not relations:
        await update.message.reply_text("Not enough co-occurring entities to build a knowledge graph.")
        return

    graph_path = save_knowledge_graph(relations, PLOTS_DIR)

    lines = ["KNOWLEDGE GRAPH:"]
    for source, relation, target in relations:
        lines.append(f"{source} --{relation}--> {target}")
    await update.message.reply_text("\n".join(lines))

    if graph_path:
        with open(graph_path, "rb") as image_file:
            await update.message.reply_photo(photo=image_file)


async def language_detect_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    try:
        params = parse_key_value_message(update.message.text)
        require_params(params, {"text"}, '/language_detect text="tekst"')
    except CommandArgsError as exc:
        await update.message.reply_text(str(exc))
        return

    language = detect_language(params["text"])
    await update.message.reply_text(f"Detected language: {language}")
