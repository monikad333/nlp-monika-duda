from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from collections import Counter
from typing import Any

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from common.classifier import (
    VALID_CLASSES,
    append_records,
    load_sentences,
    normalize_class_name,
    predict_class,
)
from common.nlp_task import ensure_nltk_resources, run_corpus_stats, run_full_pipeline, run_single_task, split_sentences
from lab02.lab2_experiment import ClassifyArgsError, parse_classify_args, run_classify_experiment
from lab03.commands import (
    add_sentiment_command,
    compare_command,
    models_command,
    sentiment_command,
    train_command,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

DATA_PATH = os.getenv("SENTENCES_PATH", "lab01/sentences.json")
PLOTS_DIR = os.getenv("PLOTS_DIR", "lab01/plots")
LAB2_PLOTS_DIR = os.getenv("LAB2_PLOTS_DIR", "lab02/plots")
LAB2_RESULTS_PATH = os.getenv("LAB2_RESULTS_PATH", "lab02/lab2results.csv")
LAB2_MAX_PLOTS_TO_SEND = int(os.getenv("LAB2_MAX_PLOTS_TO_SEND", "6"))


def _parse_quoted_args(message_text: str | None) -> list[str] | None:
    if not message_text:
        return None

    try:
        return shlex.split(message_text)
    except ValueError:
        return None


def _format_value(value: Any, max_chars: int = 3200) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)

    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _format_top_items(mapping: dict[str, Any], limit: int = 10) -> list[str]:
    items = list(mapping.items())[:limit]
    return [f"- {key}: {value}" for key, value in items]


async def _send_plot_images(update: Update, plot_paths: list[str]) -> None:
    if not update.message:
        return

    for plot_path in plot_paths:
        if not os.path.exists(plot_path):
            continue
        with open(plot_path, "rb") as image_file:
            await update.message.reply_photo(photo=image_file)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message:
        return

    message = (
        "NLP bot commands:\n"
        "/task <task_name> \"text\" \"class\"\n"
        "/full_pipeline \"text\" \"class\"\n"
        "/classifier \"text\"\n"
        "/stats\n"
        "/classify dataset=<name> method=<model|all> gridsearch=<true/false> run=<n>\n"
        '/sentiment method=<rule|nb|rf|transformer|textblob|stanza|simplernn|lstm|gru> text="tekst"\n'
        "/train model=<simplernn|lstm|gru> dataset=<amazon|imdb|custom>\n"
        "/compare dataset=<amazon|imdb|custom> methods=<lista_metod>\n"
        '/add_sentiment "tekst" "etykieta"\n'
        "/models\n\n"
        "Allowed classes: pozytywny, neutralny, negatywny"
    )
    await update.message.reply_text(message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start_command(update, context)


async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message:
        return

    args = _parse_quoted_args(update.message.text)
    if not args or len(args) != 4:
        await update.message.reply_text('Usage: /task <task_name> "text" "class"')
        return

    _, task_name, text, class_name = args
    class_name = normalize_class_name(class_name)

    if class_name not in VALID_CLASSES:
        await update.message.reply_text("Invalid class. Use: pozytywny, neutralny, negatywny")
        return

    if not text.strip():
        await update.message.reply_text("Text cannot be empty.")
        return

    try:
        result = run_single_task(task_name, text, plots_dir=PLOTS_DIR)
        append_records([{"text": text.strip(), "class": class_name}], data_path=DATA_PATH)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    except Exception:
        LOGGER.exception("/task failed")
        await update.message.reply_text("Internal error while handling /task.")
        return

    response_lines = [
        f"Task: {result['task']}",
        "Result:",
        _format_value(result.get("result")),
    ]

    await update.message.reply_text("\n".join(response_lines))
    if result.get("plots"):
        await _send_plot_images(update, result["plots"])


async def full_pipeline_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message:
        return

    args = _parse_quoted_args(update.message.text)
    if not args or len(args) != 3:
        await update.message.reply_text('Usage: /full_pipeline "text" "class"')
        return

    _, text, class_name = args
    class_name = normalize_class_name(class_name)

    if class_name not in VALID_CLASSES:
        await update.message.reply_text("Invalid class. Use: pozytywny, neutralny, negatywny")
        return

    if not text.strip():
        await update.message.reply_text("Text cannot be empty.")
        return

    try:
        pipeline_result = run_full_pipeline(text, plots_dir=PLOTS_DIR)
        sentences = split_sentences(text)
        sentence_records = [{"text": sentence, "class": class_name} for sentence in sentences]
        append_records(sentence_records, data_path=DATA_PATH)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    except Exception:
        LOGGER.exception("/full_pipeline failed")
        await update.message.reply_text("Internal error while handling /full_pipeline.")
        return

    stats = pipeline_result["stats"]
    bow_top = sorted(pipeline_result["bag_of_words"].items(), key=lambda item: item[1], reverse=True)[:10]
    tfidf_top = sorted(pipeline_result["tfidf"].items(), key=lambda item: item[1], reverse=True)[:10]

    response = [
        "Full pipeline completed.",
        f"Saved records: {len(sentences)}",
        f"Cleaned text: {pipeline_result['cleaned_text']}",
        f"Token count: {stats['token_count']}",
        f"Unique token count: {stats['unique_token_count']}",
        "Top BoW terms:",
        "\n".join(_format_top_items(dict(bow_top))),
        "Top TF-IDF terms:",
        "\n".join(_format_top_items(dict(tfidf_top))),
    ]

    await update.message.reply_text("\n".join(response))
    await _send_plot_images(update, pipeline_result["plots"])


async def classifier_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message:
        return

    args = _parse_quoted_args(update.message.text)
    if not args or len(args) != 2:
        await update.message.reply_text('Usage: /classifier "text"')
        return

    _, text = args
    if not text.strip():
        await update.message.reply_text("Text cannot be empty.")
        return

    try:
        records = load_sentences(DATA_PATH)
        if not records:
            await update.message.reply_text("No training data found in sentences.json.")
            return

        predicted_label = predict_class(records, text, use_preprocessing=True)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    except Exception:
        LOGGER.exception("/classifier failed")
        await update.message.reply_text("Internal error while handling /classifier.")
        return

    class_counts = Counter(record["class"] for record in records)
    response = [
        f"Predicted class: {predicted_label}",
        f"Training samples: {len(records)}",
        f"Class distribution: {dict(class_counts)}",
    ]
    await update.message.reply_text("\n".join(response))


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message:
        return

    args = _parse_quoted_args(update.message.text)
    if not args or len(args) != 1:
        await update.message.reply_text("Usage: /stats")
        return

    try:
        records = load_sentences(DATA_PATH)
        if not records:
            await update.message.reply_text("No data found in sentences.json.")
            return

        stats_result = run_corpus_stats(records, plots_dir=PLOTS_DIR)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    except Exception:
        LOGGER.exception("/stats failed")
        await update.message.reply_text("Internal error while handling /stats.")
        return

    top_tokens = stats_result["top_tokens"][:10]
    top_tokens_text = ", ".join([f"{token}:{count}" for token, count in top_tokens])
    unique_tokens_sample = ", ".join(stats_result["unique_tokens"][:20]) or "none"
    unique_bigrams_sample = ", ".join(stats_result["unique_bigrams"][:10]) or "none"
    unique_trigrams_sample = ", ".join(stats_result["unique_trigrams"][:10]) or "none"

    response_lines = [
        "Corpus stats:",
        f"All tokens: {stats_result['token_count']}",
        f"Unique tokens: {len(stats_result['unique_tokens'])}",
        f"Unique tokens sample: {unique_tokens_sample}",
        f"Unique 2-grams: {len(stats_result['unique_bigrams'])}",
        f"Unique 2-grams sample: {unique_bigrams_sample}",
        f"Unique 3-grams: {len(stats_result['unique_trigrams'])}",
        f"Unique 3-grams sample: {unique_trigrams_sample}",
        f"Top tokens: {top_tokens_text if top_tokens_text else 'none'}",
        f"Class counts: {stats_result['class_counts']}",
    ]

    await update.message.reply_text("\n".join(response_lines))
    await _send_plot_images(update, stats_result["plots"])


async def classify_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    args_text = update.message.text.partition(" ")[2]

    try:
        params = parse_classify_args(args_text)
        gridsearch = params["gridsearch"].strip().lower() == "true"
        run_count = int(params["run"])
    except (ClassifyArgsError, ValueError) as exc:
        await update.message.reply_text(
            f"{exc}\nUsage: /classify dataset=<name> method=<model|all> gridsearch=<true/false> run=<n>"
        )
        return

    await update.message.reply_text(
        f"Starting experiment: dataset={params['dataset']} method={params['method']} "
        f"gridsearch={gridsearch} run={run_count}. This may take a while..."
    )

    loop = asyncio.get_event_loop()
    try:
        summary = await loop.run_in_executor(
            None,
            run_classify_experiment,
            params["dataset"],
            params["method"],
            gridsearch,
            run_count,
            LAB2_PLOTS_DIR,
            LAB2_RESULTS_PATH,
        )
    except (ClassifyArgsError, ValueError) as exc:
        await update.message.reply_text(str(exc))
        return
    except Exception:
        LOGGER.exception("/classify failed")
        await update.message.reply_text("Internal error while running the experiment.")
        return

    lines = ["Averaged results (embedding | model | accuracy | macro_f1):"]
    for row in summary["averaged"]:
        lines.append(f"{row['embedding']} | {row['model']} | {row['accuracy']} | {row['macro_f1']}")
    lines.append(f"Full results saved to: {summary['results_path']}")
    lines.append(f"Generated {len(summary['generated_files'])} files in '{LAB2_PLOTS_DIR}/'.")

    await update.message.reply_text("\n".join(lines))
    image_paths = [path for path in summary["generated_files"] if path.lower().endswith(".png")]
    await _send_plot_images(update, image_paths[:LAB2_MAX_PLOTS_TO_SEND])


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    LOGGER.error("Exception while handling update %s", update, exc_info=context.error)


def main() -> None:
    ensure_nltk_resources()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in environment variables.")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("task", task_command))
    app.add_handler(CommandHandler("full_pipeline", full_pipeline_command))
    app.add_handler(CommandHandler("classifier", classifier_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("classify", classify_command))
    app.add_handler(CommandHandler("sentiment", sentiment_command))
    app.add_handler(CommandHandler("train", train_command))
    app.add_handler(CommandHandler("compare", compare_command))
    app.add_handler(CommandHandler("add_sentiment", add_sentiment_command))
    app.add_handler(CommandHandler("models", models_command))
    app.add_error_handler(error_handler)

    # Python 3.14 no longer creates a default loop for the main thread.
    # python-telegram-bot 21.x expects one to exist when run_polling() starts.
    asyncio.set_event_loop(asyncio.new_event_loop())
    app.run_polling()


if __name__ == "__main__":
    main()
