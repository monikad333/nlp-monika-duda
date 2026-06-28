from __future__ import annotations

import asyncio
import logging
import shlex

from telegram import Update
from telegram.ext import ContextTypes

from lab03.compare import run_compare
from lab03.config import PLOTS_DIR, VALID_LABELS
from lab03.datasets import append_to_custom_dataset
from lab03.model_loader import list_available_models
from lab03.sentiment_methods import SentimentMethodError, run_sentiment_method
from lab03.training import train_sequential_model
from lab03.utils import CommandArgsError, is_valid_label, normalize_label, parse_key_value_args, require_params
from lab03.visualizations import save_class_distribution_plot, save_wordcloud_for_class

LOGGER = logging.getLogger(__name__)

HELP_TEXT = (
    "Lab03 commands:\n"
    '/sentiment method=<rule|nb|rf|transformer|textblob|stanza|simplernn|lstm|gru> text="tekst"\n'
    "/train model=<simplernn|lstm|gru> dataset=<amazon|imdb|custom>\n"
    "/compare dataset=<amazon|imdb|custom> methods=<lista_metod>\n"
    '/add_sentiment "tekst" "etykieta"\n'
    "/models\n\n"
    "Examples:\n"
    '/sentiment method=rule text="To był naprawdę świetny film"\n'
    "/train model=lstm dataset=custom\n"
    "/compare dataset=custom methods=rule,nb,textblob\n"
    '/add_sentiment "Obsługa była poprawna" "neutralny"\n\n'
    f"Allowed labels: {', '.join(VALID_LABELS)}"
)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.message:
        await update.message.reply_text(HELP_TEXT)


def _parse_sentiment_args(message_text: str) -> dict[str, str]:
    args_text = message_text.partition(" ")[2]
    tokens = shlex.split(args_text)

    params: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            raise CommandArgsError(f"Invalid argument '{token}'. Expected key=value.")
        key, value = token.split("=", 1)
        params[key.strip().lower()] = value.strip()
    return params


async def sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    try:
        params = _parse_sentiment_args(update.message.text)
        require_params(params, {"method", "text"}, '/sentiment method=<metoda> text="tekst"')
    except CommandArgsError as exc:
        await update.message.reply_text(str(exc))
        return

    method = params["method"]
    text = params["text"]

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_sentiment_method, method, text)
    except SentimentMethodError as exc:
        await update.message.reply_text(f"Error: {exc}")
        return
    except Exception:
        LOGGER.exception("/sentiment failed")
        await update.message.reply_text("Internal error while running sentiment analysis.")
        return

    score_line = f"Pewnosc: {result['score']}" if result.get("score") is not None else "Pewnosc: n/a"
    response = (
        f"Model: {result['model']}\n"
        f"Predykcja: {result['label']}\n"
        f"{score_line}"
    )
    await update.message.reply_text(response)


async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    args_text = update.message.text.partition(" ")[2]

    try:
        params = parse_key_value_args(args_text)
        require_params(params, {"model", "dataset"}, "/train model=<simplernn|lstm|gru> dataset=<amazon|imdb|custom>")
    except CommandArgsError as exc:
        await update.message.reply_text(str(exc))
        return

    await update.message.reply_text(
        f"Starting training: model={params['model']} dataset={params['dataset']}. This may take a while..."
    )

    loop = asyncio.get_event_loop()
    try:
        summary = await loop.run_in_executor(None, train_sequential_model, params["model"], params["dataset"])
    except (ValueError, FileNotFoundError) as exc:
        await update.message.reply_text(f"Error: {exc}")
        return
    except Exception:
        LOGGER.exception("/train failed")
        await update.message.reply_text("Internal error while training the model.")
        return

    response = (
        f"Training finished: {summary['model']} on {summary['dataset']}\n"
        f"Epochs: {summary['epochs_ran']}/{summary['epochs_requested']} "
        f"(elapsed: {summary['elapsed_seconds']}s)\n"
        f"Validation accuracy: {summary['final_val_accuracy']}\n"
        f"Validation loss: {summary['final_val_loss']}\n"
        f"Model: {summary['artifact_paths']['model']}\n"
        f"Tokenizer: {summary['artifact_paths']['tokenizer']}\n"
        f"Label encoder: {summary['artifact_paths']['label_encoder']}"
    )
    await update.message.reply_text(response)

    if update.message and summary.get("history_plot_path"):
        with open(summary["history_plot_path"], "rb") as image_file:
            await update.message.reply_photo(photo=image_file)


async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    args_text = update.message.text.partition(" ")[2]

    try:
        params = parse_key_value_args(args_text)
        require_params(params, {"dataset", "methods"}, "/compare dataset=<amazon|imdb|custom> methods=<lista_metod>")
    except CommandArgsError as exc:
        await update.message.reply_text(str(exc))
        return

    await update.message.reply_text(
        f"Comparing methods {params['methods']} on dataset={params['dataset']}. This may take a while..."
    )

    loop = asyncio.get_event_loop()
    try:
        summary = await loop.run_in_executor(None, run_compare, params["dataset"], params["methods"])
    except (ValueError, CommandArgsError, FileNotFoundError) as exc:
        await update.message.reply_text(f"Error: {exc}")
        return
    except Exception:
        LOGGER.exception("/compare failed")
        await update.message.reply_text("Internal error while comparing methods.")
        return

    if not summary["rows"]:
        await update.message.reply_text(f"No method produced results. Skipped: {summary['skipped']}")
        return

    lines = ["Method | accuracy | precision | recall | macro_f1"]
    for row in summary["rows"]:
        lines.append(f"{row['method']} | {row['accuracy']} | {row['precision']} | {row['recall']} | {row['macro_f1']}")
    if summary["skipped"]:
        lines.append(f"Skipped: {summary['skipped']}")

    await update.message.reply_text("\n".join(lines))

    if summary.get("compare_plot_path"):
        with open(summary["compare_plot_path"], "rb") as image_file:
            await update.message.reply_photo(photo=image_file)


async def add_sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    try:
        args = shlex.split(update.message.text)
    except ValueError:
        args = None

    if not args or len(args) != 3:
        await update.message.reply_text('Usage: /add_sentiment "tekst" "etykieta"')
        return

    _, text, label = args
    normalized_label = normalize_label(label)

    if not text.strip():
        await update.message.reply_text("Text cannot be empty.")
        return

    if not is_valid_label(normalized_label):
        await update.message.reply_text(f"Invalid label. Allowed: {', '.join(VALID_LABELS)}")
        return

    append_to_custom_dataset(text.strip(), normalized_label)
    await update.message.reply_text(f"Saved: \"{text.strip()}\" -> {normalized_label}")


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message:
        return

    models = list_available_models()
    if not models:
        await update.message.reply_text("No trained models found. Use /train to create one.")
        return

    lines = ["Available models:"]
    for entry in models:
        lines.append(
            f"- {entry['model']} (dataset={entry['dataset']}) "
            f"tokenizer={'yes' if entry['has_tokenizer'] else 'no'} "
            f"label_encoder={'yes' if entry['has_label_encoder'] else 'no'}"
        )
    await update.message.reply_text("\n".join(lines))
