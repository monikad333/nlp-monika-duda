from __future__ import annotations

import asyncio
import logging
import os
import time

from telegram import Update
from telegram.ext import ContextTypes

from lab05.agent import AgentError, run_agent
from lab05.config import IMAGES_DIR
from lab05.tool_schemas import TOOL_SCHEMAS
from lab05.tools import ToolError, analyze_image

LOGGER = logging.getLogger(__name__)

HELP_TEXT = (
    "Lab05 commands:\n"
    '/ask text="pytanie"\n'
    "/tools\n"
    "Wyslij zdjecie (z opcjonalnym podpisem) - bot automatycznie je opisze (Vision tool).\n\n"
    "Examples:\n"
    '/ask text="Jaka jest pogoda w Warszawie?"\n'
    '/ask text="Porownaj pogode w Warszawie i Paryzu"\n'
    '/ask text="Ile to jest 15 * 23 + 7?"\n'
    '/ask text="Kto jest CEO Tesli?"'
)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.message:
        await update.message.reply_text(HELP_TEXT)


async def tools_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message:
        return

    lines = ["Available tools:"]
    for tool in TOOL_SCHEMAS:
        function_info = tool["function"]
        lines.append(f"- {function_info['name']}: {function_info['description']}")

    await update.message.reply_text("\n".join(lines))


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.text:
        return

    text = update.message.text.partition(" ")[2].strip()
    if text.startswith("text="):
        text = text[len("text=") :].strip()
    text = text.strip('"')

    if not text:
        await update.message.reply_text('Usage: /ask text="pytanie"')
        return

    await update.message.reply_text("Thinking, this may take a while (local LLM + tools)...")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, run_agent, text)
    except AgentError as exc:
        await update.message.reply_text(f"Error: {exc}")
        return
    except Exception:
        LOGGER.exception("/ask failed")
        await update.message.reply_text("Internal error while running the agent.")
        return

    lines = [result["answer"]]
    if result["tool_calls"]:
        lines.append("")
        lines.append("Tools used:")
        for call in result["tool_calls"]:
            lines.append(f"- {call['tool']}({call['arguments']}) -> {call['result']}")
    lines.append(f"\nElapsed: {result['elapsed_seconds']}s")

    await update.message.reply_text("\n".join(lines))


async def photo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if not update.message or not update.message.photo:
        return

    os.makedirs(IMAGES_DIR, exist_ok=True)
    photo = update.message.photo[-1]
    photo_file = await photo.get_file()

    image_path = os.path.join(IMAGES_DIR, f"photo_{int(time.time())}.jpg")
    await photo_file.download_to_drive(image_path)

    await update.message.reply_text("Analyzing image with the Vision tool...")

    loop = asyncio.get_event_loop()
    try:
        description = await loop.run_in_executor(None, analyze_image, image_path)
    except ToolError as exc:
        await update.message.reply_text(f"Error: {exc}")
        return
    except Exception:
        LOGGER.exception("Vision analysis failed")
        await update.message.reply_text("Internal error while analyzing the image.")
        return

    await update.message.reply_text(f"Vision tool result:\n{description}")
