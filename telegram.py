"""Telegram notification helper.

Uses python-telegram-bot when available. Handles both legacy sync and modern
async APIs. Falls back to Telegram HTTP API via requests if needed.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Optional

import requests

try:
    from telegram import Bot  # type: ignore
except Exception:  # pragma: no cover - library may be absent or mismatched
    Bot = None  # type: ignore


def send_telegram_message(
    token: str,
    chat_id: str,
    text: str,
    disable_preview: bool = True,
    timeout: int = 15,
) -> bool:
    """Send a message to Telegram.

    Attempts with python-telegram-bot first (supports both sync and async Bot),
    then falls back to HTTP API using requests. Returns True on success.
    """
    # Try python-telegram-bot if available
    if Bot is not None:
        try:
            bot = Bot(token=token)
            if inspect.iscoroutinefunction(getattr(bot, "send_message", None)):
                async def _run_async() -> None:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=text,
                        disable_web_page_preview=disable_preview,
                    )

                asyncio.run(_run_async())
            else:
                bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    disable_web_page_preview=disable_preview,
                )
            return True
        except Exception:
            # fall through to HTTP API
            pass

    # Fallback: direct HTTP API
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": str(bool(disable_preview)).lower(),
            },
            timeout=timeout,
        )
        return resp.ok and resp.json().get("ok", False)
    except Exception:
        return False

