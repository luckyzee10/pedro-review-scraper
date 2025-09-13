"""Sentiment classification via OpenAI's Chat Completions API.

Implements a tiny wrapper on top of the HTTP API with requests
to avoid an additional SDK dependency. Returns one of:
"Positive", "Negative", or "Neutral".
"""

from __future__ import annotations

import os
from typing import Optional

import requests


ALLOWED = {"Positive", "Negative", "Neutral"}


def classify_sentiment(
    text: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    timeout: int = 20,
) -> str:
    """Classify text sentiment via OpenAI chat completions.

    Falls back to "Neutral" if the API is not available or responds unexpectedly.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "Neutral"

    system_prompt = (
        "Classify this movie review headline or snippet as Positive, Negative, or Neutral "
        "toward the film. Respond with exactly one of those words."
    )
    user_prompt = f"Text: {text.strip()}"

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": 3,
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            return "Neutral"
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        # Normalize and validate
        upper = content.capitalize()
        if upper in ALLOWED:
            return upper
        # Common stray tokens like punctuation or explanations -> try to clean
        cleaned = upper.replace(".", "").replace("!", "").replace("\n", "").strip()
        if cleaned in ALLOWED:
            return cleaned
        return "Neutral"
    except Exception:
        return "Neutral"


def extract_market_title(
    text: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    timeout: int = 20,
) -> Optional[str]:
    """Use OpenAI to extract the referenced movie/series title from market text.

    Returns a cleaned title string or None if not clear. The model is instructed
    to output ONLY the title, no quotes or extra text.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    system_prompt = (
        "You extract the referenced film or TV title from market questions.\n"
        "If the text is about a Rotten Tomatoes or Tomatometer score, identify the specific title.\n"
        "Output EXACTLY the title text only, no quotes, no extra words. If unknown, output UNKNOWN."
    )
    user_prompt = f"Text: {text.strip()}"
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": 20,
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not content or content.upper() == "UNKNOWN":
            return None
        # Trim quotes and whitespace
        content = content.strip().strip('"').strip("'").strip()
        # Basic length guard
        if len(content) > 120:
            return None
        return content
    except Exception:
        return None
