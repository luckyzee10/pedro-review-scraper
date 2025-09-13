"""Main runner for the movie review sentiment notifier.

Responsibilities:
- Load config from environment via dotenv
- Initialize SQLite (schema: reviews(outlet, movie, headline, sentiment, timestamp))
- Poll feeds every 2 minutes, avoiding duplicates
- Classify sentiment with OpenAI
- Aggregate in-memory counts per movie and send Telegram notifications
"""

from __future__ import annotations

import os
import re
import signal
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Tuple

from dotenv import load_dotenv

from scraper import FEEDS, fetch_all, ScrapedItem
from sentiment import classify_sentiment
from telegram import send_telegram_message


DB_PATH = os.getenv("REVIEW_DB_PATH", "reviews.db")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "120"))  # seconds


def init_db(path: str = DB_PATH) -> sqlite3.Connection:
    # Ensure parent directory exists if a nested path is provided
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            outlet TEXT NOT NULL,
            movie TEXT NOT NULL,
            headline TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_review
        ON reviews(outlet, headline)
        """
    )
    conn.commit()
    return conn


def load_aggregate_counts(conn: sqlite3.Connection) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"Positive": 0, "Negative": 0, "Neutral": 0})
    rows = conn.execute(
        "SELECT movie, sentiment, COUNT(*) FROM reviews GROUP BY movie, sentiment"
    ).fetchall()
    for movie, sentiment, count in rows:
        if sentiment in ("Positive", "Negative", "Neutral"):
            counts[movie][sentiment] = int(count)
    return counts


_QUOTE_PATTERNS = [
    ("“", "”"),
    ("‘", "’"),
    ("\"", "\""),
    ("'", "'"),
]


def _find_quoted(text: str) -> str:
    best: str = ""
    for left, right in _QUOTE_PATTERNS:
        pattern = re.compile(re.escape(left) + r"([^" + re.escape(right) + r"]+)" + re.escape(right))
        for m in pattern.finditer(text):
            candidate = m.group(1).strip()
            if len(candidate) > len(best):
                best = candidate
    return best


def extract_movie_title(headline: str, summary: str = "") -> str:
    """Best-effort extraction of a movie title from a headline/snippet.

    Heuristics:
    - Prefer the longest quoted phrase in headline or summary.
    - If 'Review:' appears, use the phrase after the colon.
    - If headline ends with 'Review', use the phrase before 'Review'.
    - Fallback to the portion before the first ':' or ' - '.
    - Final fallback: first ~8 words of the headline.
    """
    text = headline.strip()
    quoted = _find_quoted(text) or _find_quoted(summary)
    if quoted:
        return re.sub(r"\s+", " ", quoted).strip()

    low = text.lower()
    if "review:" in low:
        # e.g., "Film Review: Movie Title" or "Review: Movie Title"
        after = text.split(":", 1)[1]
        cleaned = re.sub(r"\b(film|movie)\s+review\b:?\s*", "", after, flags=re.I).strip()
        cleaned = re.sub(r"\breview\b:?\s*", "", cleaned, flags=re.I).strip()
        return cleaned or text

    if low.endswith(" review"):
        before = re.sub(r"\breview\b$", "", text, flags=re.I).strip(" -:|\u2013")
        return before or text

    # Common separators
    for sep in (" — ", " – ", " - ", ": "):
        if sep in text:
            left, right = text.split(sep, 1)
            # If the left looks like a prefix (contains 'review'), take the right side
            if "review" in left.lower():
                return right.strip()
            return left.strip()

    # Fallback: first 8 words
    words = text.split()
    return " ".join(words[:8]).strip()


def row_exists(conn: sqlite3.Connection, outlet: str, headline: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM reviews WHERE outlet = ? AND headline = ? LIMIT 1",
        (outlet, headline),
    )
    return cur.fetchone() is not None


def insert_review(
    conn: sqlite3.Connection,
    outlet: str,
    movie: str,
    headline: str,
    sentiment: str,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO reviews(outlet, movie, headline, sentiment, timestamp) VALUES(?,?,?,?,?)",
        (outlet, movie, headline, sentiment, ts),
    )
    conn.commit()


def format_message(outlet: str, headline: str, sentiment: str, agg: Dict[str, int], movie: str) -> str:
    pos = agg.get("Positive", 0)
    neg = agg.get("Negative", 0)
    neu = agg.get("Neutral", 0)
    return (
        f"🎬 New Review from {outlet}\n"
        f"\"{headline}\" → {sentiment}\n"
        f"Aggregate Sentiment for {movie}: {pos}P / {neg}N / {neu}M"
    )


def main() -> None:
    load_dotenv()  # Load .env file if present

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not telegram_token or not telegram_chat_id:
        print("[!] TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
        sys.exit(1)

    conn = init_db(DB_PATH)
    counts = load_aggregate_counts(conn)
    print("[+] Loaded aggregate counts for", len(counts), "movies")

    stop = False

    def _handle_sigint(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    print("[+] Starting poll loop. Interval:", POLL_SECONDS, "seconds")
    while not stop:
        start = time.time()
        try:
            items = fetch_all(FEEDS)
        except Exception as e:
            print(f"[!] Error fetching feeds: {e}")
            items = []

        new_count = 0
        for it in items:
            outlet = it.outlet
            headline = it.title.strip()
            # Duplicate avoidance BEFORE spending on classification
            if row_exists(conn, outlet, headline):
                continue

            # Determine movie title
            movie = extract_movie_title(headline, it.summary)

            # Run sentiment classification
            text_for_model = f"{headline}\n\n{it.summary}".strip()
            sentiment = classify_sentiment(text_for_model, api_key=openai_api_key)
            if sentiment not in ("Positive", "Negative", "Neutral"):
                sentiment = "Neutral"

            # Insert into DB and update in-memory counts
            insert_review(conn, outlet, movie, headline, sentiment)

            c = counts.setdefault(movie, {"Positive": 0, "Negative": 0, "Neutral": 0})
            c[sentiment] += 1

            # Send Telegram notification
            msg = format_message(outlet, headline, sentiment, c, movie)
            ok = send_telegram_message(telegram_token, telegram_chat_id, msg)
            if not ok:
                print("[!] Failed to send Telegram message for:", headline)
            new_count += 1

        elapsed = time.time() - start
        print(f"[•] Cycle complete: {new_count} new reviews. Slept: {int(elapsed)}s")

        # Sleep remaining time of the 2-minute window
        remaining = max(0.0, POLL_SECONDS - elapsed)
        time.sleep(remaining)

    print("[+] Stopping. Goodbye!")


if __name__ == "__main__":
    main()
