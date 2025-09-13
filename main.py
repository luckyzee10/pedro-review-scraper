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
from telegram import send_telegram_message, fetch_updates, get_me, delete_webhook


DB_PATH = os.getenv("REVIEW_DB_PATH", "reviews.db")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "120"))  # seconds
MIN_REVIEWS_FOR_PERCENT = int(os.getenv("MIN_REVIEWS_FOR_PERCENT", "3"))


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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kv (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
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


def _calc_freshness_percent(agg: Dict[str, int], min_count: int = MIN_REVIEWS_FOR_PERCENT) -> str:
    pos = int(agg.get("Positive", 0))
    neg = int(agg.get("Negative", 0))
    denom = pos + neg
    if denom < min_count:
        return "N/A"
    pct = round(100 * (pos / denom))
    return f"{pct}%"


def format_message(outlet: str, headline: str, sentiment: str, agg: Dict[str, int], movie: str) -> str:
    pos = agg.get("Positive", 0)
    neg = agg.get("Negative", 0)
    neu = agg.get("Neutral", 0)
    freshness = _calc_freshness_percent(agg)
    return (
        f"🎬 New Review from {outlet}\n"
        f"\"{headline}\" → {sentiment}\n"
        f"Aggregate Sentiment for {movie}: {pos}P / {neg}N / {neu}M\n"
        f"Tomatometer-like: {freshness} Fresh"
    )


def _kv_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _kv_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def _format_movie_stats_row(movie: str, pos: int, neg: int, neu: int) -> str:
    denom = pos + neg
    fresh = f"{round(100 * (pos/denom))}%" if denom >= MIN_REVIEWS_FOR_PERCENT else "N/A"
    total = pos + neg + neu
    return f"- {movie}: {pos}P/{neg}N/{neu}M • {fresh} • {total} reviews"


def _send_batched_message(token: str, chat_id: str, lines: list[str], max_len: int = 3500) -> None:
    buf: list[str] = []
    cur_len = 0
    for line in lines:
        if cur_len + len(line) + 1 > max_len and buf:
            send_telegram_message(token, chat_id, "\n".join(buf))
            buf = []
            cur_len = 0
        buf.append(line)
        cur_len += len(line) + 1
    if buf:
        send_telegram_message(token, chat_id, "\n".join(buf))


def handle_telegram_commands(
    conn: sqlite3.Connection,
    token: str,
    bot_username: str | None,
) -> None:
    # Offset for getUpdates
    off_s = _kv_get(conn, "tg_offset")
    offset = int(off_s) if (off_s and off_s.isdigit()) else None
    updates = fetch_updates(token, offset=offset, timeout=0)
    if not updates:
        return

    last_update_id = None
    for u in updates:
        last_update_id = u.get("update_id")
        msg = u.get("message") or u.get("channel_post")
        if not msg:
            continue
        text = str(msg.get("text") or "").strip()
        if not text:
            continue
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id"))

        # Normalize command
        tlow = text.lower()
        candidates = ["/status", "/movies"]
        if bot_username:
            candidates.extend([f"/status@{bot_username.lower()}", f"/movies@{bot_username.lower()}"])
        if tlow not in candidates:
            continue

        # Build stats
        rows = conn.execute(
            """
            SELECT movie,
                   SUM(CASE WHEN sentiment='Positive' THEN 1 ELSE 0 END) AS pos,
                   SUM(CASE WHEN sentiment='Negative' THEN 1 ELSE 0 END) AS neg,
                   SUM(CASE WHEN sentiment='Neutral' THEN 1 ELSE 0 END)  AS neu,
                   COUNT(*) AS total
            FROM reviews
            GROUP BY movie
            ORDER BY total DESC, movie ASC
            LIMIT 100
            """
        ).fetchall()

        if not rows:
            send_telegram_message(token, chat_id, "No reviews yet. Check back soon!")
            continue

        lines: list[str] = ["📊 Movies Summary (Top 100):"]
        for movie, pos, neg, neu, total in rows:
            lines.append(_format_movie_stats_row(str(movie), int(pos or 0), int(neg or 0), int(neu or 0)))
        # Send possibly in multiple messages
        _send_batched_message(token, chat_id, lines)

    if last_update_id is not None:
        _kv_set(conn, "tg_offset", str(int(last_update_id) + 1))


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

    # Prepare Telegram getUpdates mode (no webhook)
    if telegram_token:
        delete_webhook(telegram_token)
    me = get_me(telegram_token) if telegram_token else {}
    bot_username = me.get("username") if isinstance(me, dict) else None

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

        # Handle Telegram commands quickly between cycles
        try:
            if telegram_token:
                handle_telegram_commands(conn, telegram_token, bot_username)
        except Exception as e:
            print(f"[!] Error handling Telegram commands: {e}")

        elapsed = time.time() - start
        print(f"[•] Cycle complete: {new_count} new reviews. Slept: {int(elapsed)}s")

        # Sleep remaining time of the 2-minute window
        remaining = max(0.0, POLL_SECONDS - elapsed)
        time.sleep(remaining)

    print("[+] Stopping. Goodbye!")


if __name__ == "__main__":
    main()
