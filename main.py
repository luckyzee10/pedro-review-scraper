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
from movie_meta import ensure_release_date, ensure_tables as ensure_movie_tables


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
    # Ensure movie metadata table exists
    try:
        ensure_movie_tables(conn)
    except Exception:
        pass
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


def _smart_titlecase(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().title()


def _normalize_title_from_url(url: str) -> str | None:
    try:
        from urllib.parse import urlparse

        p = urlparse(url)
        host = (p.netloc or "").lower()
        segs = [s for s in (p.path or "").split("/") if s]
        if not segs:
            return None
        slug = segs[-1]
        # Guardian pattern: .../<year>/<mon>/<day>/<slug>
        # or .../film/<yyyy>/<mm>/<dd>/<slug>
        if "theguardian.com" in host:
            # If last seg is numeric (date), take the previous
            if re.fullmatch(r"\d{4}|\d{2}", slug):
                slug = segs[-2] if len(segs) >= 2 else slug
            # Drop common suffix starting at -review
            slug = re.sub(r"-review(?:-.+)?$", "", slug, flags=re.I)
        else:
            # Generic cleanup: drop trailing -review and file extensions
            slug = re.sub(r"\.(html?|php|asp|aspx)$", "", slug, flags=re.I)
            slug = re.sub(r"-review(?:-.+)?$", "", slug, flags=re.I)

        # Replace separators, remove leftover dashes/underscores
        title = re.sub(r"[-_]+", " ", slug).strip()
        # If too short or empty, try previous segment
        if len(title) < 2 and len(segs) >= 2:
            title = re.sub(r"[-_]+", " ", segs[-2]).strip()
        # Capitalize words
        if title:
            return _smart_titlecase(title)
        return None
    except Exception:
        return None


def extract_movie_title(headline: str, summary: str = "", link: str | None = None) -> str:
    """Best-effort extraction of a movie title from a headline/snippet.

    Heuristics:
    - Prefer the longest quoted phrase in headline or summary.
    - If 'Review:' appears, use the phrase after the colon.
    - If headline ends with 'Review', use the phrase before 'Review'.
    - Fallback to the portion before the first ':' or ' - '.
    - Final fallback: first ~8 words of the headline.
    """
    text = headline.strip()
    # Prefer explicit Guardian-style normalization from URL if provided
    if link:
        norm = _normalize_title_from_url(link)
        if norm:
            text = norm
    # Fallback: if headline is a URL, try to recover title from it
    elif re.match(r"https?://", text, flags=re.I):
        norm = _normalize_title_from_url(text)
        if norm:
            text = norm
    quoted = _find_quoted(text) or _find_quoted(summary)
    if quoted:
        return _smart_titlecase(re.sub(r"\s+", " ", quoted).strip())

    low = text.lower()
    if "review:" in low:
        # e.g., "Film Review: Movie Title" or "Review: Movie Title"
        after = text.split(":", 1)[1]
        cleaned = re.sub(r"\b(film|movie)\s+review\b:?\s*", "", after, flags=re.I).strip()
        cleaned = re.sub(r"\breview\b:?\s*", "", cleaned, flags=re.I).strip()
        return _smart_titlecase(cleaned or text)

    if low.endswith(" review"):
        before = re.sub(r"\breview\b$", "", text, flags=re.I).strip(" -:|\u2013")
        return _smart_titlecase(before or text)

    # Common separators
    for sep in (" — ", " – ", " - ", ": "):
        if sep in text:
            left, right = text.split(sep, 1)
            # If the left looks like a prefix (contains 'review'), take the right side
            if "review" in left.lower():
                return _smart_titlecase(right.strip())
            return _smart_titlecase(left.strip())

    # Cleanup common decorations
    text = re.sub(r"\breview\b", "", text, flags=re.I)
    text = re.sub(r"\bfilm\b|\bmovie\b", "", text, flags=re.I)
    text = re.sub(r"\s+[–—-]\s+.*$", "", text).strip()

    # Fallback: first 8 words
    words = text.split()
    return _smart_titlecase(" ".join(words[:8]).strip())


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


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
    )


def format_message(outlet: str, headline: str, sentiment: str, agg: Dict[str, int], movie: str) -> str:
    pos = agg.get("Positive", 0)
    neg = agg.get("Negative", 0)
    neu = agg.get("Neutral", 0)
    freshness = _calc_freshness_percent(agg)
    outlet_h = _html_escape(outlet)
    headline_h = _html_escape(headline)
    movie_h = _html_escape(movie)
    sentiment_h = _html_escape(sentiment)
    return (
        f"🎬 New Review from <b>{outlet_h}</b>\n"
        f"“{headline_h}” → <b>{sentiment_h}</b>\n"
        f"Aggregate for <b>{movie_h}</b>: {pos}P / {neg}N / {neu}M\n"
        f"Tomatometer-like: <b>{freshness}</b> Fresh"
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


def _format_movie_stats_row(movie: str, pos: int, neg: int, neu: int, release_date: str | None = None) -> str:
    denom = pos + neg
    fresh = f"{round(100 * (pos/denom))}%" if denom >= MIN_REVIEWS_FOR_PERCENT else "N/A"
    total = pos + neg + neu
    movie_h = _html_escape(movie)
    rd = release_date or "n/a"
    rd_h = _html_escape(rd)
    return f"• <b>{movie_h}</b> ({rd_h}) — {pos}P / {neg}N / {neu}M • {fresh} • {total} reviews"


def _send_batched_message(token: str, chat_id: str, lines: list[str], max_len: int = 3500) -> None:
    buf: list[str] = []
    cur_len = 0
    for line in lines:
        if cur_len + len(line) + 1 > max_len and buf:
            send_telegram_message(token, chat_id, "\n".join(buf), parse_mode="HTML")
            buf = []
            cur_len = 0
        buf.append(line)
        cur_len += len(line) + 1
    if buf:
        send_telegram_message(token, chat_id, "\n".join(buf), parse_mode="HTML")


def handle_telegram_commands(
    conn: sqlite3.Connection,
    token: str,
    bot_username: str | None,
    tmdb_api_key: str | None = None,
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

        # Basic parsing for command + optional argument
        tlow = text.strip().lower()
        base_cmds = ["/status", "/movies", "/backfill", "/normalize"]
        bot_cmds = []
        if bot_username:
            bot_cmds = [
                f"/status@{bot_username.lower()}",
                f"/movies@{bot_username.lower()}",
                f"/backfill@{bot_username.lower()}",
                f"/normalize@{bot_username.lower()}",
            ]
        all_cmds = base_cmds + bot_cmds

        # Identify which command is used and extract argument (movie name) if provided
        used_cmd = None
        for c in all_cmds:
            if tlow.startswith(c):
                used_cmd = c
                break
        if not used_cmd:
            continue

        arg = text[len(used_cmd):].strip()

        # Backfill missing release dates on demand
        if used_cmd.startswith("/backfill"):
            updated = 0
            missing = conn.execute(
                "SELECT DISTINCT movie FROM reviews WHERE movie NOT IN (SELECT movie FROM movies WHERE COALESCE(release_date,'')!='')"
            ).fetchall()
            for (m,) in missing:
                try:
                    rd = ensure_release_date(conn, str(m), tmdb_api_key)
                    if rd:
                        updated += 1
                except Exception:
                    pass
            send_telegram_message(token, chat_id, f"Backfill complete. Updated {updated} titles.")
            continue

        # Normalize existing URL-like movie titles (Guardian-specific by default)
        if used_cmd.startswith("/normalize"):
            scope = arg.strip().lower() if arg else "guardian"
            where = "movie LIKE 'http%'"
            if scope == "guardian":
                where += " AND movie LIKE '%theguardian.com%'"

            rows = conn.execute(f"SELECT DISTINCT movie FROM reviews WHERE {where}").fetchall()
            total_rows_updated = 0
            titles_changed = 0
            preview = []
            for (old_movie,) in rows:
                new_title = _normalize_title_from_url(str(old_movie))
                if new_title and new_title != old_movie:
                    # Update all rows with this movie value
                    cur = conn.execute(
                        "UPDATE reviews SET movie=? WHERE movie=?",
                        (new_title, old_movie),
                    )
                    conn.commit()
                    changed = cur.rowcount or 0
                    total_rows_updated += changed
                    titles_changed += 1
                    if len(preview) < 10:
                        preview.append(f"• {old_movie} → {new_title}")
                    # Ensure release date for the normalized title
                    try:
                        if tmdb_api_key:
                            ensure_release_date(conn, new_title, tmdb_api_key)
                    except Exception:
                        pass

            msg_lines = [
                f"Normalization complete. Titles changed: {titles_changed}; rows updated: {total_rows_updated}.",
            ]
            if preview:
                msg_lines.append("Examples:\n" + "\n".join(_html_escape(x) for x in preview))
            send_telegram_message(token, chat_id, "\n".join(msg_lines), parse_mode="HTML")
            continue
        if arg:
            # Single-movie status
            row = conn.execute(
                """
                SELECT movie,
                       SUM(CASE WHEN sentiment='Positive' THEN 1 ELSE 0 END) AS pos,
                       SUM(CASE WHEN sentiment='Negative' THEN 1 ELSE 0 END) AS neg,
                       SUM(CASE WHEN sentiment='Neutral' THEN 1 ELSE 0 END)  AS neu,
                       COUNT(*) AS total
                FROM reviews
                WHERE LOWER(movie) LIKE LOWER(?)
                GROUP BY movie
                ORDER BY total DESC
                LIMIT 1
                """,
                (f"%{arg}%",),
            ).fetchone()

            if not row:
                arg_h = _html_escape(arg)
                send_telegram_message(token, chat_id, f"No results for ‘{arg_h}’. Try a different title.", parse_mode="HTML")
            else:
                movie, pos, neg, neu, total = row
                # Ensure/fetch release date (cached)
                rel = None
                try:
                    rel = ensure_release_date(conn, str(movie), tmdb_api_key) if tmdb_api_key else (
                        (conn.execute("SELECT release_date FROM movies WHERE movie=?", (movie,)).fetchone() or [None])[0]
                    )
                except Exception:
                    rel = (conn.execute("SELECT release_date FROM movies WHERE movie=?", (movie,)).fetchone() or [None])[0]
                msg = "📊 <b>Movie Status</b>\n" + _format_movie_stats_row(str(movie), int(pos or 0), int(neg or 0), int(neu or 0), rel)
                send_telegram_message(token, chat_id, msg, parse_mode="HTML")
            continue

        # General summary ordered by release date proximity
        rows = conn.execute(
            """
            SELECT movie,
                   SUM(CASE WHEN sentiment='Positive' THEN 1 ELSE 0 END) AS pos,
                   SUM(CASE WHEN sentiment='Negative' THEN 1 ELSE 0 END) AS neg,
                   SUM(CASE WHEN sentiment='Neutral' THEN 1 ELSE 0 END)  AS neu,
                   COUNT(*) AS total
            FROM reviews
            GROUP BY movie
            LIMIT 300
            """
        ).fetchall()

        if not rows:
            send_telegram_message(token, chat_id, "No reviews yet. Check back soon!")
            continue

        # Ensure and attach release dates (best-effort, cached)
        rel_map = {}
        for m, *_rest in rows:
            rel = None
            try:
                rel = ensure_release_date(conn, str(m), tmdb_api_key) if tmdb_api_key else None
            except Exception:
                rel = None
            if not rel:
                rel = (conn.execute("SELECT release_date FROM movies WHERE movie=?", (m,)).fetchone() or [None])[0]
            rel_map[m] = rel

        # Sort by proximity to future release date: future soonest first, then unknown, then past
        from datetime import date

        def sort_key(item):
            m, pos, neg, neu, total = item
            rd = rel_map.get(m)
            try:
                if rd:
                    y, mo, d = map(int, rd.split("-"))
                    rdate = date(y, mo, d)
                    today = date.today()
                    if rdate >= today:
                        return (0, (rdate - today).days, m.lower())
                    else:
                        return (2, (today - rdate).days, m.lower())
                else:
                    return (1, 99999, m.lower())
            except Exception:
                return (1, 99999, m.lower())

        rows_sorted = sorted(rows, key=sort_key)

        lines: list[str] = ["📊 <b>Movies Summary (Upcoming First)</b>"]
        for movie, pos, neg, neu, total in rows_sorted[:100]:
            rel = rel_map.get(movie)
            lines.append(_format_movie_stats_row(str(movie), int(pos or 0), int(neg or 0), int(neu or 0), rel))
        _send_batched_message(token, chat_id, lines)

    if last_update_id is not None:
        _kv_set(conn, "tg_offset", str(int(last_update_id) + 1))


def main() -> None:
    load_dotenv()  # Load .env file if present

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    tmdb_api_key = os.getenv("TMDB_API_KEY", "").strip()

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
            movie = extract_movie_title(headline, it.summary, it.link)

            # Try to resolve and cache release date for new movies (best-effort)
            try:
                if tmdb_api_key:
                    ensure_release_date(conn, movie, tmdb_api_key)
            except Exception:
                pass

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
                handle_telegram_commands(conn, telegram_token, bot_username, tmdb_api_key)
        except Exception as e:
            print(f"[!] Error handling Telegram commands: {e}")

        elapsed = time.time() - start
        print(f"[•] Cycle complete: {new_count} new reviews. Slept: {int(elapsed)}s")

        # Sleep remaining time of the window, but poll commands every ~5s for responsiveness
        remaining = max(0.0, POLL_SECONDS - elapsed)
        end_at = time.time() + remaining
        while not stop and time.time() < end_at:
            try:
                if telegram_token:
                    handle_telegram_commands(conn, telegram_token, bot_username, tmdb_api_key)
            except Exception as e:
                print(f"[!] Error handling Telegram commands: {e}")
            time.sleep(min(5.0, max(0.0, end_at - time.time())))

    print("[+] Stopping. Goodbye!")


if __name__ == "__main__":
    main()
