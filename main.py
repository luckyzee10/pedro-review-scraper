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
import requests

from scraper import FEEDS, fetch_all, ScrapedItem
from sentiment import classify_sentiment
from telegram import send_telegram_message, fetch_updates, get_me, delete_webhook
from movie_meta import (
    ensure_release_date,
    ensure_tables as ensure_movie_tables,
    refresh_catalog_window,
    load_catalog_index,
    match_movie_from_url,
)
from markets import (
    ensure_tables as ensure_market_tables,
    refresh_market_titles,
    load_market_index,
    load_market_canon,
)


DB_PATH = os.getenv("REVIEW_DB_PATH", "reviews.db")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "120"))  # seconds
MIN_REVIEWS_FOR_PERCENT = int(os.getenv("MIN_REVIEWS_FOR_PERCENT", "3"))
CATALOG_PAST_DAYS = int(os.getenv("CATALOG_PAST_DAYS", "5"))
CATALOG_FUTURE_DAYS = int(os.getenv("CATALOG_FUTURE_DAYS", "90"))
CATALOG_REFRESH_SECONDS = int(os.getenv("CATALOG_REFRESH_SECONDS", "21600"))  # 6h
MARKET_REFRESH_SECONDS = int(os.getenv("MARKET_REFRESH_SECONDS", "3600"))  # 1h


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
    # Ensure market titles table exists
    try:
        ensure_market_tables(conn)
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


def _slugify_title(value: str) -> str:
    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9]+", "-", v)
    v = re.sub(r"-+", "-", v).strip("-")
    return v


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
        base_cmds = [
            "/status",
            "/movies",
            "/catalog",
            "/markets",
            "/refreshmarkets",
            "/backfill",
            "/normalize",
            "/refreshcatalog",
            "/health",
            "/testapi",
        ]
        bot_cmds = []
        if bot_username:
            bot_cmds = [
                f"/status@{bot_username.lower()}",
                f"/movies@{bot_username.lower()}",
                f"/catalog@{bot_username.lower()}",
                f"/markets@{bot_username.lower()}",
                f"/refreshmarkets@{bot_username.lower()}",
                f"/backfill@{bot_username.lower()}",
                f"/normalize@{bot_username.lower()}",
                f"/refreshcatalog@{bot_username.lower()}",
                f"/health@{bot_username.lower()}",
                f"/testapi@{bot_username.lower()}",
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

        # List current market-matched titles (optional filter)
        if used_cmd.startswith("/markets"):
            markets = load_market_index(conn)
            # Build (title, source) pairs; catalog cross-check is optional here
            items = [(title, src) for (_slug, (title, src)) in markets.items()]
            if arg:
                q = arg.strip().lower()
                items = [(t, s) for (t, s) in items if q in t.lower()]
            if not items:
                # Attempt on-demand refresh when empty
                try:
                    n = refresh_market_titles(
                        conn,
                        os.getenv("KALSHI_API_KEY", "").strip(),
                        os.getenv("KALSHI_API_SECRET", "").strip(),
                        os.getenv("TMDB_API_KEY", "").strip(),
                    )
                    markets = load_market_index(conn)
                    items = [(title, src) for (_slug, (title, src)) in markets.items()]
                except Exception:
                    items = []
            if not items:
                # Provide debug counts by source to help diagnose
                try:
                    rows = conn.execute(
                        "SELECT source, COUNT(*) FROM market_titles GROUP BY source"
                    ).fetchall()
                    breakdown = ", ".join(f"{s}:{c}" for s, c in rows) or "none"
                except Exception:
                    breakdown = "unknown"
                send_telegram_message(
                    token,
                    chat_id,
                    f"No market-matched titles yet. Sources: {breakdown}. Try /refreshmarkets.",
                )
                continue
            # De-dup by title preferring polymarket label if both exist
            best = {}
            for t, s in items:
                if t not in best or best[t] != "polymarket":
                    best[t] = s
            # Sort alphabetically
            out = sorted(best.items(), key=lambda kv: kv[0].lower())
            lines = ["📈 <b>Market Titles</b>"]
            for t, s in out[:200]:
                t_h = _html_escape(t)
                lines.append(f"• <b>{t_h}</b> — {s}")
            _send_batched_message(token, chat_id, lines)
            continue

        # Force markets refresh
        if used_cmd.startswith("/refreshmarkets"):
            try:
                n = refresh_market_titles(
                    conn,
                    os.getenv("KALSHI_API_KEY", "").strip(),
                    os.getenv("KALSHI_API_SECRET", "").strip(),
                    os.getenv("TMDB_API_KEY", "").strip(),
                )
                msize = len(load_market_index(conn))
                send_telegram_message(
                    token,
                    chat_id,
                    f"Markets refreshed: upserted {n}. Total market titles now {msize}.",
                )
            except Exception as e:
                send_telegram_message(token, chat_id, f"Markets refresh failed: {e}")
            continue

        # List current catalog titles (optionally filter by substring)
        if used_cmd.startswith("/catalog"):
            catalog = load_catalog_index(conn)
            items = [(title, rd) for (_slug, (title, rd)) in catalog.items()]
            # Optional filter
            if arg:
                q = arg.strip().lower()
                items = [(t, rd) for (t, rd) in items if q in t.lower()]

            if not items:
                send_telegram_message(token, chat_id, "Catalog is empty or no matches.")
                continue

            # Sort by release date proximity (upcoming first, then unknown, then past)
            from datetime import date

            def ckey(itm):
                t, rd = itm
                try:
                    if rd:
                        y, mo, d = map(int, rd.split("-"))
                        rdate = date(y, mo, d)
                        today = date.today()
                        if rdate >= today:
                            return (0, (rdate - today).days, t.lower())
                        else:
                            return (2, (today - rdate).days, t.lower())
                    else:
                        return (1, 99999, t.lower())
                except Exception:
                    return (1, 99999, t.lower())

            items = sorted(items, key=ckey)

            lines = ["🎞️ <b>Catalog Titles (Upcoming First)</b>"]
            for t, rd in items[:200]:
                t_h = _html_escape(t)
                rd_h = _html_escape(rd or "n/a")
                lines.append(f"• <b>{t_h}</b> ({rd_h})")
            _send_batched_message(token, chat_id, lines)
            continue

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

        # Force TMDb catalog refresh
        if used_cmd.startswith("/refreshcatalog"):
            n = refresh_catalog_now(conn, tmdb_api_key)
            size = len(load_catalog_index(conn))
            send_telegram_message(token, chat_id, f"Catalog refreshed: upserted {n}. Total titles now {size}.")
            continue

        # Health check: show basic connectivity/status
        if used_cmd.startswith("/health"):
            from telegram import get_me
            me = get_me(token) or {}
            cat_size = len(load_catalog_index(conn))
            last = _kv_get(conn, "catalog_last_refresh") or "never"
            parts = [
                f"Bot: @{me.get('username','?')}",
                f"Catalog titles: {cat_size}",
                f"Last catalog refresh: {last}",
                f"DB reviews: {conn.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]}",
                f"TMDb key: {'set' if tmdb_api_key else 'missing'}",
            ]
            send_telegram_message(token, chat_id, "\n".join(parts))
            continue

        # Active API connectivity test
        if used_cmd.startswith("/testapi"):
            results = []
            # OpenAI
            try:
                ok = False
                if os.getenv("OPENAI_API_KEY", "").strip():
                    r = requests.get(
                        "https://api.openai.com/v1/models",
                        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY').strip()}"},
                        timeout=10,
                    )
                    ok = r.status_code == 200
                    results.append(f"OpenAI: {'ok' if ok else f'fail ({r.status_code})'}")
                else:
                    results.append("OpenAI: missing key")
            except Exception as e:
                results.append(f"OpenAI: error ({type(e).__name__})")
            # TMDb
            try:
                if tmdb_api_key:
                    r = requests.get(
                        "https://api.themoviedb.org/3/configuration",
                        params={"api_key": tmdb_api_key},
                        timeout=10,
                    )
                    results.append(f"TMDb: {'ok' if r.status_code == 200 else f'fail ({r.status_code})'}")
                else:
                    results.append("TMDb: missing key")
            except Exception as e:
                results.append(f"TMDb: error ({type(e).__name__})")
            # Polymarket
            try:
                r = requests.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={"limit": 5, "closed": "false", "search": "rotten"},
                    timeout=10,
                )
                ok = r.status_code == 200
                count = 0
                if ok:
                    d = r.json() or {}
                    ms = d.get("markets") or d.get("data") or []
                    if isinstance(ms, dict):
                        ms = ms.get("markets", []) or []
                    count = len(ms)
                results.append(f"Polymarket: {'ok' if ok else f'fail ({r.status_code})'} ({count})")
            except Exception as e:
                results.append(f"Polymarket: error ({type(e).__name__})")
            # Kalshi public v2
            try:
                r = requests.get(
                    "https://api.elections.kalshi.com/trade-api/v2/markets",
                    params={"limit": 5},
                    timeout=10,
                )
                ok = r.status_code == 200
                count = 0
                if ok:
                    d = r.json() or {}
                    ms = d.get("markets") or d.get("data") or []
                    if isinstance(ms, dict):
                        ms = ms.get("markets", []) or []
                    count = len(ms)
                results.append(f"Kalshi v2: {'ok' if ok else f'fail ({r.status_code})'} ({count})")
            except Exception as e:
                results.append(f"Kalshi v2: error ({type(e).__name__})")

            send_telegram_message(token, chat_id, "API Connectivity\n" + "\n".join(results))
            continue
        if arg:
            # Single-movie status
            # Choose best match from catalog titles
            catalog = load_catalog_index(conn)
            markets = load_market_index(conn)
            want = arg.strip().lower()
            chosen_slug = None
            chosen_title = None
            chosen_rd: str | None = None
            for slug, (title, rd) in catalog.items():
                if want in title.lower() and slug in markets:
                    chosen_slug, chosen_title, chosen_rd = slug, title, rd
                    break

            if not chosen_slug:
                arg_h = _html_escape(arg)
                send_telegram_message(token, chat_id, f"No results for ‘{arg_h}’. Try a different title.", parse_mode="HTML")
                continue

            # Aggregate across rows whose slug maps to chosen_slug
            rows = conn.execute(
                """
                SELECT movie,
                       SUM(CASE WHEN sentiment='Positive' THEN 1 ELSE 0 END) AS pos,
                       SUM(CASE WHEN sentiment='Negative' THEN 1 ELSE 0 END) AS neg,
                       SUM(CASE WHEN sentiment='Neutral' THEN 1 ELSE 0 END)  AS neu,
                       COUNT(*) AS total
                FROM reviews
                GROUP BY movie
                """
            ).fetchall()

            pos = neg = neu = total = 0
            for movie, p, n, m, t in rows:
                if _slugify_title(str(movie)) == chosen_slug:
                    pos += int(p or 0)
                    neg += int(n or 0)
                    neu += int(m or 0)
                    total += int(t or 0)

            rel = None
            try:
                rel = ensure_release_date(conn, chosen_title, tmdb_api_key) if tmdb_api_key else chosen_rd
            except Exception:
                rel = chosen_rd
            msg = "📊 <b>Movie Status</b>\n" + _format_movie_stats_row(chosen_title, pos, neg, neu, rel)
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

        # Filter to market-matching titles; collapse to canonical market title
        markets = load_market_index(conn)
        mcanon = load_market_canon(conn)
        agg: dict[str, dict[str, int]] = {}
        rel_map: dict[str, str | None] = {}
        for movie, p, n, m, t in rows:
            slug = _slugify_title(str(movie))
            if slug not in markets:
                continue
            # Prefer canonical title from TMDb resolution if available; else market title
            ct_tuple = mcanon.get(slug)
            if ct_tuple and (ct_tuple[0] or ct_tuple[1]):
                canon_title = ct_tuple[0] or markets[slug][0]
                rd = ct_tuple[1]
            else:
                canon_title = markets[slug][0]
                rd = None
            c = agg.setdefault(canon_title, {"Positive": 0, "Negative": 0, "Neutral": 0, "Total": 0})
            c["Positive"] += int(p or 0)
            c["Negative"] += int(n or 0)
            c["Neutral"] += int(m or 0)
            c["Total"] += int(t or 0)
            rel_map[canon_title] = rd

        if not agg:
            send_telegram_message(token, chat_id, "No market-matched reviews yet. Check back soon!")
            continue

        # Sort by proximity to future release date: future soonest first, then unknown, then past
        from datetime import date

        def sort_key(item):
            m, counts = item
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

        items = list(agg.items())
        rows_sorted = sorted(items, key=sort_key)

        lines: list[str] = ["📊 <b>Movies Summary (Upcoming First)</b>"]
        for movie, counts in rows_sorted[:100]:
            rel = rel_map.get(movie)
            lines.append(
                _format_movie_stats_row(
                    str(movie),
                    int(counts.get("Positive", 0)),
                    int(counts.get("Negative", 0)),
                    int(counts.get("Neutral", 0)),
                    rel,
                )
            )
        _send_batched_message(token, chat_id, lines)

    if last_update_id is not None:
        _kv_set(conn, "tg_offset", str(int(last_update_id) + 1))


def refresh_catalog_if_needed(conn: sqlite3.Connection, tmdb_api_key: str | None) -> Dict[str, Tuple[str, str | None]]:
    """Refresh windowed TMDb catalog on a schedule and return the index.

    If no TMDb key is present, returns whatever is in the catalog (may be empty).
    """
    from datetime import datetime, timezone

    last = _kv_get(conn, "catalog_last_refresh")
    now = datetime.now(timezone.utc)
    do_refresh = True
    if last:
        try:
            last_dt = datetime.fromisoformat(last)
            do_refresh = (now - last_dt).total_seconds() >= CATALOG_REFRESH_SECONDS
        except Exception:
            do_refresh = True

    if tmdb_api_key and do_refresh:
        try:
            n = refresh_catalog_window(
                conn,
                api_key=tmdb_api_key,
                days_past=CATALOG_PAST_DAYS,
                days_future=CATALOG_FUTURE_DAYS,
            )
            _kv_set(conn, "catalog_last_refresh", now.isoformat())
            print(f"[+] Catalog refresh: upserted {n} titles")
        except Exception as e:
            print(f"[!] Catalog refresh failed: {e}")

    return load_catalog_index(conn)


def refresh_catalog_now(conn: sqlite3.Connection, tmdb_api_key: str | None) -> int:
    if not tmdb_api_key:
        return 0
    try:
        n = refresh_catalog_window(
            conn,
            api_key=tmdb_api_key,
            days_past=CATALOG_PAST_DAYS,
            days_future=CATALOG_FUTURE_DAYS,
        )
        from datetime import datetime, timezone

        _kv_set(conn, "catalog_last_refresh", datetime.now(timezone.utc).isoformat())
        return n
    except Exception:
        return 0


def main() -> None:
    load_dotenv()  # Load .env file if present

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    tmdb_api_key = os.getenv("TMDB_API_KEY", "").strip()
    kalshi_key = os.getenv("KALSHI_API_KEY", "").strip()
    kalshi_secret = os.getenv("KALSHI_API_SECRET", "").strip()

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
    catalog_index = refresh_catalog_if_needed(conn, tmdb_api_key)
    # Initial market titles refresh and load (now primary gating)
    last_market_refresh = 0.0
    try:
        n = refresh_market_titles(conn, kalshi_key, kalshi_secret, tmdb_api_key)
        print(f"[+] Market titles refresh: upserted {n}")
    except Exception as e:
        print(f"[!] Market titles refresh failed: {e}")
    market_index = load_market_index(conn)
    market_canon = load_market_canon(conn)
    while not stop:
        start = time.time()
        # Periodically refresh catalog (e.g., every 6 hours)
        try:
            catalog_index = refresh_catalog_if_needed(conn, tmdb_api_key)
        except Exception:
            pass
        # Refresh market titles periodically (and update canon via TMDb)
        try:
            if time.time() - last_market_refresh >= MARKET_REFRESH_SECONDS:
                n = refresh_market_titles(conn, kalshi_key, kalshi_secret, tmdb_api_key)
                market_index = load_market_index(conn)
                market_canon = load_market_canon(conn)
                last_market_refresh = time.time()
                print(f"[+] Market titles refresh: upserted {n}; total {len(market_index)}")
        except Exception as e:
            print(f"[!] Market titles refresh failed: {e}")
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

            # Match movie via market titles (primary gate): find slug in URL
            link_or_title = it.link or headline
            path = link_or_title.lower()
            best_slug = None
            best_len = 0
            for slug in market_index.keys():
                if slug in path and len(slug) > best_len:
                    best_slug, best_len = slug, len(slug)
            if not best_slug:
                continue
            # Resolve display title and release date from market canon when available
            canon = market_canon.get(best_slug)
            if canon:
                canon_title, rel_date, _tid = canon
            else:
                canon_title, rel_date = None, None
            base_title = market_index.get(best_slug, (None, None))[0]
            movie = canon_title or base_title or ""
            if not movie:
                continue

            # Try to resolve and cache release date for new movies (best-effort)
            try:
                if rel_date:
                    # Cache into movies table for sorting if we already have it
                    from movie_meta import cache_release_date

                    cache_release_date(conn, movie, rel_date)
                elif tmdb_api_key:
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
