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
import email.utils as eut

from scraper import FEEDS, fetch_all, ScrapedItem
from sentiment import classify_sentiment, extract_market_title
from telegram import send_telegram_message, fetch_updates, get_me, delete_webhook
from movie_meta import (
    ensure_release_date,
    ensure_tables as ensure_movie_tables,
    refresh_catalog_window,
    load_catalog_index,
    match_movie_from_url,
    fetch_tmdb_by_id,
)
import markets as markets_mod
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ignored (
            outlet TEXT NOT NULL,
            headline TEXT NOT NULL,
            reason TEXT,
            timestamp TEXT,
            PRIMARY KEY(outlet, headline)
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


def row_ignored(conn: sqlite3.Connection, outlet: str, headline: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM ignored WHERE outlet = ? AND headline = ? LIMIT 1",
        (outlet, headline),
    )
    return cur.fetchone() is not None


def add_ignored(conn: sqlite3.Connection, outlet: str, headline: str, reason: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO ignored(outlet, headline, reason, timestamp) VALUES(?,?,?,?)",
        (outlet, headline, reason, ts),
    )
    conn.commit()


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


def _infer_title_from_url(url: str, openai_api_key: str | None) -> str | None:
    try:
        from urllib.parse import urlparse

        p = urlparse(url)
        segs = [s for s in (p.path or "").split("/") if s]
        # Combine last 2 segments as context if present
        context = " ".join(segs[-2:]) if segs else url
        # Prefer LLM extraction; reject ticker-like
        title = extract_market_title(f"URL: {url}\nContext: {context}", api_key=openai_api_key)
        if title and not _is_ticker_like(title):
            return title
        # Fallback: clean typical RT suffixes
        last = segs[-1] if segs else url
        # Kalshi pattern: /markets/<ticker>/<description>/<ticker>
        if len(segs) >= 3 and (segs[-3].lower().startswith("kxr") or segs[-1].lower().startswith("kxr")):
            last = segs[-2]
        last = re.sub(r"-rotten-?tomatoes-?score.*$", "", last, flags=re.I)
        last = re.sub(r"rt-?score.*$", "", last, flags=re.I)
        last = re.sub(r"^rt-", "", last, flags=re.I)
        last = re.sub(r"[-_]+", " ", last).strip()
        if last:
            return _smart_titlecase(last)
        return None
    except Exception:
        return None


def _is_ticker_like(title: str) -> bool:
    t = (title or "").strip()
    if not t or " " in t:
        return False
    low = t.lower()
    if low.startswith("kxrt") or low.startswith("kxr"):
        return True
    if len(low) >= 8 and low.isalnum():
        return True
    return False


def _roman_to_int(word: str) -> str:
    mapping = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}
    w = word.lower()
    return mapping.get(w, word)


def _alias_slugs_for_title(title: str) -> set[str]:
    base = title or ""
    candidates = set()
    # canonical slug
    candidates.add(_slugify_title(base))
    # before colon
    if ":" in base:
        candidates.add(_slugify_title(base.split(":", 1)[0]))
    # without leading article
    low = base.lower().strip()
    for art in ("the ", "a ", "an "):
        if low.startswith(art):
            candidates.add(_slugify_title(base[len(art):]))
            break
    # roman numerals to numbers
    tokens = [ _roman_to_int(t) for t in re.split(r"\s+", base) ]
    candidates.add(_slugify_title(" ".join(tokens)))
    # numbers to roman (basic)
    num_to_rom = {"2": "II", "3": "III", "4": "IV", "5": "V"}
    tokens2 = [ num_to_rom.get(t, t) for t in tokens ]
    candidates.add(_slugify_title(" ".join(tokens2)))
    return {c for c in candidates if c}


AMBIGUOUS_SHORT_TITLES = {"it", "us", "up", "her", "him", "you", "me"}


def _is_ambiguous_movie_title(title: str) -> bool:
    slug = _slugify_title(title)
    # Single token and <= 3 chars or in ambiguous set
    tokens = slug.split("-")
    if len(tokens) == 1 and (len(slug) <= 3 or slug in AMBIGUOUS_SHORT_TITLES):
        return True
    return False


def _phrase_in_headline(headline: str, phrase: str, require_quoted: bool = False) -> bool:
    text = headline or ""
    # Normalize spaces
    phrase_norm = re.sub(r"\s+", " ", phrase).strip()
    if not phrase_norm:
        return False
    # If require quoted, check inside any quote pair
    if require_quoted:
        # Simple quote scanning for ASCII and smart quotes
        quote_patterns = [
            ("\"", "\""),
            ("'", "'"),
            ("“", "”"),
            ("‘", "’"),
        ]
        for lq, rq in quote_patterns:
            pattern = re.compile(
                re.escape(lq) + r"\s*" + re.escape(phrase_norm) + r"\s*" + re.escape(rq),
                re.IGNORECASE,
            )
            if pattern.search(text):
                return True
        return False
    # Otherwise check as whole phrase with word boundaries
    pattern = re.compile(r"\b" + re.escape(phrase_norm) + r"\b", re.IGNORECASE)
    return bool(pattern.search(text))


def get_counts_for_movie(conn: sqlite3.Connection, movie: str) -> Dict[str, int]:
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    try:
        rows = conn.execute(
            "SELECT sentiment, COUNT(*) FROM reviews WHERE movie=? GROUP BY sentiment",
            (movie,),
        ).fetchall()
        for s, c in rows:
            if s in counts:
                counts[s] = int(c or 0)
    except Exception:
        pass
    return counts


def format_message(conn: sqlite3.Connection, outlet: str, headline: str, sentiment: str, movie: str) -> str:
    agg = get_counts_for_movie(conn, movie)
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
            "/scanrules",
            "/addmarketurl",
            "/refreshmarkets",
            "/normalizemarkets",
            "/normalizereviews",
            "/relinkreviews",
            "/commands",
            "/setcanonical",
            "/setrelease",
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
                f"/scanrules@{bot_username.lower()}",
                f"/refreshmarkets@{bot_username.lower()}",
                f"/addmarketurl@{bot_username.lower()}",
                f"/normalizemarkets@{bot_username.lower()}",
                f"/normalizereviews@{bot_username.lower()}",
                f"/relinkreviews@{bot_username.lower()}",
                f"/commands@{bot_username.lower()}",
                f"/setcanonical@{bot_username.lower()}",
                f"/setrelease@{bot_username.lower()}",
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

        # List current market-matched titles (optional filter), deduped and enriched
        if used_cmd.startswith("/markets"):
            markets = load_market_index(conn)
            mcanon = load_market_canon(conn)

            def build_items() -> list[tuple[str, str, str, int, int, int]]:
                groups: dict[tuple, dict] = {}
                for slug, (title, src) in markets.items():
                    ct, rd, tid = mcanon.get(slug, (None, None, None))
                    raw = ct or title or ""
                    base = raw.split(":", 1)[0].strip().lower()
                    key = (tid or None, base)
                    g = groups.setdefault(
                        key, {"titles": set(), "sources": set(), "release": rd}
                    )
                    g["titles"].add(raw)
                    g["sources"].add(src)
                    if not g.get("release") and rd:
                        g["release"] = rd
                # Optional filter
                if arg:
                    q = arg.strip().lower()
                    groups = {
                        k: v for k, v in groups.items() if any(q in t.lower() for t in v["titles"])
                    }
                # Build display tuples with counts and release date
                items_local: list[tuple[str, str, str, int, int, int]] = []
                for v in groups.values():
                    disp = max(v["titles"], key=lambda t: len(t)) if v["titles"] else ""
                    # Ensure/fetch release date if missing
                    rd = v.get("release")
                    if not rd and tmdb_api_key:
                        try:
                            rd = ensure_release_date(conn, disp, tmdb_api_key)
                        except Exception:
                            rd = None
                    # Aggregate review counts across variant titles in group
                    title_list = list(v["titles"]) if v["titles"] else [disp]
                    pos = neg = neu = 0
                    try:
                        placeholders = ",".join(["?"] * len(title_list))
                        qrows = conn.execute(
                            f"SELECT movie, sentiment, COUNT(*) FROM reviews WHERE movie IN ({placeholders}) GROUP BY movie, sentiment",
                            title_list,
                        ).fetchall()
                        for _mv, snt, cnt in qrows:
                            c = int(cnt or 0)
                            if snt == "Positive":
                                pos += c
                            elif snt == "Negative":
                                neg += c
                            elif snt == "Neutral":
                                neu += c
                    except Exception:
                        pass
                    sources_str = ", ".join(sorted(v["sources"]))
                    items_local.append((disp, sources_str, (rd or "n/a"), pos, neg, neu))
                items_local.sort(key=lambda kv: kv[0].lower())
                return items_local

            items = build_items()
            if not items:
                # Attempt on-demand refresh when empty
                try:
                    n = refresh_market_titles(
                        conn,
                        os.getenv("KALSHI_API_KEY", "").strip(),
                        os.getenv("KALSHI_API_SECRET", "").strip(),
                        os.getenv("TMDB_API_KEY", "").strip(),
                        os.getenv("OPENAI_API_KEY", "").strip(),
                    )
                    markets = load_market_index(conn)
                    mcanon = load_market_canon(conn)
                    items = build_items()
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
            lines = ["📈 <b>Market Titles</b>"]
            for t, s, rd, pos, neg, neu in items[:200]:
                t_h = _html_escape(t)
                rd_h = _html_escape(rd or "n/a")
                lines.append(f"• <b>{t_h}</b> ({rd_h}) — {s} — 👍 {pos} / 👎 {neg} / 😐 {neu}")
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
                    os.getenv("OPENAI_API_KEY", "").strip(),
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

        # Normalize market titles (dedupe tickers to canonical TMDb titles)
        if used_cmd.startswith("/normalizemarkets"):
            fixed_rows = 0
            titles_changed = 0
            examples = []
            rows = conn.execute("SELECT slug, title, source FROM market_titles").fetchall()
            for slug, title, source in rows:
                ct_tuple = (load_market_canon(conn).get(slug) or (None, None, None))
                canon_title = ct_tuple[0]
                if not canon_title:
                    # Try to fetch via TMDb if not already cached
                    try:
                        if tmdb_api_key:
                            from movie_meta import fetch_tmdb_canonical

                            ct, rd, tid = fetch_tmdb_canonical(title, tmdb_api_key)
                            if ct or rd or tid:
                                markets_mod.upsert_market_meta(conn, slug, ct, rd, tid)
                                canon_title = ct or canon_title
                    except Exception:
                        pass
                # If ticker-like, attempt to strip ticker prefix (e.g., kxrtHIM -> HIM)
                if not canon_title and _is_ticker_like(title):
                    low = title.lower()
                    rest = None
                    if low.startswith("kxrt") and len(title) > 4:
                        rest = title[4:]
                    elif low.startswith("kxr") and len(title) > 3:
                        rest = title[3:]
                    if rest and rest.strip():
                        canon_title = _smart_titlecase(re.sub(r"[-_]+", " ", rest).strip())
                        # Try TMDb on derived title
                        try:
                            if tmdb_api_key:
                                from movie_meta import fetch_tmdb_canonical
                                ct, rd, tid = fetch_tmdb_canonical(canon_title, tmdb_api_key)
                                if ct or rd or tid:
                                    canon_title = ct or canon_title
                                    markets_mod.upsert_market_meta(conn, slug, ct, rd, tid)
                        except Exception:
                            pass
                # If no canon title, skip
                if not canon_title:
                    continue
                new_title = canon_title
                new_slug = _slugify_title(new_title)
                if new_title == title and new_slug == slug and not _is_ticker_like(title):
                    continue
                try:
                    ts = datetime.now(timezone.utc).isoformat()
                    # Upsert under new slug
                    conn.execute(
                        "INSERT INTO market_titles(slug, title, source, updated_at) VALUES(?,?,?,?) "
                        "ON CONFLICT(slug) DO UPDATE SET title=excluded.title, source=excluded.source, updated_at=excluded.updated_at",
                        (new_slug, new_title, source, ts),
                    )
                    # Move or update meta
                    ct, rd, tid = (load_market_canon(conn).get(slug) or (None, None, None))
                    if ct or rd or tid:
                        markets_mod.upsert_market_meta(conn, new_slug, ct, rd, tid)
                    # If slug changed, delete old row to avoid duplicates
                    if new_slug != slug:
                        conn.execute("DELETE FROM market_titles WHERE slug = ?", (slug,))
                        conn.execute("DELETE FROM market_meta WHERE slug = ?", (slug,))
                    conn.commit()
                    fixed_rows += 1
                    if len(examples) < 6:
                        examples.append(f"• {title} → {new_title}")
                    if new_title != title:
                        titles_changed += 1
                except Exception:
                    continue

            msg = f"Normalized {fixed_rows} market entries. Titles changed: {titles_changed}."
            if examples:
                msg += "\nExamples:\n" + "\n".join(_html_escape(x) for x in examples)
            send_telegram_message(token, chat_id, msg)
            # Second pass: collapse variants like 'Title' vs 'Title: Subtitle' to the longer title
            try:
                variants = conn.execute("SELECT slug, title FROM market_titles").fetchall()
                groups: dict[str, list[tuple[str, str]]] = {}
                for s, t in variants:
                    base = t.lower().split(":")[0].strip()
                    base = re.sub(r"\s+", " ", base)
                    groups.setdefault(base, []).append((s, t))
                for base, items in groups.items():
                    if len(items) <= 1:
                        continue
                    # Choose longest title as canonical for the group
                    canonical = max(items, key=lambda it: len(it[1] or ""))[1]
                    cslug = _slugify_title(canonical)
                    ts = datetime.now(timezone.utc).isoformat()
                    # Upsert canonical row
                    conn.execute(
                        "INSERT INTO market_titles(slug, title, source, updated_at) VALUES(?,?,?,?) "
                        "ON CONFLICT(slug) DO UPDATE SET title=excluded.title, updated_at=excluded.updated_at",
                        (cslug, canonical, "merged", ts),
                    )
                    # Migrate others to canonical slug
                    for s, t in items:
                        if s == cslug:
                            continue
                        conn.execute("DELETE FROM market_titles WHERE slug = ?", (s,))
                        conn.execute("UPDATE market_meta SET slug=? WHERE slug=?", (cslug, s))
                conn.commit()
            except Exception:
                pass
            continue

        # Normalize review rows' movie field to canonical titles from markets/TMDb
        if used_cmd.startswith("/normalizereviews"):
            markets = load_market_index(conn)
            mcanon = load_market_canon(conn)
            changed_rows = 0
            titles_changed = 0
            examples = []
            # Get distinct movies from reviews
            dmovies = conn.execute("SELECT DISTINCT movie FROM reviews").fetchall()
            for (old_title,) in dmovies:
                old_title_s = str(old_title)
                slug = _slugify_title(old_title_s)
                new_title = None
                # Prefer canonical TMDb title
                ct, rd, tid = mcanon.get(slug, (None, None, None))
                if ct:
                    new_title = ct
                elif slug in markets:
                    new_title = markets[slug][0]
                # Skip if nothing to map
                if not new_title or new_title == old_title_s or _is_ticker_like(new_title):
                    continue
                try:
                    cur = conn.execute("UPDATE reviews SET movie=? WHERE movie=?", (new_title, old_title_s))
                    conn.commit()
                    count = cur.rowcount or 0
                    if count > 0:
                        changed_rows += count
                        titles_changed += 1
                        if len(examples) < 6:
                            examples.append(f"• {old_title_s} → {new_title} ({count})")
                    # Cache release date if we have it
                    try:
                        if rd:
                            from movie_meta import cache_release_date

                            cache_release_date(conn, new_title, rd)
                    except Exception:
                        pass
                except Exception:
                    continue

            msg = f"Normalized review titles in {changed_rows} rows. Titles changed: {titles_changed}."
            if examples:
                msg += "\nExamples:\n" + "\n".join(_html_escape(x) for x in examples)
            send_telegram_message(token, chat_id, msg)
            continue

        # Relink (prune) reviews for a specific movie using current matching rules (headline gating)
        if used_cmd.startswith("/relinkreviews"):
            target = arg.strip()
            if not target:
                send_telegram_message(token, chat_id, "Usage: /relinkreviews <movie title>")
                continue
            # Build phrase set from market/canonical if available
            slug_target = _slugify_title(target)
            phrases = set()
            ct_default = None
            # Find matching market entry
            for s, (t, _src) in load_market_index(conn).items():
                if s == slug_target or _slugify_title(t) == slug_target:
                    phrases.add(t)
                    ct_default = (load_market_canon(conn).get(s) or (None, None, None))[0]
                    if ct_default:
                        phrases.add(ct_default)
                    break
            # Fallback to provided target
            if not phrases:
                phrases.add(target)
            # Add base-before-colon variants
            for ph in list(phrases):
                if ":" in ph:
                    phrases.add(ph.split(":", 1)[0])
            ambiguous = _is_ambiguous_movie_title(next(iter(phrases)))
            # Fetch candidate rows
            rows = conn.execute(
                "SELECT id, outlet, headline FROM reviews WHERE LOWER(movie)=LOWER(?)",
                (target,),
            ).fetchall()
            removed = 0
            kept = 0
            examples = []
            for rid, outlet, hl in rows:
                ok = False
                for ph in phrases:
                    if not ph:
                        continue
                    if _phrase_in_headline(hl or "", ph, require_quoted=ambiguous):
                        ok = True
                        break
                if ok:
                    kept += 1
                    continue
                # prune false positive
                try:
                    conn.execute("DELETE FROM reviews WHERE id=?", (rid,))
                    add_ignored(conn, outlet, hl or "", reason="relink_prune")
                    removed += 1
                    if len(examples) < 5:
                        examples.append(f"• {outlet}: {hl[:80]}…")
                except Exception:
                    pass
            conn.commit()
            # No stale in-memory aggregates: we compute on-demand for notifications. Optionally, report fresh counts.
            fresh = get_counts_for_movie(conn, target)
            send_telegram_message(
                token,
                chat_id,
                (f"Relink complete for ‘{_html_escape(target)}’. Removed {removed}, kept {kept}. Current: 👍 {fresh.get('Positive',0)} / 👎 {fresh.get('Negative',0)} / 😐 {fresh.get('Neutral',0)}" +
                 ("\nExamples removed:\n" + "\n".join(_html_escape(x) for x in examples) if examples else "")),
                parse_mode="HTML",
            )
            continue

        # Commands help
        if used_cmd.startswith("/commands"):
            lines = [
                "🧭 <b>Bot Commands</b>",
                "/markets [filter] — Title (date) • sources • 👍/👎/😐",
                "/status — Summary sorted by upcoming releases",
                "/status &lt;movie&gt; — Single‑film totals",
                "/addmarketurl &lt;urls...&gt; — Seed films from Polymarket/Kalshi URLs",
                "/refreshmarkets — Refresh market titles now",
                "/normalizemarkets — Strip tickers, merge Title vs Title: Subtitle",
                "/normalizereviews — Rewrite reviews to canonical titles",
                "/relinkreviews &lt;movie&gt; — Prune false positives (adds tombstones)",
                "/scanrules — Show what gets accepted",
                "/backfill — Fill missing release dates via TMDb",
                "/catalog [filter] — TMDb window (info)",
                "/refreshcatalog — Refresh TMDb window (info)",
                "/health — Basic status snapshot",
                "/testapi — Check OpenAI/TMDb/Polymarket/Kalshi connectivity",
                "",
                "ℹ️ Matching: needs review cues + film match.",
                "• Cues: ‘review’, ‘film review’, ‘critic review’, ‘verdict’, stars (★/‘3 stars’).",
                "• Match via URL alias slugs and headline phrase (word‑boundaries).",
                "• Ambiguous short titles (HIM/IT/US/UP/HER/ME/YOU): headline must quote the title.",
                "• Tombstones stop deleted headlines from reappearing.",
                "",
                "Manual pins:",
                "/setcanonical “<title>” <tmdb_id> — pin TMDb entry",
                "/setrelease “<title>” <YYYY-MM-DD> — override date",
            ]
            _send_batched_message(token, chat_id, lines)
            continue

        # Scan rules overview
        if used_cmd.startswith("/scanrules"):
            from urllib.parse import urlparse

            # Outlets
            hosts = []
            for u in FEEDS:
                try:
                    hosts.append((urlparse(u).netloc or u).lower())
                except Exception:
                    hosts.append(u)
            hosts = sorted(set(hosts))
            # Films (market titles)
            markets = load_market_index(conn)
            mcanon = load_market_canon(conn)
            films = []
            for slug, (t, src) in markets.items():
                ct = (mcanon.get(slug) or (None, None, None))[0]
                films.append((ct or t or slug, src))
            films.sort(key=lambda x: x[0].lower())

            lines = [
                "🔎 <b>Scanner Rules</b>",
                f"Interval: {POLL_SECONDS}s",
                f"Outlets ({len(hosts)}): " + ", ".join(hosts[:15]) + (" …" if len(hosts) > 15 else ""),
                "Review cues: ‘review’, ‘film review’, ‘critic review’, ‘verdict’, stars (★/‘3 stars’)",
                "Gating: must match a tracked film (market/seeded)",
                "Match logic:",
                "• URL alias slugs (full, base-before-colon, no article, roman↔number)",
                "• OR headline phrase (word‑boundaries)",
                "• Ambiguous short titles (HIM/IT/US/UP/HER/ME/YOU): headline must quote the title",
                "De‑dup + tombstones: (outlet, headline) + ignored table to block re‑ingest",
                "Films (sample):",
            ]
            for name, src in films[:15]:
                lines.append(f"• { _html_escape(name) } — {src}")
            if len(films) > 15:
                lines.append(f"… and {len(films)-15} more")
            _send_batched_message(token, chat_id, lines)
            continue

        # Backfill last N days for a specific film by scanning current feeds
        if used_cmd.startswith("/backfillfilm"):
            m = re.match(r"^\s*\/?backfillfilm\s+\“([^\”]+)\”(?:\s+(\d+))?\s*$", text) or \
                re.match(r"^\s*\/?backfillfilm\s+\"([^\"]+)\"(?:\s+(\d+))?\s*$", text) or \
                re.match(r"^\s*\/?backfillfilm\s+(.+?)(?:\s+(\d+))?\s*$", text)
            if not m:
                send_telegram_message(token, chat_id, "Usage: /backfillfilm \"Title\" [days]")
                continue
            film = m.group(1).strip()
            days = int(m.group(2)) if m.group(2) else 7
            # Build alias/root mapping for all tracked films
            market_index = load_market_index(conn)
            market_canon = load_market_canon(conn)
            alias_to_root: dict[str, str] = {}
            for slug, (mtitle, _src) in market_index.items():
                alias_to_root[_slugify_title(mtitle)] = slug
                alias = _alias_slugs_for_title(mtitle)
                ct = (market_canon.get(slug) or (None, None, None))[0]
                if ct:
                    alias |= _alias_slugs_for_title(ct)
                for a in alias:
                    alias_to_root.setdefault(a, slug)
            # Determine target root slug(s)
            target_slug = alias_to_root.get(_slugify_title(film))
            if not target_slug:
                send_telegram_message(token, chat_id, f"Film not tracked: {film}")
                continue
            # Scan feeds once
            try:
                items = fetch_all(FEEDS)
            except Exception as e:
                send_telegram_message(token, chat_id, f"Backfill error fetching feeds: {e}")
                continue
            # Filter by date window
            from datetime import datetime, timezone, timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            added = 0
            checked = 0
            for it in items:
                # Date filter
                dt = None
                try:
                    dt = eut.parsedate_to_datetime(it.published) if it.published else None
                    if dt and dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                except Exception:
                    dt = None
                if dt and dt < cutoff:
                    continue
                outlet = it.outlet
                headline = (it.title or "").strip()
                link = it.link or ""
                if row_exists(conn, outlet, headline) or row_ignored(conn, outlet, headline):
                    continue
                # Must pass review cues
                if not _is_review_candidate(headline, it.summary, link):
                    continue
                checked += 1
                # Match gating using aliases + headline phrase rules
                path = (link or "").lower()
                headline_l = headline.lower()
                haystack = f"{path} {headline_l}"
                matched_root = None
                # Try alias slug match
                for als, root in alias_to_root.items():
                    if root != target_slug:
                        continue
                    if _is_ambiguous_movie_title(als):
                        if re.search(r"\b" + re.escape(als) + r"\b", path):
                            matched_root = root
                            break
                    else:
                        if re.search(r"\b" + re.escape(als) + r"\b", haystack):
                            matched_root = root
                            break
                # Try phrase in headline if not matched
                if not matched_root:
                    ct = (market_canon.get(target_slug) or (None, None, None))[0]
                    phrases = set([film])
                    if ct:
                        phrases.add(ct)
                    for ph in list(phrases):
                        if ":" in ph:
                            phrases.add(ph.split(":", 1)[0])
                    ambiguous = _is_ambiguous_movie_title(next(iter(phrases)))
                    for ph in phrases:
                        if _phrase_in_headline(headline, ph, require_quoted=ambiguous):
                            matched_root = target_slug
                            break
                if not matched_root:
                    continue
                # Insert
                # Resolve movie name
                canon = market_canon.get(matched_root)
                movie = (canon and canon[0]) or (market_index.get(matched_root) or (film,))[0]
                # Classify
                text_for_model = f"{headline}\n\n{it.summary}".strip()
                sentiment = classify_sentiment(text_for_model, api_key=os.getenv("OPENAI_API_KEY", "").strip())
                if sentiment not in ("Positive", "Negative", "Neutral"):
                    sentiment = "Neutral"
                insert_review(conn, outlet, movie, headline, sentiment)
                added += 1
            send_telegram_message(token, chat_id, f"Backfill complete for ‘{film}’: added {added}, scanned {checked} items.")
            continue

        # Pin canonical TMDb mapping
        if used_cmd.startswith("/setcanonical"):
            # Accept: /setcanonical "Title" 12345 or /setcanonical Title 12345
            m = re.match(r"^\s*\/?setcanonical\s+\“([^\”]+)\”\s+(\d+)\s*$", text) or \
                re.match(r"^\s*\/?setcanonical\s+\"([^\"]+)\"\s+(\d+)\s*$", text) or \
                re.match(r"^\s*\/?setcanonical\s+(.+?)\s+(\d+)\s*$", text)
            if not m:
                send_telegram_message(token, chat_id, "Usage: /setcanonical \"Title\" <tmdb_id>")
                continue
            title_arg, id_arg = m.group(1), m.group(2)
            tmdb_id = int(id_arg)
            if not tmdb_api_key:
                send_telegram_message(token, chat_id, "TMDB_API_KEY not set")
                continue
            ct, rd = fetch_tmdb_by_id(tmdb_id, tmdb_api_key)
            canon_title = ct or title_arg
            # Resolve or create market slug
            slug = _slugify_title(title_arg)
            try:
                # Ensure there is a market_titles row
                ts = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO market_titles(slug, title, source, updated_at) VALUES(?,?,?,?) "
                    "ON CONFLICT(slug) DO UPDATE SET title=excluded.title, updated_at=excluded.updated_at",
                    (slug, canon_title, "manual", ts),
                )
                # Upsert meta
                markets_mod.upsert_market_meta(conn, slug, canon_title, rd, tmdb_id)
                # Cache release date for summaries
                if rd:
                    try:
                        from movie_meta import cache_release_date

                        cache_release_date(conn, canon_title, rd)
                    except Exception:
                        pass
                conn.commit()
                send_telegram_message(token, chat_id, f"Pinned: {canon_title} (tmdb:{tmdb_id}) date={rd or 'n/a'}")
            except Exception as e:
                send_telegram_message(token, chat_id, f"Failed to pin: {e}")
            continue

        # Override release date
        if used_cmd.startswith("/setrelease"):
            m = re.match(r"^\s*\/?setrelease\s+\“([^\”]+)\”\s+(\d{4}-\d{2}-\d{2})\s*$", text) or \
                re.match(r"^\s*\/?setrelease\s+\"([^\"]+)\"\s+(\d{4}-\d{2}-\d{2})\s*$", text) or \
                re.match(r"^\s*\/?setrelease\s+(.+?)\s+(\d{4}-\d{2}-\d{2})\s*$", text)
            if not m:
                send_telegram_message(token, chat_id, "Usage: /setrelease \"Title\" YYYY-MM-DD")
                continue
            title_arg, rd = m.group(1), m.group(2)
            slug = _slugify_title(title_arg)
            try:
                # Ensure market_titles exists
                ts = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO market_titles(slug, title, source, updated_at) VALUES(?,?,?,?) "
                    "ON CONFLICT(slug) DO UPDATE SET updated_at=excluded.updated_at",
                    (slug, title_arg, "manual", ts),
                )
                # Keep existing canon title if present
                ct, _, tid = (load_market_canon(conn).get(slug) or (None, None, None))
                markets_mod.upsert_market_meta(conn, slug, ct or title_arg, rd, tid or 0)
                # Cache into movies
                try:
                    from movie_meta import cache_release_date

                    cache_release_date(conn, ct or title_arg, rd)
                except Exception:
                    pass
                conn.commit()
                send_telegram_message(token, chat_id, f"Release set: {title_arg} → {rd}")
            except Exception as e:
                send_telegram_message(token, chat_id, f"Failed to set release: {e}")
            continue

        # Manually seed market titles via URLs (space/newline separated)
        if used_cmd.startswith("/addmarketurl"):
            urls = [u for u in re.split(r"\s+", arg) if u.startswith("http")]
            if not urls:
                send_telegram_message(token, chat_id, "Usage: /addmarketurl <url1> <url2> ...")
                continue
            added = 0
            examples = []
            for u in urls:
                title = _infer_title_from_url(u, os.getenv("OPENAI_API_KEY", "").strip())
                if not title:
                    continue
                slug = _slugify_title(title)
                # Infer source from URL host
                try:
                    from urllib.parse import urlparse

                    host = (urlparse(u).netloc or "").lower()
                    src = "polymarket" if "polymarket" in host else ("kalshi" if "kalshi" in host else "manual")
                except Exception:
                    src = "manual"
                try:
                    ts = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        "INSERT INTO market_titles(slug, title, source, updated_at) VALUES(?,?,?,?) "
                        "ON CONFLICT(slug) DO UPDATE SET title=excluded.title, source=excluded.source, updated_at=excluded.updated_at",
                        (slug, title, src, ts),
                    )
                    # Resolve TMDb canonical info if possible
                    try:
                        if tmdb_api_key:
                            from movie_meta import fetch_tmdb_canonical
                            from markets import upsert_market_meta

                            ct, rd, tid = fetch_tmdb_canonical(title, tmdb_api_key)
                            if ct or rd or tid:
                                upsert_market_meta(conn, slug, ct, rd, tid)
                    except Exception:
                        pass
                    conn.commit()
                    added += 1
                    if len(examples) < 5:
                        examples.append(f"• {title}")
                except Exception:
                    continue
            send_telegram_message(token, chat_id, f"Seeded {added} market titles.\n" + ("\n".join(examples) if examples else ""))
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
        n = refresh_market_titles(conn, kalshi_key, kalshi_secret, tmdb_api_key, os.getenv("OPENAI_API_KEY", "").strip())
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
                n = refresh_market_titles(conn, kalshi_key, kalshi_secret, tmdb_api_key, os.getenv("OPENAI_API_KEY", "").strip())
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
            if row_exists(conn, outlet, headline) or row_ignored(conn, outlet, headline):
                continue

            # Match movie via market titles + alias slugs (primary gate): find slug in URL or phrase in headline
            link_or_title = it.link or ""
            path = (link_or_title or "").lower()
            # Match against both URL path and headline text for robustness
            headline_l = (headline or "").lower()
            haystack = f"{path} {headline_l}"
            best_slug = None
            best_len = 0
            # Build alias map on the fly
            alias_map: dict[str, str] = {}
            for slug, (mtitle, _src) in market_index.items():
                alias_map[slug] = slug
                alias_set = _alias_slugs_for_title(mtitle)
                # Include canonical title alias if available
                ct = (market_canon.get(slug) or (None, None, None))[0]
                if ct:
                    alias_set |= _alias_slugs_for_title(ct)
                for als in alias_set:
                    alias_map.setdefault(als, slug)
            for als, root_slug in alias_map.items():
                if not als:
                    continue
                matched = False
                # Ambiguous short titles: only accept from URL path, not headline
                if als in AMBIGUOUS_SHORT_TITLES:
                    matched = als in path
                else:
                    # Require word-boundary match in either URL path or headline
                    if re.search(r"\b" + re.escape(als) + r"\b", path):
                        matched = True
                    elif re.search(r"\b" + re.escape(als) + r"\b", headline_l):
                        matched = True
                if matched and len(als) > best_len:
                    best_slug = root_slug
                    best_len = len(als)
            if not best_slug:
                # Try phrase matching on headline for non-ambiguous titles
                for slug, (mtitle, _src) in market_index.items():
                    ct = (market_canon.get(slug) or (None, None, None))[0]
                    phrases = set()
                    if mtitle:
                        phrases.add(mtitle)
                        if ":" in mtitle:
                            phrases.add(mtitle.split(":", 1)[0])
                    if ct:
                        phrases.add(ct)
                        if ":" in ct:
                            phrases.add(ct.split(":", 1)[0])
                    ambiguous = _is_ambiguous_movie_title(ct or mtitle or "")
                    for ph in phrases:
                        if not ph:
                            continue
                        if _phrase_in_headline(headline or "", ph, require_quoted=ambiguous):
                            best_slug = slug
                            best_len = len(ph)
                            break
                    if best_slug:
                        break
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

            # Insert into DB
            insert_review(conn, outlet, movie, headline, sentiment)

            # Send Telegram notification
            msg = format_message(conn, outlet, headline, sentiment, movie)
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
