"""Prediction market integrations (Polymarket + Kalshi).

Fetches currently tradable markets about Rotten Tomatoes/film scores and
extracts probable movie titles. Stores a union set in SQLite table
`market_titles(slug, title, source, updated_at)`.

Both integrations are best-effort. Polymarket is public HTTP. Kalshi usually
requires credentials; we only query Kalshi when creds are provided.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import re
import requests
from sentiment import extract_market_title


def ensure_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS market_titles (
            slug TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS market_meta (
            slug TEXT PRIMARY KEY,
            canon_title TEXT,
            release_date TEXT,
            tmdb_id INTEGER,
            updated_at TEXT
        )
        """
    )
    conn.commit()


def _slugify(value: str) -> str:
    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9]+", "-", v)
    v = re.sub(r"-+", "-", v).strip("-")
    return v


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


def _guess_movie_from_text(text: str, openai_api_key: Optional[str] = None) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    # Preferred: ask OpenAI to extract the title when key present
    try:
        title = extract_market_title(t, api_key=openai_api_key)
        if title and not _is_ticker_like(title):
            return title
    except Exception:
        pass
    # Prefer quoted phrases
    m = re.search(r"[“\"]([^”\"]+)[”\"]", t)
    if m:
        return m.group(1).strip()
    # Try before Rotten Tomatoes or Tomatometer
    m = re.search(r"(.+?)\s+(?:Rotten Tomatoes|Tomatometer)", t, re.I)
    if m:
        return m.group(1).strip(" -:•|\u2013\u2014")
    # Otherwise take first 6 words
    return " ".join(t.split()[:6])


def fetch_polymarket_titles(timeout: int = 15, openai_api_key: Optional[str] = None) -> List[Tuple[str, str]]:
    """Return list of (slug, title) from active Polymarket markets about RT scores.

    Endpoint is best-effort; Polymarket has multiple APIs. We try a public one.
    """
    titles: List[Tuple[str, str]] = []
    try:
        # Common public endpoint pattern
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 1000, "closed": "false"},
            timeout=timeout,
        )
        if not resp.ok:
            return titles
        data = resp.json() or {}
        markets = data.get("markets") or data.get("data") or []
        if isinstance(markets, dict):
            markets = markets.get("markets", []) or []
        for m in markets or []:
            # Try common fields
            q = str(m.get("question") or m.get("title") or m.get("name") or m.get("condition") or "")
            closed = m.get("closed")
            status = str(m.get("status") or "").lower()
            is_open = not closed and status not in {"closed", "resolved", "settled"}
            if not is_open:
                continue
            hay = q.lower()
            # broader cues for RT/tomatometer
            cues = (
                "rotten tomatoes",
                "tomatometer",
                "rt score",
                "rt %",
                " rt ",
                "(rt",
                "rt)",
                "rt-",
                "tomatoes",
            )
            if not any(k in hay for k in cues):
                continue
            mv = _guess_movie_from_text(q, openai_api_key)
            if mv:
                titles.append((_slugify(mv), mv))
        # Additionally, perform explicit search queries to catch filtered results
        try:
            seen = set(slug for slug, _ in titles)
            search_terms = ("rotten tomatoes", "tomatometer", "rt score")
            for term in search_terms:
                r = requests.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={"limit": 1000, "closed": "false", "search": term},
                    timeout=timeout,
                )
                if not r.ok:
                    continue
                d2 = r.json() or {}
                ms2 = d2.get("markets") or d2.get("data") or []
                if isinstance(ms2, dict):
                    ms2 = ms2.get("markets", []) or []
                for m2 in ms2 or []:
                    q2 = str(m2.get("question") or m2.get("title") or m2.get("name") or m2.get("condition") or "")
                    closed2 = m2.get("closed")
                    status2 = str(m2.get("status") or "").lower()
                    if closed2 or status2 in {"closed", "resolved", "settled"}:
                        continue
                    hay2 = q2.lower()
                    if not any(k in hay2 for k in cues):
                        continue
                    mv2 = _guess_movie_from_text(q2, openai_api_key)
                    if mv2:
                        s2 = _slugify(mv2)
                        if s2 not in seen:
                            titles.append((s2, mv2))
                            seen.add(s2)
        except Exception:
            pass
        # Final fallback: GraphQL query (best-effort) if REST returned nothing
        if not titles:
            try:
                gql_url = "https://gamma-api.polymarket.com/graphql"
                query = (
                    "query MarketList($limit:Int,$closed:Boolean){\n"
                    "  markets(limit:$limit, closed:$closed){\n"
                    "    question title name condition status closed description\n"
                    "  }\n"
                    "}"
                )
                r = requests.post(
                    gql_url,
                    json={"query": query, "variables": {"limit": 1000, "closed": False}},
                    timeout=timeout,
                )
                if r.ok:
                    d = r.json() or {}
                    markets = ((d.get("data") or {}).get("markets")) or []
                    for m in markets:
                        q = str(m.get("question") or m.get("title") or m.get("name") or m.get("condition") or "")
                        status = str(m.get("status") or "").lower()
                        closed = m.get("closed")
                        if closed or status in {"closed", "resolved", "settled"}:
                            continue
                        hay = q.lower() + " " + str(m.get("description") or "").lower()
                        if not any(k in hay for k in cues):
                            continue
                        mv = _guess_movie_from_text(q, openai_api_key)
                        if mv:
                            titles.append((_slugify(mv), mv))
            except Exception:
                pass
    except Exception:
        return titles
    return titles


def fetch_kalshi_titles_public(timeout: int = 15, openai_api_key: Optional[str] = None) -> List[Tuple[str, str]]:
    """Attempt to fetch markets from Kalshi v2 public endpoint (no auth).

    Returns list of (slug, title). If unavailable, returns empty.
    """
    titles: List[Tuple[str, str]] = []
    try:
        # Try new public hostname first
        candidates = [
            "https://api.elections.kalshi.com/trade-api/v2/markets",
            "https://trading-api.kalshi.com/trade-api/v2/markets",
        ]
        data = None
        for url in candidates:
            try:
                resp = requests.get(url, params={"limit": 1000}, timeout=timeout)
                if resp.ok:
                    data = resp.json() or {}
                    break
            except Exception:
                continue
        if not data:
            return []
        markets = data.get("markets") or data.get("data") or []
        if isinstance(markets, dict):
            markets = markets.get("markets", []) or []
        for m in markets:
            title = str(m.get("title") or m.get("name") or "")
            status = str(m.get("status") or "").lower()
            if status in {"closed", "settled", "resolved"}:
                continue
            hay = (title or "").lower() + " " + str(m.get("description") or "").lower()
            if not any(k in hay for k in ("rotten tomatoes", "tomatometer", "rt score", "rt %", "tomatoes", " rt ")):
                continue
            mv = _guess_movie_from_text(title, openai_api_key)
            if mv:
                titles.append((_slugify(mv), mv))
    except Exception:
        return []
    return titles


def fetch_kalshi_titles(api_key: Optional[str], api_secret: Optional[str], timeout: int = 15, openai_api_key: Optional[str] = None) -> List[Tuple[str, str]]:
    """Return list of (slug, title) from Kalshi markets if credentials provided.

    Kalshi typically requires auth; if missing, returns empty.
    """
    titles: List[Tuple[str, str]] = []
    # Try public v2 first
    titles.extend(fetch_kalshi_titles_public(timeout=timeout, openai_api_key=openai_api_key))
    # If creds present, also try authenticated v1 as a fallback/supplement
    if not api_key or not api_secret:
        return titles
    try:
        # Basic markets list; Kalshi may change endpoints/fields
        url = "https://trading-api.kalshi.com/v1/markets"
        headers = {"KALSHI-API-KEY": api_key, "KALSHI-API-SECRET": api_secret}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if not resp.ok:
            return titles
        data = resp.json() or {}
        markets = data.get("markets") or []
        for m in markets:
            title = str(m.get("title") or m.get("name") or "")
            status = str(m.get("status") or "").lower()
            if status in {"closed", "settled", "resolved"}:
                continue
            hay = title.lower()
            if not any(k in hay for k in ("rotten tomatoes", "tomatometer", "rt score", "rt %")):
                continue
            mv = _guess_movie_from_text(title, openai_api_key)
            if mv:
                titles.append((_slugify(mv), mv))
    except Exception:
        return titles
    return titles


def refresh_market_titles(conn, kalshi_key: Optional[str], kalshi_secret: Optional[str], tmdb_api_key: Optional[str] = None, openai_api_key: Optional[str] = None) -> int:
    ensure_tables(conn)
    upserted = 0
    seen = set()

    def upsert(src: str, slug: str, title: str) -> None:
        nonlocal upserted
        if (src, slug) in seen:
            return
        seen.add((src, slug))
        ts = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO market_titles(slug, title, source, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(slug) DO UPDATE SET title=excluded.title, source=excluded.source, updated_at=excluded.updated_at",
            (slug, title, src, ts),
        )
        upserted += 1

    # Polymarket (public)
    for slug, title in fetch_polymarket_titles(openai_api_key=openai_api_key):
        upsert("polymarket", slug, title)

    # Kalshi (public v2 + optional auth)
    for slug, title in fetch_kalshi_titles(kalshi_key, kalshi_secret, openai_api_key=openai_api_key):
        upsert("kalshi", slug, title)

    conn.commit()
    # Optionally resolve canonical titles/release dates via TMDb for newly seen slugs
    if tmdb_api_key:
        try:
            from movie_meta import fetch_tmdb_canonical

            rows = conn.execute("SELECT slug, title FROM market_titles").fetchall()
            for slug, title in rows:
                ct, rd, tid = fetch_tmdb_canonical(title, tmdb_api_key)
                if ct or rd or tid:
                    upsert_market_meta(conn, slug, ct, rd, tid)
        except Exception:
            pass

    return upserted


def load_market_index(conn) -> dict[str, tuple[str, str]]:
    rows = conn.execute("SELECT slug, title, source FROM market_titles").fetchall()
    return {slug: (title, source) for slug, title, source in rows}


def upsert_market_meta(conn, slug: str, canon_title: Optional[str], release_date: Optional[str], tmdb_id: Optional[int]) -> None:
    ts = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO market_meta(slug, canon_title, release_date, tmdb_id, updated_at) VALUES(?,?,?,?,?) "
        "ON CONFLICT(slug) DO UPDATE SET canon_title=excluded.canon_title, release_date=excluded.release_date, tmdb_id=excluded.tmdb_id, updated_at=excluded.updated_at",
        (slug, canon_title or "", release_date or "", tmdb_id or 0, ts),
    )
    conn.commit()


def load_market_canon(conn) -> dict[str, tuple[Optional[str], Optional[str], Optional[int]]]:
    rows = conn.execute("SELECT slug, NULLIF(canon_title,''), NULLIF(release_date,''), NULLIF(tmdb_id,0) FROM market_meta").fetchall()
    return {slug: (ct, rd, tid) for slug, ct, rd, tid in rows}
