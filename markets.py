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
    conn.commit()


def _slugify(value: str) -> str:
    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9]+", "-", v)
    v = re.sub(r"-+", "-", v).strip("-")
    return v


def _guess_movie_from_text(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
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


def fetch_polymarket_titles(timeout: int = 15) -> List[Tuple[str, str]]:
    """Return list of (slug, title) from active Polymarket markets about RT scores.

    Endpoint is best-effort; Polymarket has multiple APIs. We try a public one.
    """
    titles: List[Tuple[str, str]] = []
    try:
        # Common public endpoint pattern
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 1000},
            timeout=timeout,
        )
        if not resp.ok:
            return titles
        data = resp.json() or {}
        markets = data.get("data") or data.get("markets") or data
        if isinstance(markets, dict):
            markets = markets.get("markets", [])
        for m in markets or []:
            # Try common fields
            q = str(m.get("question") or m.get("title") or m.get("name") or "")
            status = str(m.get("status") or m.get("closed") or "").lower()
            is_open = "close" not in status and status not in {"closed", "resolved"}
            if not is_open:
                continue
            hay = q.lower()
            if not any(k in hay for k in ("rotten tomatoes", "tomatometer", "rt score", "rt %")):
                continue
            mv = _guess_movie_from_text(q)
            if mv:
                titles.append((_slugify(mv), mv))
    except Exception:
        return titles
    return titles


def fetch_kalshi_titles(api_key: Optional[str], api_secret: Optional[str], timeout: int = 15) -> List[Tuple[str, str]]:
    """Return list of (slug, title) from Kalshi markets if credentials provided.

    Kalshi typically requires auth; if missing, returns empty.
    """
    if not api_key or not api_secret:
        return []
    titles: List[Tuple[str, str]] = []
    try:
        # Basic markets list; Kalshi may change endpoints/fields
        url = "https://trading-api.kalshi.com/v1/markets"
        headers = {"KALSHI-API-KEY": api_key, "KALSHI-API-SECRET": api_secret}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if not resp.ok:
            return []
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
            mv = _guess_movie_from_text(title)
            if mv:
                titles.append((_slugify(mv), mv))
    except Exception:
        return []
    return titles


def refresh_market_titles(conn, kalshi_key: Optional[str], kalshi_secret: Optional[str]) -> int:
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
    for slug, title in fetch_polymarket_titles():
        upsert("polymarket", slug, title)

    # Kalshi (optional auth)
    for slug, title in fetch_kalshi_titles(kalshi_key, kalshi_secret):
        upsert("kalshi", slug, title)

    conn.commit()
    return upserted


def load_market_index(conn) -> dict[str, tuple[str, str]]:
    rows = conn.execute("SELECT slug, title, source FROM market_titles").fetchall()
    return {slug: (title, source) for slug, title, source in rows}

