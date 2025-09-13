"""Lightweight movie metadata lookup and caching.

Fetches movie release dates from TMDb (if TMDB_API_KEY is provided) and caches
them in SQLite to avoid repeated lookups. If no API key is configured, the
functions safely no-op and return None.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import requests


def ensure_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS movies (
            movie TEXT PRIMARY KEY,
            release_date TEXT,
            source TEXT,
            last_checked TEXT
        )
        """
    )
    conn.commit()


def get_cached_release_date(conn, movie: str) -> Optional[str]:
    row = conn.execute(
        "SELECT release_date FROM movies WHERE movie = ?",
        (movie,),
    ).fetchone()
    return row[0] if row and row[0] else None


def cache_release_date(conn, movie: str, release_date: Optional[str], source: str = "tmdb") -> None:
    ts = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO movies(movie, release_date, source, last_checked) VALUES(?,?,?,?)"
        " ON CONFLICT(movie) DO UPDATE SET release_date=excluded.release_date, source=excluded.source, last_checked=excluded.last_checked",
        (movie, release_date or "", source, ts),
    )
    conn.commit()


def fetch_tmdb_release_date(movie: str, api_key: str, timeout: int = 10) -> Optional[str]:
    try:
        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": api_key, "query": movie, "include_adult": "false"},
            timeout=timeout,
        )
        if not resp.ok:
            return None
        data = resp.json()
        results = data.get("results") or []
        if not results:
            return None
        # Prefer exact (case-insensitive) title match; otherwise first result
        lowered = movie.lower().strip()
        best = None
        for r in results:
            if str(r.get("title", "")).lower().strip() == lowered:
                best = r
                break
        best = best or results[0]
        rd = best.get("release_date")
        if not rd:
            return None
        # Validate format YYYY-MM-DD
        try:
            datetime.strptime(rd, "%Y-%m-%d")
        except Exception:
            return None
        return rd
    except Exception:
        return None


def ensure_release_date(conn, movie: str, tmdb_api_key: Optional[str]) -> Optional[str]:
    """Get or fetch-and-cache a movie's release date (YYYY-MM-DD)."""
    if not movie:
        return None
    ensure_tables(conn)
    cached = get_cached_release_date(conn, movie)
    if cached:
        return cached
    if not tmdb_api_key:
        return None
    rd = fetch_tmdb_release_date(movie, tmdb_api_key)
    cache_release_date(conn, movie, rd, source="tmdb")
    return rd

