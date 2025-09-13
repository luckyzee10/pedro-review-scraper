"""Movie metadata utilities backed by TMDb.

Capabilities:
- Fetch and cache a movie's release date (on-demand) into the `movies` table.
- Maintain a rolling catalog of titles within a date window (e.g., 5 days past
  to 90 days future) in the `catalog` table for URL-based matching.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Dict, Tuple, Iterable

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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS catalog (
            slug TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            release_date TEXT,
            tmdb_id INTEGER,
            updated_at TEXT
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


def fetch_tmdb_canonical(title: str, api_key: str, timeout: int = 10) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Return (canonical_title, release_date, tmdb_id) for a title via TMDb.

    Chooses an exact case-insensitive match if available; otherwise the first result.
    Returns (None, None, None) if not found or on error.
    """
    try:
        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": api_key, "query": title, "include_adult": "false"},
            timeout=timeout,
        )
        if not resp.ok:
            return None, None, None
        data = resp.json() or {}
        results = data.get("results") or []
        if not results:
            return None, None, None
        lowered = title.lower().strip()
        best = None
        for r in results:
            if str(r.get("title", "")).lower().strip() == lowered:
                best = r
                break
        best = best or results[0]
        canon_title = best.get("title") or best.get("original_title")
        rd = best.get("release_date") or None
        tmdb_id = best.get("id")
        if rd:
            try:
                datetime.strptime(rd, "%Y-%m-%d")
            except Exception:
                rd = None
        return canon_title, rd, tmdb_id
    except Exception:
        return None, None, None


def _slugify(value: str) -> str:
    import re

    v = value.strip().lower()
    v = re.sub(r"[^a-z0-9]+", "-", v)
    v = re.sub(r"-+", "-", v).strip("-")
    return v


def refresh_catalog_window(
    conn,
    api_key: str,
    days_past: int = 5,
    days_future: int = 90,
    region: str = "US",
    language: str = "en-US",
    page_limit: int = 5,
    timeout: int = 12,
) -> int:
    """Refresh the rolling catalog of movies in the window.

    Returns number of titles upserted. Limits to `page_limit` pages to keep
    runtime and API costs modest.
    """
    from datetime import date, timedelta

    start = date.today() - timedelta(days=days_past)
    end = date.today() + timedelta(days=days_future)
    start_s = start.isoformat()
    end_s = end.isoformat()

    base = "https://api.themoviedb.org/3/discover/movie"
    upserted = 0
    for page in range(1, page_limit + 1):
        try:
            resp = requests.get(
                base,
                params={
                    "api_key": api_key,
                    "language": language,
                    "region": region,
                    "sort_by": "primary_release_date.asc",
                    "include_adult": "false",
                    "include_video": "false",
                    "page": page,
                    "primary_release_date.gte": start_s,
                    "primary_release_date.lte": end_s,
                },
                timeout=timeout,
            )
            if not resp.ok:
                break
            data = resp.json()
            results = data.get("results") or []
            if not results:
                break
            for r in results:
                title = (r.get("title") or r.get("original_title") or "").strip()
                rd = (r.get("release_date") or "").strip() or None
                tmdb_id = r.get("id")
                if not title:
                    continue
                slug = _slugify(title)
                ts = datetime.utcnow().isoformat()
                conn.execute(
                    "INSERT INTO catalog(slug, title, release_date, tmdb_id, updated_at) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(slug) DO UPDATE SET title=excluded.title, release_date=excluded.release_date, tmdb_id=excluded.tmdb_id, updated_at=excluded.updated_at",
                    (slug, title, rd or "", tmdb_id, ts),
                )
                upserted += 1
            conn.commit()
            total_pages = int(data.get("total_pages") or 1)
            if page >= total_pages:
                break
        except Exception:
            break
    return upserted


def load_catalog_index(conn) -> Dict[str, Tuple[str, Optional[str]]]:
    rows = conn.execute("SELECT slug, title, NULLIF(release_date,'') FROM catalog").fetchall()
    return {slug: (title, rd) for slug, title, rd in rows}


def match_movie_from_url(url: str, catalog: Dict[str, Tuple[str, Optional[str]]]) -> Optional[Tuple[str, Optional[str]]]:
    """Return (title, release_date) if any catalog slug appears in the URL path.

    Uses longest slug match to reduce false positives.
    """
    try:
        from urllib.parse import urlparse

        p = urlparse(url)
        path = (p.path or "").lower()
    except Exception:
        path = (url or "").lower()

    best = None
    best_len = 0
    for slug, (title, rd) in catalog.items():
        if slug and slug in path and len(slug) > best_len:
            best = (title, rd)
            best_len = len(slug)
    return best
