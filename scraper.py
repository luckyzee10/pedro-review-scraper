"""Scraper utilities for fetching movie review items from RSS feeds.

Uses feedparser to read feeds and applies light heuristics to decide whether
an entry is a movie review. Normalizes entries to a simple dict shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Any
from urllib.parse import urlparse

import feedparser
import os


# Default feeds provided by the user
FEEDS: List[str] = [
    "https://www.nytimes.com/svc/collections/v1/publish/arts/rss.xml",
    "https://www.latimes.com/entertainment/rss2.0.xml",
    "https://variety.com/feed/",
    "https://www.indiewire.com/c/film/feed/",
    "https://www.hollywoodreporter.com/c/movies/feed/",
    "https://www.theguardian.com/film/rss",
    "https://www.rogerebert.com/feed",
    "https://www.rollingstone.com/movies/feed/",
    # Expanded outlets (best-effort; failures are ignored gracefully)
    "https://www.avclub.com/rss",
    "https://theplaylist.net/feed/",
    "https://www.slashfilm.com/feed/",
    "https://www.vulture.com/rss/movies/index.xml",
    "https://www.telegraph.co.uk/films/rss.xml",
    "https://www.independent.co.uk/arts-entertainment/films/rss",
    "https://feeds.ign.com/ign/movies",
    "https://screenrant.com/feed/",
]


@dataclass
class ScrapedItem:
    outlet: str
    title: str
    summary: str
    link: str
    published: str
    feed_url: str


def _guess_outlet(feed: Dict[str, Any], feed_url: str) -> str:
    title = (
        (feed.get("feed") or {}).get("title")
        or (feed.get("channel") or {}).get("title")
        or ""
    )
    if title:
        return str(title).strip()
    parsed = urlparse(feed_url)
    return parsed.netloc or feed_url


def _is_review_candidate(title: str, summary: str, link: str) -> bool:
    """Decide if an entry is a review worth processing.

    Permanent cues (case-insensitive) in title/summary/URL required:
    - "review", "critic review", "film review", "our take on", "verdict"
    - a star cue: the unicode star "★" or patterns like "3 stars", "4 star"
    """
    title_l = (title or "").lower()
    summary_l = (summary or "").lower()
    link_l = (link or "").lower()
    hay = f"{title_l} {summary_l} {link_l}"

    phrase_cues = [
        "review",
        "critic review",
        "film review",
        "our take on",
        "verdict",
        "stars",  # e.g., "3 stars" or "four stars"
    ]

    # Fast path: unicode star anywhere
    if "★" in (title or "") or "★" in (summary or ""):
        return True

    # Numeric star rating like "3 stars" or "4 star"
    import re as _re

    if _re.search(r"\b[1-5]\s*star(s)?\b", hay):
        return True

    # Any phrase cue present
    if any(cue in hay for cue in phrase_cues):
        return True

    return False


def fetch_feed(feed_url: str) -> List[ScrapedItem]:
    parsed = feedparser.parse(feed_url)
    outlet = _guess_outlet(parsed, feed_url)
    items: List[ScrapedItem] = []
    for e in parsed.get("entries", []):
        title = str(e.get("title") or "").strip()
        summary = str(e.get("summary") or e.get("description") or "").strip()
        link = str(e.get("link") or "").strip()
        published = str(
            e.get("published") or e.get("updated") or e.get("pubDate") or ""
        ).strip()

        if not title:
            continue
        if not _is_review_candidate(title, summary, link):
            continue

        items.append(
            ScrapedItem(
                outlet=outlet,
                title=title,
                summary=summary,
                link=link,
                published=published,
                feed_url=feed_url,
            )
        )
    return items


def fetch_all(feeds: Iterable[str] = FEEDS) -> List[ScrapedItem]:
    results: List[ScrapedItem] = []
    for url in feeds:
        try:
            results.extend(fetch_feed(url))
        except Exception:
            # Be resilient to malformed feeds or transient parse errors
            continue
    return results
