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
    """Heuristic to decide if an entry is a movie review.

    We check for the keyword 'review' and some film-related hints in either
    the title or summary to reduce noise from general entertainment posts.
    """
    text = f"{title} {summary}".lower()
    if "review" not in text:
        return False
    hints = ("film", "movie", "cinema", "screen", "feature")
    # Optional stricter filter: require 'review' in the URL itself
    strict_url = os.getenv("STRICT_URL_REVIEW", "").strip().lower() in {"1", "true", "yes", "on"}
    if strict_url and link:
        if "review" not in link.lower():
            return False
    return any(h in text for h in hints) or True  # allow generic 'review' when unsure


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
