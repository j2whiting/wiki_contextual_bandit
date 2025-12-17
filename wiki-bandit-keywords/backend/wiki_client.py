from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import requests

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


@dataclass
class WikiPageSummary:
    pageid: int
    title: str
    extract: str
    url: str
    thumbnail_url: Optional[str] = None


class WikipediaClient:
    """Thin wrapper around the public Wikipedia API."""

    def __init__(
        self,
        user_agent: str = "wiki-bandit-demo/0.1 (https://example.com; your-email@example.com)",
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _request(self, params: dict) -> dict:
        resp = self.session.get(WIKIPEDIA_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_page_summary_by_title(self, title: str) -> Optional[WikiPageSummary]:
        """Fetch first paragraph + canonical URL for a given title."""
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info|pageimages",
            "exintro": 1,
            "explaintext": 1,
            "titles": title,
            "inprop": "url",
            # Optional thumbnail image (not available for all pages)
            "piprop": "thumbnail",
            "pithumbsize": 320,
        }
        data = self._request(params)
        pages = data.get("query", {}).get("pages", {})

        for page in pages.values():
            if "missing" in page:
                return None
            extract = page.get("extract", "") or ""
            fullurl = page.get("fullurl", "")
            thumbnail = page.get("thumbnail") or {}
            thumbnail_url = thumbnail.get("source")
            return WikiPageSummary(
                pageid=int(page["pageid"]),
                title=page.get("title", title),
                extract=extract,
                url=fullurl,
                thumbnail_url=thumbnail_url,
            )
        return None

    def get_random_articles(self, limit: int = 10) -> List[WikiPageSummary]:
        """Get a batch of random articles (main namespace)."""
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": limit,
        }
        data = self._request(params)
        random_pages = data.get("query", {}).get("random", [])

        summaries: List[WikiPageSummary] = []
        for page in random_pages:
            title = page["title"]
            summary = self.get_page_summary_by_title(title)
            if summary:
                summaries.append(summary)
        return summaries

    def get_linked_articles(self, pageid: int, max_articles: int = 10) -> List[WikiPageSummary]:
        """Get summaries for articles linked from a given page (out-links)."""
        params = {
            "action": "query",
            "format": "json",
            "prop": "links",
            "pageids": pageid,
            "plnamespace": 0,
            "pllimit": "max",
        }
        data = self._request(params)
        pages = data.get("query", {}).get("pages", {})

        link_titles: List[str] = []
        for page in pages.values():
            links = page.get("links", [])
            for link in links:
                if len(link_titles) >= max_articles:
                    break
                link_titles.append(link["title"])
            break

        summaries: List[WikiPageSummary] = []
        for title in link_titles:
            summary = self.get_page_summary_by_title(title)
            if summary:
                summaries.append(summary)
        return summaries

    def search_articles(self, query_text: str, limit: int = 10) -> List[WikiPageSummary]:
        """Use Wikipedia's text search for global keyword-driven retrieval."""
        query_text = (query_text or "").strip()
        if not query_text:
            return []

        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query_text,
            "srnamespace": 0,
            "srlimit": limit,
        }
        data = self._request(params)
        search_results = data.get("query", {}).get("search", [])

        summaries: List[WikiPageSummary] = []
        for item in search_results:
            title = item.get("title")
            if not title:
                continue
            summary = self.get_page_summary_by_title(title)
            if summary:
                summaries.append(summary)
        return summaries

    def get_page_html(self, pageid: int) -> Optional[dict]:
        """Fetch the full HTML content of a Wikipedia page for in-app viewing."""
        params = {
            "action": "parse",
            "format": "json",
            "pageid": pageid,
            "prop": "text|displaytitle",
            "disableeditsection": 1,
            "disabletoc": 0,
        }
        try:
            data = self._request(params)
            parse = data.get("parse", {})
            if not parse:
                return None
            html = parse.get("text", {}).get("*", "")
            title = parse.get("displaytitle", "") or parse.get("title", "")
            return {
                "pageid": pageid,
                "title": title,
                "html": html,
            }
        except Exception:
            return None
