"""AnySearch web provider: web and vertical search plus page extraction.

AnySearch (https://www.anysearch.com) returns search results that already
carry page text, so `search` needs no second fetch per result. `fetch` goes
through AnySearch's extract endpoint, and `fetch_many` runs extract calls
concurrently for the batch fetch waves.

Configuration:
    # in .hyperresearch/config.toml
    [web]
    provider = "anysearch"

    # optional: higher rate limits (https://anysearch.com/console/api-keys)
    export ANYSEARCH_API_KEY="your-api-key"

    # optional: vertical search, e.g. "academic.paper" or "code.repo"
    export ANYSEARCH_TAG="academic.paper"

Without a key, requests go to api.anysearch.com anonymously at lower rate
limits. The provider is only used when it is selected in config.

No extra install: uses the httpx dependency the core already carries.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import httpx

from hyperresearch.web.base import WebResult

_BASE_URL = "https://api.anysearch.com"
_SEARCH_PATH = "/v1/search"
_EXTRACT_PATH = "/v1/extract"
_TIMEOUT = 30.0
_MAX_RESULTS_CAP = 10  # the API's per-request ceiling


class AnySearchProvider:
    """Web provider backed by the AnySearch REST API.

    Supports `search` (general or vertical), `fetch` (URL extraction) and
    `fetch_many` (concurrent extraction).
    """

    name = "anysearch"

    def __init__(
        self,
        api_key: str | None = None,
        tag: str | None = None,
        max_characters: int = 8000,
        max_workers: int = 4,
        transport: httpx.BaseTransport | None = None,
    ):
        key = api_key or os.environ.get("ANYSEARCH_API_KEY", "").strip()
        headers = {"User-Agent": "hyperresearch"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(
            base_url=_BASE_URL, headers=headers, timeout=_TIMEOUT, transport=transport
        )
        self._tag = tag or os.environ.get("ANYSEARCH_TAG", "").strip() or None
        self._max_characters = max_characters
        self._max_workers = max(1, max_workers)

    def search(self, query: str, max_results: int = 5) -> list[WebResult]:
        """Search via AnySearch and return ranked results with page text."""
        payload: dict[str, Any] = {
            "query": query,
            "max_results": max(1, min(max_results, _MAX_RESULTS_CAP)),
        }
        if self._tag:
            payload["tag"] = self._tag
        data = self._post(_SEARCH_PATH, payload)
        items = data.get("results") or []

        results: list[WebResult] = []
        for item in items[:max_results]:
            result = _to_web_result(item, self._max_characters)
            if result.url:
                results.append(result)
        return results

    def fetch(self, url: str) -> WebResult:
        """Extract a single URL via AnySearch and return its text."""
        data = self._post(_EXTRACT_PATH, {"url": url})
        content = (data.get("content") or "").strip()[: self._max_characters]
        if not content:
            raise RuntimeError(f"AnySearch returned no content for {url}")
        return WebResult(
            url=data.get("url") or url,
            title=data.get("title") or "",
            content=content,
            fetched_at=datetime.now(UTC),
        )

    def fetch_many(self, urls: list[str]) -> list[WebResult]:
        """Extract several URLs concurrently.

        Raises on the first failed URL so the batch caller falls back to
        per-URL fetches and reports exactly which URLs were lost.
        """
        if not urls:
            return []
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(urls))) as pool:
            return list(pool.map(self.fetch, urls))

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post(path, json=payload)
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict):
            raise RuntimeError("AnySearch returned a malformed response")
        if body.get("code", 0) != 0:
            message = body.get("message") or "unknown error"
            raise RuntimeError(f"AnySearch error: {message}")
        data = body.get("data")
        return data if isinstance(data, dict) else {}


def _to_web_result(item: dict[str, Any], max_characters: int) -> WebResult:
    """Convert an AnySearch search result dict into a hyperresearch WebResult."""
    content = item.get("content") or item.get("snippet") or ""
    metadata: dict[str, Any] = {}
    snippet = item.get("snippet")
    if snippet:
        metadata["snippet"] = snippet
    return WebResult(
        url=item.get("url", "") or "",
        title=item.get("title", "") or "",
        content=content[:max_characters],
        fetched_at=datetime.now(UTC),
        metadata=metadata,
    )
