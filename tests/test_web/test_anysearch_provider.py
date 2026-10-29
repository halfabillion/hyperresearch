"""Tests for the AnySearch web provider: offline via the httpx transport seam."""

from __future__ import annotations

import json
import threading
from typing import Any

import httpx
import pytest

from hyperresearch.web.anysearch_provider import AnySearchProvider

SEARCH_PATH = "/v1/search"
EXTRACT_PATH = "/v1/extract"


def _ok(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"code": 0, "message": "success", "data": data})


def _search_item(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "title": "Example Article",
        "url": "https://example.com/article",
        "snippet": "Snippet of the article body.",
        "content": "# Example Article\n\nFull page text.",
    }
    item.update(overrides)
    return item


def _transport(
    seen: list[httpx.Request],
    search_items: list[dict[str, Any]] | None = None,
    extract: Any = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == SEARCH_PATH:
            return _ok({"results": search_items or []})
        if request.url.path == EXTRACT_PATH:
            url = json.loads(request.content)["url"]
            if callable(extract):
                return extract(url)
            return extract
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)
    monkeypatch.delenv("ANYSEARCH_TAG", raising=False)


def test_provider_registered_via_factory() -> None:
    from hyperresearch.web.base import get_provider

    prov = get_provider("anysearch")
    assert prov.name == "anysearch"


def test_search_maps_results_and_sends_no_auth_without_key() -> None:
    seen: list[httpx.Request] = []
    transport = _transport(seen, [_search_item()])

    results = AnySearchProvider(transport=transport).search("q", max_results=3)

    assert len(results) == 1
    r = results[0]
    assert r.url == "https://example.com/article"
    assert r.title == "Example Article"
    assert r.content == "# Example Article\n\nFull page text."
    assert r.metadata == {"snippet": "Snippet of the article body."}

    (request,) = seen
    assert "Authorization" not in request.headers
    assert json.loads(request.content) == {"query": "q", "max_results": 3}


def test_api_key_from_env_is_sent_as_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANYSEARCH_API_KEY", "as-test")
    seen: list[httpx.Request] = []

    AnySearchProvider(transport=_transport(seen, [_search_item()])).search("q")

    assert seen[0].headers["Authorization"] == "Bearer as-test"


def test_vertical_tag_from_env_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANYSEARCH_TAG", "academic.paper")
    seen: list[httpx.Request] = []

    AnySearchProvider(transport=_transport(seen, [_search_item()])).search("q")

    assert json.loads(seen[0].content)["tag"] == "academic.paper"


def test_search_clamps_max_results_and_skips_items_without_url() -> None:
    seen: list[httpx.Request] = []
    items = [_search_item(url=""), _search_item(), _search_item(url="https://example.com/2")]

    results = AnySearchProvider(transport=_transport(seen, items)).search("q", max_results=50)

    assert json.loads(seen[0].content)["max_results"] == 10
    assert [r.url for r in results] == ["https://example.com/article", "https://example.com/2"]


def test_search_falls_back_to_snippet_when_content_missing() -> None:
    transport = _transport([], [_search_item(content="")])

    results = AnySearchProvider(transport=transport).search("q")

    assert results[0].content == "Snippet of the article body."


def test_error_envelope_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 429, "message": "Rate limited", "data": None})

    prov = AnySearchProvider(transport=httpx.MockTransport(handler))

    with pytest.raises(RuntimeError, match="Rate limited"):
        prov.search("q")


def test_http_error_raises() -> None:
    prov = AnySearchProvider(transport=httpx.MockTransport(lambda r: httpx.Response(500)))

    with pytest.raises(httpx.HTTPStatusError):
        prov.fetch("https://example.com/p")


def test_fetch_maps_extract_and_clips() -> None:
    seen: list[httpx.Request] = []
    extract = _ok({"url": "https://example.com/p", "title": "Page", "content": "x" * 50})

    r = AnySearchProvider(max_characters=20, transport=_transport(seen, extract=extract)).fetch(
        "https://example.com/p"
    )

    assert (r.url, r.title, r.content) == ("https://example.com/p", "Page", "x" * 20)
    assert json.loads(seen[0].content) == {"url": "https://example.com/p"}


def test_fetch_no_content_raises() -> None:
    transport = _transport([], extract=_ok({"url": "https://example.com/p", "content": "  "}))

    with pytest.raises(RuntimeError, match="no content"):
        AnySearchProvider(transport=transport).fetch("https://example.com/p")


def test_fetch_many_runs_concurrently_and_keeps_order() -> None:
    urls = [f"https://example.com/{i}" for i in range(4)]
    barrier = threading.Barrier(4, timeout=5)

    def extract(url: str) -> httpx.Response:
        barrier.wait()  # only passes if all four requests are in flight at once
        return _ok({"url": url, "title": url, "content": f"text {url}"})

    results = AnySearchProvider(max_workers=4, transport=_transport([], extract=extract)).fetch_many(
        urls
    )

    assert [r.url for r in results] == urls


def test_fetch_many_raises_when_any_url_fails() -> None:
    def extract(url: str) -> httpx.Response:
        if url.endswith("/bad"):
            return _ok({"url": url, "content": ""})
        return _ok({"url": url, "content": "ok"})

    prov = AnySearchProvider(transport=_transport([], extract=extract))

    with pytest.raises(RuntimeError, match="no content"):
        prov.fetch_many(["https://example.com/good", "https://example.com/bad"])


def test_fetch_many_empty() -> None:
    assert AnySearchProvider(transport=_transport([])).fetch_many([]) == []
