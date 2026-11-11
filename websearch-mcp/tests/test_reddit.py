"""Unit tests for the Reddit .rss backend (no network access needed)."""

import asyncio
import json

import pytest

import httpx

import reddit_backend as rb
import server


# ── parse_reddit_url ──────────────────────────────────────────────────────────

def test_parse_thread_with_slug_and_query():
    parsed = rb.parse_reddit_url(
        "https://www.reddit.com/r/LocalLLaMA/comments/1pwh0q9/best_local_llms_2025/?utm_source=x")
    assert parsed == {"mode": "thread", "sub": "LocalLLaMA", "post_id": "1pwh0q9"}

def test_parse_thread_no_slug():
    assert rb.parse_reddit_url("https://www.reddit.com/r/x/comments/abc123/") == {
        "mode": "thread", "sub": "x", "post_id": "abc123"}

def test_parse_listing_plain():
    assert rb.parse_reddit_url("https://www.reddit.com/r/announcements/") == {
        "mode": "listing", "sub": "announcements", "sort": "hot", "time": None}

def test_parse_listing_sort_and_t():
    assert rb.parse_reddit_url("https://reddit.com/r/foo/top/?t=week") == {
        "mode": "listing", "sub": "foo", "sort": "top", "time": "week"}

def test_parse_listing_invalid_t_dropped():
    assert rb.parse_reddit_url("https://www.reddit.com/r/foo/?t=bogus")["time"] is None

def test_parse_user_and_u_shortform():
    assert rb.parse_reddit_url("https://www.reddit.com/user/spez/") == {
        "mode": "user", "name": "spez"}
    assert rb.parse_reddit_url("https://old.reddit.com/u/spez") == {
        "mode": "user", "name": "spez"}

def test_parse_old_reddit_thread():
    assert rb.parse_reddit_url("https://old.reddit.com/r/x/comments/abc/t/")["mode"] == "thread"

def test_parse_non_reddit_and_unknown_paths():
    assert rb.parse_reddit_url("https://example.com/r/x/comments/abc/") is None
    assert rb.parse_reddit_url("https://www.reddit.com/r/x/wiki/index") is None
    assert rb.parse_reddit_url("https://www.reddit.com/search?q=x") is None


# ── atom parsing + rendering ─────────────────────────────────────────────────

_ATOM_FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>x : announcements</title>
  <entry>
    <title>Post title</title>
    <link href="https://www.reddit.com/r/x/comments/abc/post/"/>
    <updated>2026-09-01T10:00:00+00:00</updated>
    <author><name>/u/poster</name></author>
    <content type="html">&lt;table&gt;&lt;tr&gt;&lt;td&gt;submitted by
      &lt;a href="/u/poster"&gt;/u/poster&lt;/a&gt; [link] [comments]&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;
      &lt;div&gt;Post &amp;amp; body text&lt;/div&gt;</content>
  </entry>
  <entry>
    <title>/u/commenter on Post title</title>
    <link href="https://www.reddit.com/r/x/comments/abc/post/h1/"/>
    <updated>2026-09-01T11:00:00+00:00</updated>
    <author><name>/u/commenter</name></author>
    <content type="html">&lt;div&gt;First comment&lt;/div&gt;</content>
  </entry>
</feed>"""


def test_parse_atom_entries():
    entries = rb._parse_atom(_ATOM_FIXTURE)
    assert len(entries) == 2
    post = entries[0]
    assert post["title"] == "Post title"
    assert post["author"] == "poster"          # /u/ prefix stripped
    assert post["url"].endswith("/abc/post/")
    assert "Post & body text" in post["body"]  # html-escaped, footer stripped
    assert "[link]" not in post["body"]


def test_render_thread():
    entries = rb._parse_atom(_ATOM_FIXTURE)
    out = rb.render_thread(entries, "https://www.reddit.com/r/x/comments/abc/", 5000)
    assert out["title"] == "Post title"
    assert out["via"] == "reddit-rss"
    assert out["truncated"] is False
    assert "# Post title" in out["content"]
    assert "- u/commenter: First comment" in out["content"]
    assert "Comments (1" in out["content"]


def test_render_thread_truncates():
    entries = rb._parse_atom(_ATOM_FIXTURE)
    out = rb.render_thread(entries, "u", 50)
    assert out["truncated"] is True
    assert out["content"].endswith("... [content truncated]")


def test_render_listing():
    entries = rb._parse_atom(_ATOM_FIXTURE)
    out = rb.render_listing("r/x (hot)", entries, "https://www.reddit.com/r/x/", 5000)
    assert "r/x (hot)" in out["content"]
    assert "Post title" in out["content"]


def test_feed_urls():
    assert rb._feed_url({"mode": "thread", "sub": "x", "post_id": "abc"}, 50) == \
        "https://www.reddit.com/r/x/comments/abc/.rss?limit=50"
    assert rb._feed_url({"mode": "listing", "sub": "foo", "sort": "top", "time": "week"}, 25) == \
        "https://www.reddit.com/r/foo/top.rss?limit=25&t=week"
    assert rb._feed_url({"mode": "user", "name": "spez"}, 10) == \
        "https://www.reddit.com/user/spez/.rss?limit=10"
    url = rb._search_feed_url("q x", "foo", "new", "all", 15)
    assert url.startswith("https://www.reddit.com/r/foo/search.rss?")
    assert "restrict_sr=1" in url and "sort=new" in url and "limit=15" in url
    assert rb._search_feed_url("q", None, "relevance", "all", 15).startswith(
        "https://www.reddit.com/search.rss?")


# ── gate: rate limit + cache (injected clock/fetch, no network) ──────────────


class FakeClock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def _mk_gate():
    gate = rb.RedditGate(min_interval=65.0, cache_ttl=600.0)
    clock = FakeClock()
    calls = []

    async def fetch(url):
        calls.append(url)
        return _ATOM_FIXTURE, {}

    return gate, clock, fetch, calls


def test_gate_first_call_fetches_and_caches():
    gate, clock, fetch, calls = _mk_gate()
    entries, err = asyncio.run(gate.get("https://x/.rss", fetch, clock))
    assert err is None and len(entries) == 2 and len(calls) == 1
    # immediate second call: cache hit, no fetch, no rate error
    entries, err = asyncio.run(gate.get("https://x/.rss", fetch, clock))
    assert err is None and len(calls) == 1


def test_gate_second_url_within_window_errors_with_retry_after():
    gate, clock, fetch, calls = _mk_gate()
    asyncio.run(gate.get("https://x/.rss", fetch, clock))
    entries, err = asyncio.run(gate.get("https://y/.rss", fetch, clock))
    assert entries is None and err is not None
    assert err["retry_after"] == 66         # next_allowed - now, +1s boundary buffer
    assert len(calls) == 1                   # second url NOT fetched


def test_gate_waits_short_remaining_interval():
    gate, clock, fetch, calls = _mk_gate()
    asyncio.run(gate.get("https://x/.rss", fetch, clock))
    clock.t += 62.0                          # 3 s left in window < WAIT_BUDGET
    entries, err = asyncio.run(gate.get("https://y/.rss", fetch, clock))
    assert err is None and len(calls) == 2   # waited, then fetched


def test_gate_429_pushes_window_out():
    gate, clock, _default_fetch, calls = _mk_gate()
    gate._next_allowed = 0.0

    async def fetch_429(url):
        calls.append(url)
        raise rb._RateLimited(90)

    entries, err = asyncio.run(gate.get("https://x/.rss", fetch_429, clock))
    assert "429" in err["error"] and err["retry_after"] == 90
    # next caller immediately: window now 90 s out > WAIT_BUDGET -> honest error
    _, err2 = asyncio.run(gate.get("https://y/.rss", fetch_429, clock))
    assert "retry" in err2["error"]


def test_gate_fetch_failure_generic_error():
    async def fetch_boom(url):
        raise httpx.ConnectError("nope")

    gate, clock, _, _ = _mk_gate()
    entries, err = asyncio.run(gate.get("https://x/.rss", fetch_boom, clock))
    assert entries is None and "failed" in err["error"]


def test_gate_status_before_and_after_gate_init(monkeypatch):
    # uninitialized singleton: backend reported off, no crash (regression:
    # live health_check silently swallowed AttributeError -> enabled:false)
    monkeypatch.setattr(rb, "_gate", None)
    assert rb.gate_status() == {"enabled": False, "cached_feeds": 0}

    # initialized + one cached feed: enabled with real cache size (regression:
    # gate_status read _gate.cache, raised AttributeError -> tool call failed)
    gate, clock, fetch, _ = _mk_gate()
    entries, err = asyncio.run(gate.get("https://x/.rss", fetch, clock))
    assert err is None and entries is not None and len(entries) == 2
    monkeypatch.setattr(rb, "_gate", gate)
    status = rb.gate_status()
    assert status["enabled"] is True
    assert status["cached_feeds"] == 1
    assert status["min_interval_s"] == 65.0
    assert status["cache_ttl_s"] == 600.0
    assert status["last_success"] is not None


# ── server integration ───────────────────────────────────────────────────────


def test_fetch_page_intercepts_reddit_thread(monkeypatch):
    sentinel = {"url": "x", "title": "t", "content": "c", "content_length": 1,
                "truncated": False, "via": "reddit-rss"}
    async def fake_fetch(parsed, url, max_length, mi, ttl):
        assert parsed["mode"] == "thread"
        return sentinel
    monkeypatch.setattr(server, "reddit_fetch", fake_fetch)
    out = asyncio.run(server._fetch_page(
        "https://www.reddit.com/r/x/comments/abc/t/", 5000))
    assert out is sentinel


def test_fetch_page_leaves_non_reddit_alone(monkeypatch):
    async def fail_fetch(*a, **k):
        raise AssertionError("reddit backend must not be called")
    monkeypatch.setattr(server, "reddit_fetch", fail_fetch)
    out = asyncio.run(server._fetch_page("http://127.0.0.1:1/", 5000))
    assert "Failed to fetch" in out["content"]


def test_fetch_page_reddit_disabled(monkeypatch):
    async def fail_fetch(*a, **k):
        raise AssertionError("reddit backend must not be called when disabled")
    monkeypatch.setattr(server, "reddit_fetch", fail_fetch)
    monkeypatch.setattr(server, "USE_REDDIT_BACKEND", False)
    out = asyncio.run(server._fetch_page("http://127.0.0.1:1/", 5000))
    assert "Failed to fetch" in out["content"]


def test_search_reddit_tool(monkeypatch):
    async def fake_search(query, subreddit, sort, tr, limit, mi, ttl):
        assert (query, subreddit, sort) == ("ollama", "LocalLLaMA", "new")
        return rb._parse_atom(_ATOM_FIXTURE), None
    monkeypatch.setattr(server, "search_feeds", fake_search)
    out = asyncio.run(server.search_reddit("ollama", subreddit="LocalLLaMA", sort="new"))
    parsed = json.loads(out)
    assert parsed["total_results"] == 2
    assert parsed["results"][0]["title"] == "Post title"


def test_search_reddit_rejects_bad_sort(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("must validate before fetching")
    monkeypatch.setattr(server, "search_feeds", boom)
    out = asyncio.run(server.search_reddit("q", sort="bogus"))
    assert "error" in json.loads(out)


def test_search_reddit_rate_limited_passthrough(monkeypatch):
    async def limited(*a, **k):
        return None, {"error": "reddit rate limited (429); retry in 90s", "retry_after": 90}
    monkeypatch.setattr(server, "search_feeds", limited)
    parsed = json.loads(asyncio.run(server.search_reddit("q")))
    assert parsed["retry_after"] == 90
