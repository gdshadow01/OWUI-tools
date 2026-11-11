"""Reddit backend for websearch-mcp — anonymous .rss Atom feeds.

Why this exists: reddit.com HTML and .json endpoints are edge-blocked from
datacentre IPs (403 or a JS shell that never hydrates), even through the
tier-1 (curl_cffi) and tier-2 (CDP browser) escalation — solve-and-bounce
cookies unlock the HTML shell with the post body, but comments never arrive
and .json stays blocked. The only unauthenticated route Reddit still serves
to server IPs are the public Atom feeds (.rss): the post plus up to ~100
comments, no scores, no reply nesting (verified 2026-09-27).

Anonymous feeds are throttled to roughly one request per minute per IP, so
every outbound request goes through a shared gate (serialised, min-interval,
429-aware) fronted by a TTL cache. Concurrent callers get a cache hit, wait
a short budget (5 s), or an honest rate-limit error with retry guidance —
the MCP client timeout is ~30 s, so nothing here may sleep a full minute.
"""

from __future__ import annotations

import asyncio
import html
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Awaitable, Callable, Optional

import httpx

REDDIT_WWW = "https://www.reddit.com"
REDDIT_HOSTS = {"reddit.com", "www.reddit.com", "old.reddit.com", "np.reddit.com", "new.reddit.com"}
LISTING_SORTS = {"hot", "new", "top", "rising"}
SEARCH_SORTS = {"relevance", "new", "top", "comments"}
TIME_RANGES = {"hour", "day", "week", "month", "year", "all"}

USER_AGENT = "websearch-mcp/1.0 (reddit .rss backend; self-hosted MCP server)"
FETCH_TIMEOUT = 20.0        # seconds per outbound feed request
WAIT_BUDGET = 5.0           # max seconds a caller waits for the rate-limit gate
FEED_LIMIT = 100            # reddit honours limit up to ~100 (94 observed live)
CACHE_MAX_ENTRIES = 128

_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_THREAD_RE = re.compile(r"^/r/([^/]+)/comments/([a-z0-9]+)", re.I)
_USER_PATH_RE = re.compile(r"^/(?:user|u)/([^/]+)/?$", re.I)
_SUB_PATH_RE = re.compile(r"^/r/([^/]+)(?:/(hot|new|top|rising))?/?$", re.I)


def parse_reddit_url(url: str) -> Optional[dict]:
    """Classify a URL as an interceptable Reddit target (see module docstring)."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    if (parts.hostname or "").lower() not in REDDIT_HOSTS:
        return None
    path = parts.path
    m = _THREAD_RE.match(path)
    if m:
        return {"mode": "thread", "sub": m.group(1), "post_id": m.group(2)}
    m = _SUB_PATH_RE.match(path)
    if m:
        t = urllib.parse.parse_qs(parts.query).get("t", [None])[0]
        return {"mode": "listing", "sub": m.group(1),
                "sort": (m.group(2) or "hot").lower(),
                "time": t if t in TIME_RANGES else None}
    m = _USER_PATH_RE.match(path)
    if m:
        return {"mode": "user", "name": m.group(1)}
    return None


# ── Atom parsing ─────────────────────────────────────────────────────────────

def _strip_html(text: str | None) -> str:
    if not text:
        return ""
    text = _TAG_RE.sub(" ", html.unescape(text))
    text = re.sub(r"submitted by\s+/u/\S+|\[link\]|\[comments\]", " ", text)
    return _WS_RE.sub(" ", html.unescape(text)).strip()


def _parse_atom(data: bytes) -> list[dict]:
    root = ET.fromstring(data)
    entries = []
    for e in root.findall("a:entry", _ATOM):
        link = e.find("a:link", _ATOM)
        entries.append({
            "title": _strip_html(e.findtext("a:title", default="", namespaces=_ATOM)),
            "author": (e.findtext("a:author/a:name", default="", namespaces=_ATOM)
                       or "").removeprefix("/u/") or None,
            "created": e.findtext("a:updated", default="", namespaces=_ATOM) or None,
            "url": link.get("href") if link is not None else None,
            "body": _strip_html(e.findtext("a:content", default="", namespaces=_ATOM))[:4000],
        })
    return entries


# ── feed URL builders ────────────────────────────────────────────────────────

def _feed_url(parsed: dict, limit: int) -> str:
    limit = min(limit, FEED_LIMIT)
    if parsed["mode"] == "thread":
        return f"{REDDIT_WWW}/r/{parsed['sub']}/comments/{parsed['post_id']}/.rss?limit={limit}"
    if parsed["mode"] == "listing":
        params = {"limit": str(limit)}
        if parsed.get("time"):
            params["t"] = parsed["time"]
        return f"{REDDIT_WWW}/r/{parsed['sub']}/{parsed['sort']}.rss?{urllib.parse.urlencode(params)}"
    return f"{REDDIT_WWW}/user/{parsed['name']}/.rss?limit={limit}"


def _search_feed_url(query: str, subreddit: Optional[str], sort: str,
                     time_range: str, limit: int) -> str:
    params = {"q": query, "sort": sort, "t": time_range, "limit": str(min(limit, FEED_LIMIT))}
    if subreddit:
        params["restrict_sr"] = "1"
        path = f"/r/{subreddit}/search"
    else:
        path = "/search"
    return f"{REDDIT_WWW}{path}.rss?{urllib.parse.urlencode(params)}"


# ── rendering (fetch_page contract) ─────────────────────────────────────────

def _truncate(content: str, max_length: int) -> tuple[str, bool]:
    if len(content) <= max_length:
        return content, False
    return content[:max_length] + "... [content truncated]", True


def render_thread(entries: list[dict], url: str, max_length: int) -> dict:
    """First feed entry is the post; the rest are top-level comments."""
    if not entries:
        return {"url": url, "title": "", "content": f"Reddit feed returned no entries for {url}",
                "content_length": 0, "truncated": False, "via": "reddit-rss"}
    post, comments = entries[0], entries[1:]
    lines = [
        f"# {post['title']}",
        f"u/{post['author']} · {post.get('created') or ''} · {url}",
        "",
        post["body"],
        "",
        f"## Comments ({len(comments)} in feed — anonymous .rss: no scores, no reply nesting)",
    ]
    for c in comments:
        lines.append(f"- u/{c['author']}: {c['body'][:600]}")
    content, truncated = _truncate("\n".join(lines), max_length)
    return {"url": url, "title": post["title"], "content": content,
            "content_length": len(content), "truncated": truncated, "via": "reddit-rss"}


def render_listing(title: str, entries: list[dict], url: str, max_length: int) -> dict:
    lines = [f"# {title}", ""]
    for e in entries:
        lines.append(f"- {e['title']} — u/{e['author']}")
        if e["url"]:
            lines.append(f"  {e['url']}")
        if e["body"]:
            lines.append(f"  {e['body'][:300]}")
    content, truncated = _truncate("\n".join(lines) or "(no entries)", max_length)
    return {"url": url, "title": title, "content": content,
            "content_length": len(content), "truncated": truncated, "via": "reddit-rss"}


# ── outbound fetch + rate-limit gate ─────────────────────────────────────────

class _RateLimited(Exception):
    def __init__(self, retry_after: int):
        super().__init__(f"HTTP 429, retry after {retry_after}s")
        self.retry_after = retry_after


def _reset_seconds(headers: dict) -> int:
    for key in ("x-ratelimit-reset", "retry-after"):
        val = headers.get(key)
        if val:
            try:
                return max(1, min(int(float(val)) + 1, 120))
            except ValueError:
                continue
    return 61


async def _fetch_feed(url: str) -> tuple[bytes, dict]:
    """Real feed fetch. Raises _RateLimited on 429, httpx errors otherwise.

    NOTE: deliberately no browser/stealth escalation — the browser tier
    cannot read comments (verified 2026-09-27); an honest error beats a
    45 s doomed escalation.
    """
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/atom+xml,*/*"})
        if resp.status_code == 429:
            raise _RateLimited(_reset_seconds(dict(resp.headers)))
        resp.raise_for_status()
        return resp.content, dict(resp.headers)


class RedditGate:
    """Serialises outbound Reddit feed requests: min-interval + TTL cache.

    Callers get cached entries, wait up to WAIT_BUDGET for the next free
    slot, or an error dict with retry guidance. Never sleeps a full minute
    (MCP client timeout is ~30 s).
    """

    def __init__(self, min_interval: float = 65.0, cache_ttl: float = 600.0):
        self.min_interval = min_interval
        self.cache_ttl = cache_ttl
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0
        self._cache: "OrderedDict[str, tuple[float, list[dict]]]" = OrderedDict()
        self.last_success: Optional[float] = None

    def _cached(self, feed_url: str, now: float) -> Optional[list[dict]]:
        hit = self._cache.get(feed_url)
        if hit is not None and now - hit[0] < self.cache_ttl:
            self._cache.move_to_end(feed_url)
            return hit[1]
        return None

    async def get(self, feed_url: str,
                  fetch: Callable[[str], Awaitable[tuple[bytes, dict]]],
                  clock: Callable[[], float] = time.monotonic
                  ) -> tuple[Optional[list[dict]], Optional[dict]]:
        now = clock()
        hit = self._cached(feed_url, now)
        if hit is not None:
            return hit, None
        async with self._lock:
            now = clock()
            hit = self._cached(feed_url, now)
            if hit is not None:
                return hit, None
            wait = self._next_allowed - now
            if wait > WAIT_BUDGET:
                return None, {"error": f"reddit rate limit: window busy, retry in {int(wait) + 1}s",
                              "retry_after": int(wait) + 1}
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                data, _headers = await fetch(feed_url)
            except _RateLimited as exc:
                self._next_allowed = clock() + exc.retry_after
                return None, {"error": f"reddit rate limited (429); retry in {exc.retry_after}s",
                              "retry_after": exc.retry_after}
            except Exception as exc:  # httpx errors, XML parse errors
                return None, {"error": f"reddit feed fetch failed: {exc}"}
            self._next_allowed = clock() + self.min_interval
            entries = _parse_atom(data)
            self._cache[feed_url] = (clock(), entries)
            while len(self._cache) > CACHE_MAX_ENTRIES:
                self._cache.popitem(last=False)
            self.last_success = time.time()
            return entries, None


_gate: Optional[RedditGate] = None


def get_gate(min_interval: float, cache_ttl: float) -> RedditGate:
    global _gate
    if _gate is None:
        _gate = RedditGate(min_interval, cache_ttl)
    return _gate


def gate_status() -> dict:
    if _gate is None:
        return {"enabled": False, "cached_feeds": 0}
    return {"enabled": True, "min_interval_s": _gate.min_interval,
            "cache_ttl_s": _gate.cache_ttl, "cached_feeds": len(_gate._cache),
            "last_success": _gate.last_success}


# ── public API ───────────────────────────────────────────────────────────────

async def fetch_reddit(parsed: dict, url: str, max_length: int,
                       min_interval: float, cache_ttl: float) -> dict:
    """Serve a classified reddit URL (thread/listing/user) for fetch_page."""
    gate = get_gate(min_interval, cache_ttl)
    entries, err = await gate.get(_feed_url(parsed, FEED_LIMIT), _fetch_feed)
    if err is not None:
        return {"url": url, "title": "", "content_length": 0, "truncated": False, **err}
    if parsed["mode"] == "thread":
        return render_thread(entries, url, max_length)
    if parsed["mode"] == "listing":
        return render_listing(f"r/{parsed['sub']} ({parsed['sort']})", entries, url, max_length)
    return render_listing(f"u/{parsed['name']} recent activity", entries, url, max_length)


async def search_feeds(query: str, subreddit: Optional[str], sort: str, time_range: str,
                       limit: int, min_interval: float, cache_ttl: float
                       ) -> tuple[Optional[list[dict]], Optional[dict]]:
    gate = get_gate(min_interval, cache_ttl)
    return await gate.get(_search_feed_url(query, subreddit, sort, time_range, limit), _fetch_feed)
