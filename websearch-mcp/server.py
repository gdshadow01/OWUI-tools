"""
Web Search MCP Server — Streamable HTTP Transport
===================================================
MCP server providing web search (via SearXNG) and page fetching
with content extraction (trafilatura + readability fallback).

Uses MCP Streamable HTTP transport (protocol version 2025-03-26) instead of
the deprecated HTTP+SSE or stdio transports.
"""

import asyncio
import os
import urllib.parse
from typing import Optional

import httpx
from starlette.applications import Starlette
from starlette.routing import Mount

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from readability import Document as ReadabilityDocument
except ImportError:
    ReadabilityDocument = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

import reddit_backend
from reddit_backend import (SEARCH_SORTS, TIME_RANGES, parse_reddit_url,
                            search_feeds)
from reddit_backend import fetch_reddit as reddit_fetch
from reddit_backend import gate_status as reddit_gate_status

# ── Configuration ────────────────────────────────────────────────────────────

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8040")
SEARXNG_TIMEOUT = int(os.getenv("SEARXNG_TIMEOUT", "15"))
USE_SEARXNG_SEARCH = os.getenv("USE_SEARXNG_SEARCH", "yes").lower() == "yes"

TIKA_SERVER_URL = os.getenv("TIKA_SERVER_URL", "")
USE_TIKA = bool(TIKA_SERVER_URL)

PAGE_FETCH_TIMEOUT = int(os.getenv("PAGE_FETCH_TIMEOUT", "15"))
PAGE_MAX_CONTENT_LENGTH = int(os.getenv("PAGE_MAX_CONTENT_LENGTH", "50000"))

MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "9200"))

# Tiered fetch escalation (tier 1 curl_cffi stealth, tier 2 browser CDP sidecar)
USE_BROWSER_TIER = os.getenv("USE_BROWSER_TIER", "yes").lower() == "yes"
BROWSER_CDP_URL = os.getenv("BROWSER_CDP_URL", "http://browser:9222")
ESCALATION_TIMEOUT = int(os.getenv("ESCALATION_TIMEOUT", "45"))

# Reddit .rss backend — anonymous Atom feeds (see reddit_backend.py docstring)
USE_REDDIT_BACKEND = os.getenv("USE_REDDIT_BACKEND", "yes").lower() == "yes"
REDDIT_MIN_INTERVAL = float(os.getenv("REDDIT_MIN_INTERVAL", "65"))
REDDIT_CACHE_TTL = float(os.getenv("REDDIT_CACHE_TTL", "600"))

_EMPTY_EXTRACT_CHARS = 500  # cleaned text shorter than this counts as an empty extraction
_SCRIPT_TAG_THRESHOLD = 5  # raw HTML with >= N "<script" occurrences counts as script-heavy
_BROWSER_MIN_HTML_CHARS = 5000  # rendered DOM must reach this size to count as challenge-free
_BROWSER_POLL_INTERVAL = 1.0  # seconds between challenge-clear polls in the browser tier
_BROWSER_POLL_BUDGET = 15.0  # total seconds spent polling for challenge clearance
_BROWSER_CHALLENGE_TITLES = (
    "just a moment",
    "checking your browser",
    "attention required",
    "verify you are human",
)

# Allowed Host header values for FastMCP's DNS-rebinding protection
# (Starlette TrustedHostMiddleware-equivalent). "*" disables protection and
# allows any host; otherwise a comma-separated list of allowed host values.
# Each entry may use a ":*" suffix as a port wildcard, e.g. "192.168.0.209:*".
_allowed_hosts_raw = os.getenv("ALLOWED_HOSTS", "*").strip()
if _allowed_hosts_raw == "*":
    # Disable host validation entirely (server reachable from any host).
    _transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    )
else:
    _allowed_hosts = [h.strip() for h in _allowed_hosts_raw.split(",") if h.strip()]
    _transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_allowed_hosts,
    )

# Common browser headers for fetching
_HEADERS = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    },
]

_FETCH_ATTEMPTS = [  # (index into _HEADERS, timeout_extra seconds)
    (0, 0),   # primary UA, base timeout
    (0, 5),   # primary UA, +5 s — slow host
    (1, 10),  # fallback UA, +10 s — UA-blocked and/or slow
]


# ── Search: SearXNG ──────────────────────────────────────────────────────────

async def _search_searxng(query: str, num_results: int) -> list[dict]:
    """Search via SearXNG JSON API."""
    if not SEARXNG_URL:
        return []

    params = {
        "q": query,
        "format": "json",
        "language": "auto",
        "safesearch": "0",
        "categories": "general",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=SEARXNG_TIMEOUT) as client:
        try:
            resp = await client.get(f"{SEARXNG_URL}/search", params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", []):
                title = item.get("title", "")
                url = item.get("url", "")
                snippet = item.get("content", "")
                if title and url:
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "engine": "searxng",
                    })
            return results[:num_results]
        except Exception as e:
            print(f"SearXNG search error: {e}", flush=True)
            return []


# ── Page Fetching ────────────────────────────────────────────────────────────

def _is_pdf_url(url: str) -> bool:
    return url.lower().endswith(".pdf") or ".pdf?" in url.lower()


async def _extract_pdf_with_tika(pdf_url: str) -> str:
    """Extract text from a PDF via Apache Tika (optional)."""
    if not USE_TIKA:
        return f"[PDF detected at {pdf_url} — Tika server not configured, cannot extract text]"

    async with httpx.AsyncClient(timeout=PAGE_FETCH_TIMEOUT, follow_redirects=True) as dl:
        resp = await dl.get(pdf_url, headers=_HEADERS[0])
        resp.raise_for_status()
        pdf_bytes = resp.content

    async with httpx.AsyncClient(timeout=PAGE_FETCH_TIMEOUT) as tika:
        resp = await tika.put(
            f"{TIKA_SERVER_URL}/tika",
            headers={"Accept": "text/plain", "Content-Type": "application/pdf"},
            content=pdf_bytes,
        )
        resp.raise_for_status()
        return resp.text


def _clean_html(html: str, url: str) -> str:
    """Extract main text content from HTML using trafilatura → readability → BS4 fallback."""
    text = None

    # 1. trafilatura (best quality)
    if trafilatura is not None:
        text = trafilatura.extract(html)

    # 2. readability + BS4
    if (not text or len(text.strip()) < 100) and ReadabilityDocument is not None:
        try:
            doc = ReadabilityDocument(html)
            summary_html = doc.summary()
            soup = BeautifulSoup(summary_html, "html.parser")
            text = soup.get_text(separator=" ")
        except Exception:
            pass

    # 3. Plain BS4 fallback
    if (not text or len(text.strip()) < 50) and BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

    if not text or len(text.strip()) < 50:
        return f"Page content could not be extracted from {url}. The page may contain mostly images, videos, or JavaScript."

    return text.strip()


def _extract_links(html: str, base_url: str, max_links: int = 200) -> list[dict]:
    """Extract navigational anchor links from raw HTML.

    Returns a list of {"text": ..., "href": ...} dicts with absolute URLs,
    deduplicated by (href, text) and capped at max_links.
    """
    if not html or BeautifulSoup is None:
        return []

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []

    seen = set()
    links: list[dict] = []

    for a in soup.find_all("a", href=True):
        if len(links) >= max_links:
            break

        href = a["href"].strip()
        if not href:
            continue
        # Skip non-navigational hrefs
        lower = href.lower()
        if lower.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        # Resolve relative → absolute
        absolute = urllib.parse.urljoin(base_url, href)
        if not absolute:
            continue

        text = a.get_text(strip=True)

        key = (absolute, text)
        if key in seen:
            continue
        seen.add(key)

        links.append({"text": text, "href": absolute})

    return links


# ── Stealth Fetch Escalation (tiers 1 & 2) ───────────────────────────────────
# Tier 0 is the plain httpx loop in _fetch_page (fast path). When it gets
# blocked (403/429) or returns a script-heavy page with no extractable text,
# we escalate: tier 1 retries with curl_cffi browser impersonation, tier 2
# renders the page in a real browser attached via CDP. Both heavy libraries
# are imported lazily so the module imports fine without them installed.

def _is_script_heavy(html: Optional[str]) -> bool:
    """Return True when raw HTML carries at least _SCRIPT_TAG_THRESHOLD script tags."""
    if not html:
        return False
    return html.lower().count("<script") >= _SCRIPT_TAG_THRESHOLD


def _looks_like_js_shell(html: str) -> bool:
    """Return True for the app-shell variant of an empty 200: substantial raw
    HTML (>= _BROWSER_MIN_HTML_CHARS) carrying at least one script tag that
    nonetheless extracts no text. Catches single-script custom-element apps
    (Reddit's shell: ~8KB HTML, 1 script, hydrated client-side) which the
    >=5-scripts heuristic alone misses."""
    return len(html) >= _BROWSER_MIN_HTML_CHARS and "<script" in html.lower()


def _is_empty_extraction(cleaned: Optional[str]) -> bool:
    """Return True when cleaned text is missing or suspiciously short."""
    return cleaned is None or len(cleaned.strip()) < _EMPTY_EXTRACT_CHARS


def _should_escalate_tier2(status: Optional[int], html: Optional[str], cleaned: Optional[str]) -> bool:
    """Decide whether a fetch result justifies escalating to the browser tier.

    Escalates on persisted 403/429 blocks, or on HTTP 200 responses whose
    extraction came up empty while the raw HTML looks like a JS challenge.
    """
    if status in (403, 429):
        return True
    if status == 200 and _is_empty_extraction(cleaned) and html is not None and (
        _is_script_heavy(html) or _looks_like_js_shell(html)
    ):
        return True
    return False


def _normalize_cookies(cookies: Optional[list[dict]]) -> list[tuple[str, str, str]]:
    """Normalize playwright cookie dicts to sorted, deduped (name, value, domain) tuples.

    Extra keys (path/expires/httpOnly/...) are ignored; the dedupe key is
    (name, domain) with the LAST occurrence winning; leading dots are stripped
    from domains.
    """
    merged: dict[tuple[str, str], str] = {}
    for cookie in cookies or []:
        name = str(cookie.get("name", ""))
        value = str(cookie.get("value", ""))
        domain = str(cookie.get("domain", "")).removeprefix(".")
        merged[(name, domain)] = value
    return sorted((name, value, domain) for (name, domain), value in merged.items())


def _cookies_for_request(cookies: list[tuple[str, str, str]], url: str) -> dict[str, str]:
    """Filter (name, value, domain) cookie tuples down to a {name: value} jar for url.

    A cookie matches when its domain equals the URL host or the host is a
    subdomain of it (host.endswith("." + domain)).
    """
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    jar: dict[str, str] = {}
    for name, value, domain in cookies:
        domain = domain.lower().removeprefix(".")
        if not domain or not host:
            continue
        if domain == host or host.endswith("." + domain):
            jar[name] = value
    return jar


async def _stealth_fetch(url: str, headers: dict, cookies: Optional[dict[str, str]] = None) -> tuple[Optional[str], Optional[int]]:
    """Tier 1: fetch via curl_cffi with Chrome TLS/JA3 impersonation.

    Returns (text, status_code) on success, (None, None) if curl_cffi is not
    installed or the request fails for any reason.
    """
    try:
        from curl_cffi.requests import AsyncSession
    except Exception:
        return None, None

    try:
        async with AsyncSession(impersonate="chrome") as session:
            resp = await session.get(
                url,
                headers=headers,
                allow_redirects=True,
                timeout=PAGE_FETCH_TIMEOUT,
                cookies=cookies,
            )
            return resp.text, resp.status_code
    except Exception as e:
        print(f"Stealth fetch error for {url}: {e}", flush=True)
        return None, None


_browser_singleton = None
_playwright_instance = None
_browser_lock = asyncio.Lock()


async def _get_browser():
    """Lazily connect to the CDP sidecar browser (cached singleton).

    Starts the playwright driver on first use, connects over CDP to
    BROWSER_CDP_URL and caches the Browser. Raises on failure; never returns
    a disconnected browser (a dead one is replaced by a fresh connection).
    """
    global _browser_singleton, _playwright_instance

    async with _browser_lock:
        if _browser_singleton is not None:
            try:
                if _browser_singleton.is_connected():
                    return _browser_singleton
            except Exception:
                pass
            _browser_singleton = None

        from playwright.async_api import async_playwright

        if _playwright_instance is None:
            _playwright_instance = await async_playwright().start()
        cdp_url = BROWSER_CDP_URL
        # Chromium DevTools rejects Host headers that are not an IP literal or
        # localhost (DNS-rebinding protection) -> "500 Host header is
        # specified and is not an IP address or localhost". A docker service
        # name like http://browser:9222 therefore fails. Resolve the hostname
        # to its (container) IP at connect time so the Host header is an IP;
        # resolving lazily (not at module import) keeps the resolution current
        # across sidecar restarts that change the container IP.
        import socket
        from urllib.parse import urlsplit

        parts = urlsplit(cdp_url)
        if parts.hostname and parts.hostname not in ("localhost", "127.0.0.1", "::1"):
            try:
                ip = socket.gethostbyname(parts.hostname)
                port = f":{parts.port}" if parts.port else ""
                cdp_url = f"{parts.scheme}://{ip}{port}"
            except socket.gaierror:
                pass  # already an IP literal or unresolvable: use as-is
        _browser_singleton = await _playwright_instance.chromium.connect_over_cdp(cdp_url)
        return _browser_singleton


async def _fetch_with_browser(url: str) -> tuple[Optional[str], Optional[str]]:
    """Tier 2: render url in the CDP sidecar browser and defeat JS challenges.

    Polls the rendered DOM until the challenge clears, then bounces any
    cookies back through tier 1 ("browser-bounce"); if still blocked, returns
    the rendered DOM itself ("browser-render"). All failures yield
    (None, None); this function never raises.
    """

    async def _attempt() -> tuple[Optional[str], Optional[str]]:
        browser = await _get_browser()
        # Fingerprint discipline: reuse the same UA/Accept headers as tier 0/1.
        ctx = await browser.new_context(
            user_agent=_HEADERS[0]["User-Agent"],
            locale="en-US",
            extra_http_headers={
                "Accept": _HEADERS[0]["Accept"],
                "Accept-Language": _HEADERS[0]["Accept-Language"],
            },
        )
        try:
            page = await ctx.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass  # networkidle is best-effort; keep polling regardless

            # Poll until the DOM looks big enough and carries no challenge title.
            html: Optional[str] = None
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _BROWSER_POLL_BUDGET
            while loop.time() < deadline:
                html = await page.content()
                title = await page.title()
                title_lower = title.lower()
                clear = len(html) >= _BROWSER_MIN_HTML_CHARS and not any(
                    marker in title_lower for marker in _BROWSER_CHALLENGE_TITLES
                )
                if clear:
                    break
                await asyncio.sleep(_BROWSER_POLL_INTERVAL)

            # Bounce solved cookies back through tier 1 — cheapest success path.
            cookie_tuples = _normalize_cookies(await ctx.cookies())
            jar = _cookies_for_request(cookie_tuples, url)
            text, status = await _stealth_fetch(url, _HEADERS[0], cookies=jar)
            if status == 200 and text and not _is_empty_extraction(_clean_html(text, url)):
                return text, "browser-bounce"

            return html, "browser-render"
        finally:
            await ctx.close()

    try:
        return await asyncio.wait_for(_attempt(), ESCALATION_TIMEOUT)
    except asyncio.TimeoutError:
        return None, None
    except Exception as e:
        print(f"Browser fetch error for {url}: {e}", flush=True)
        return None, None


async def _fetch_page(url: str, max_length: int = PAGE_MAX_CONTENT_LENGTH, include_links: bool = False) -> dict:
    """Fetch a URL and return its cleaned text content."""
    if _is_pdf_url(url):
        content = await _extract_pdf_with_tika(url)
        truncated_content = content[:max_length]
        return {
            "url": url,
            "title": os.path.basename(urllib.parse.urlparse(url).path) or url,
            "content": truncated_content,
            "content_length": len(truncated_content),
            "truncated": len(content) > max_length,
        }

    if USE_REDDIT_BACKEND:
        parsed = parse_reddit_url(url)
        if parsed is not None:
            return await reddit_fetch(parsed, url, max_length,
                                      REDDIT_MIN_INTERVAL, REDDIT_CACHE_TTL)

    html = None
    last_error = None
    fatal_error = False
    blocked = False
    via = None

    async with httpx.AsyncClient(follow_redirects=True) as client:
        for headers_idx, timeout_extra in _FETCH_ATTEMPTS:
            try:
                resp = await client.get(
                    url,
                    headers=_HEADERS[headers_idx],
                    timeout=PAGE_FETCH_TIMEOUT + timeout_extra,
                )
                resp.raise_for_status()
                html = resp.text
                break
            except httpx.TimeoutException:
                continue
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (403, 429, 401):
                    if e.response.status_code in (403, 429):
                        blocked = True
                    continue
                last_error = f"HTTP {e.response.status_code}"
                fatal_error = True
                break
            except httpx.RequestError:
                continue

    if html is None:
        # Tiered escalation: only when every httpx attempt was actively
        # blocked (403/429) — timeouts and other request errors don't escalate.
        if blocked and not fatal_error:
            # Tier 1: curl_cffi browser-impersonated retry.
            text, status = await _stealth_fetch(url, _HEADERS[0])
            if status == 200 and text:
                html = text
                via = "stealth-http"
            elif USE_BROWSER_TIER:
                # Tier 2: render in the CDP sidecar browser.
                html2, via2 = await _fetch_with_browser(url)
                if html2 is not None:
                    html = html2
                    via = via2
        if html is None:
            return {
                "url": url,
                "title": "",
                "content": f"Failed to fetch page: {last_error or 'all attempts failed'}.",
                "content_length": 0,
                "truncated": False,
            }

    if BeautifulSoup is None:
        return {
            "url": url,
            "title": "",
            "content": "Failed to parse HTML: BeautifulSoup not available",
            "content_length": 0,
            "truncated": False,
        }

    # Extract title
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True)
    elif soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)

    # Extract content
    cleaned = _clean_html(html, url)

    # Empty extraction on script-heavy HTML smells like a JS challenge —
    # escalate to tier 2 unless this result already came from the browser.
    if (
        via not in ("browser-bounce", "browser-render")
        and _should_escalate_tier2(200, html, cleaned)
        and USE_BROWSER_TIER
    ):
        html2, via2 = await _fetch_with_browser(url)
        if html2 is not None:
            html = html2
            via = via2
            soup = BeautifulSoup(html, "html.parser")
            title = ""
            if soup.title:
                title = soup.title.get_text(strip=True)
            elif soup.find("h1"):
                title = soup.find("h1").get_text(strip=True)
            cleaned = _clean_html(html, url)

    truncated = len(cleaned) > max_length
    if truncated:
        cleaned = cleaned[:max_length] + "... [content truncated]"

    result = {
        "url": url,
        "title": title,
        "content": cleaned,
        "content_length": len(cleaned),
        "truncated": truncated,
    }

    if via is not None:
        result["via"] = via

    if include_links:
        result["links"] = _extract_links(html, url)

    return result


# ── MCP Server Definition ────────────────────────────────────────────────────

mcp = FastMCP(
    "websearch",
    json_response=True,  # Use JSON responses instead of SSE where possible
    transport_security=_transport_security,  # host validation (ALLOWED_HOSTS)
)


@mcp.tool()
async def web_search(
    query: str,
    num_results: int = 10,
    engines: Optional[list[str]] = None,
) -> str:
    """Search the web using SearXNG. Returns a list of results
    with title, URL, and snippet for each. Use this tool to find information on
    the internet. Prefer specific queries for better results.

    Args:
        query: The search query — be specific for better results.
        num_results: Number of results to return (1-20, default: 10).
        engines: Which engines to use. Default: all available. Currently only
                 "searxng" is supported. Options: ["searxng"].
    """
    import asyncio
    import json

    num_results = max(1, min(20, num_results))

    if engines is None:
        engines = []
        if USE_SEARXNG_SEARCH and SEARXNG_URL:
            engines.append("searxng")

    if not engines:
        return json.dumps({
            "error": "No search engines available. Configure SEARXNG_URL."
        })

    # Run searches concurrently
    tasks = []
    for engine in engines:
        if engine == "searxng" and USE_SEARXNG_SEARCH:
            tasks.append(_search_searxng(query, num_results))

    results_raw = await asyncio.gather(*tasks, return_exceptions=True)

    all_results = []
    for i, res in enumerate(results_raw):
        if isinstance(res, Exception):
            engine_name = engines[i] if i < len(engines) else "unknown"
            print(f"Engine {engine_name} error: {res}", flush=True)
            continue
        if res:
            all_results.extend(res)

    # Deduplicate by URL
    seen_urls = set()
    unique = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique.append(r)

    limited = unique[:num_results]

    output = {
        "query": query,
        "total_results": len(limited),
        "results": limited,
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


@mcp.tool()
async def fetch_page(
    url: str,
    max_length: int = 50000,
    include_links: bool = False,
) -> str:
    """Fetch a web page and extract its main text content, removing navigation,
    ads, and boilerplate. Use this to read the full content of a URL found via
    web_search or any web page. Returns the page title and cleaned text.

    Args:
        url: Full URL to fetch (must start with http:// or https://).
        max_length: Maximum characters to extract (default: 50000, max: 100000).
        include_links: If true, extract anchor links from the page HTML and return
            them as a separate "links" array of {text, href} objects with absolute URLs.
            Default: false (backward compatible — no links field in output).
    """
    import json

    if not url or not url.startswith(("http://", "https://")):
        return json.dumps({"error": "url is required and must start with http:// or https://."})

    max_length = max(1, min(100000, max_length))

    try:
        result = await _fetch_page(url, max_length, include_links=include_links)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)


@mcp.tool()
async def search_reddit(
    query: str,
    subreddit: Optional[str] = None,
    sort: str = "relevance",
    time_range: str = "all",
    num_results: int = 10,
) -> str:
    """Search Reddit posts natively via Reddit's public feeds. Works from
    server IPs where reddit.com pages are blocked. Rate-limited to ~1 request
    per minute for ALL reddit tools combined; on a rate-limit error, wait
    retry_after seconds and call again.

    Args:
        query: Search query — be specific.
        subreddit: Restrict search to one subreddit (optional).
        sort: result order — relevance | new | top | comments.
        time_range: hour | day | week | month | year | all.
        num_results: Max results (1-25, default 10).
    """
    import json

    if sort not in SEARCH_SORTS or time_range not in TIME_RANGES:
        return json.dumps({"error": f"sort must be one of {sorted(SEARCH_SORTS)}; "
                                    f"time_range one of {sorted(TIME_RANGES)}"})
    num_results = max(1, min(25, num_results))
    entries, err = await search_feeds(query, subreddit, sort, time_range, num_results,
                                      REDDIT_MIN_INTERVAL, REDDIT_CACHE_TTL)
    if err is not None:
        return json.dumps(err)
    results = [{
        "title": e["title"], "url": e["url"], "author": e["author"],
        "created": e["created"], "snippet": e["body"][:300],
    } for e in entries[:num_results]]
    return json.dumps({"query": query, "subreddit": subreddit,
                       "total_results": len(results), "results": results},
                      ensure_ascii=False, indent=2)


@mcp.tool()
async def health_check() -> str:
    """Check if the web search MCP server is healthy and which backends are available."""
    import json

    available = []
    if USE_SEARXNG_SEARCH and SEARXNG_URL:
        available.append("searxng")

    # Quick SearXNG ping
    searxng_ok = False
    if USE_SEARXNG_SEARCH and SEARXNG_URL:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{SEARXNG_URL}/healthz")
                searxng_ok = resp.status_code == 200
        except Exception:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(
                        f"{SEARXNG_URL}/search",
                        params={"q": "test", "format": "json"},
                        headers={"Accept": "application/json"},
                    )
                    searxng_ok = resp.status_code == 200
            except Exception:
                pass

    reddit = reddit_gate_status() if USE_REDDIT_BACKEND else {"enabled": False}

    status = {
        "status": "ok",
        "available_engines": available,
        "searxng_reachable": searxng_ok,
        "tika_enabled": USE_TIKA,
        "reddit_rss": reddit,
        "config": {
            "SEARXNG_URL": SEARXNG_URL,
            "USE_SEARXNG_SEARCH": USE_SEARXNG_SEARCH,
            "PAGE_FETCH_TIMEOUT": PAGE_FETCH_TIMEOUT,
            "PAGE_MAX_CONTENT_LENGTH": PAGE_MAX_CONTENT_LENGTH,
            "USE_BROWSER_TIER": USE_BROWSER_TIER,
            "BROWSER_CDP_URL": BROWSER_CDP_URL,
            "ESCALATION_TIMEOUT": ESCALATION_TIMEOUT,
        },
    }
    return json.dumps(status, indent=2)


# ── ASGI App (Streamable HTTP) ───────────────────────────────────────────────
# streamable_http_app() returns a complete Starlette app with routes at /mcp
# No additional Mount or wrapping needed — use it directly as the ASGI app.
app = mcp.streamable_http_app()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT)
