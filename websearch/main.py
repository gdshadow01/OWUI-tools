from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import asyncio
import os
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import trafilatura
from readability import Document
import urllib.parse

app = FastAPI(
    title="Web Search Tool",
    description="A comprehensive web search API with detailed tool descriptions and error handling for local models.",
    version="1.0.0"
)

# Configuration
GOOGLE_PSE_API_KEY = os.getenv("GOOGLE_PSE_API_KEY", "")
GOOGLE_PSE_CX = os.getenv("GOOGLE_PSE_CX", "")
TIKA_SERVER_URL = os.getenv("TIKA_SERVER_URL", "http://tika:9998")  # Default Tika server URL
SEARXNG_URL = os.getenv("SEARXNG_URL", "")

# Enablement flags
USE_GOOGLE_SEARCH = os.getenv("USE_GOOGLE_SEARCH", "yes").lower() == "yes"
USE_DUCKDUCKGO_SEARCH = os.getenv("USE_DUCKDUCKGO_SEARCH", "yes").lower() == "yes"
USE_SEARXNG_SEARCH = os.getenv("USE_SEARXNG_SEARCH", "yes").lower() == "yes"
SEARXNG_TIMEOUT = int(os.getenv("SEARXNG_TIMEOUT", "10"))

# Page fetch configuration
PAGE_FETCH_TIMEOUT = int(os.getenv("PAGE_FETCH_TIMEOUT", "10"))  # seconds
PAGE_MAX_CONTENT_LENGTH = int(os.getenv("PAGE_MAX_CONTENT_LENGTH", "50000"))  # characters

class SearchRequest(BaseModel):
    """
    Request model for web search functionality

    Attributes:
        query (str): The search query - should be clear and specific (REQUIRED)
        num_results (int): Number of search results to return (1-20, default: 10)
        engines (List[str]): Which search engines to use - options: "google", "duckduckgo", "searxng" (default: uses all available)
        concurrent_requests (int): Number of concurrent requests (default: 3)
    """
    query: str
    num_results: int = 10
    engines: Optional[List[str]] = None  # Will be set to enabled engines if None
    concurrent_requests: int = 3

    class Config:
        json_schema_extra = {
            "example": {
                "query": "latest developments in renewable energy",
                "num_results": 5,
                "engines": ["google", "duckduckgo", "searxng"],
                "concurrent_requests": 3
            }
        }

class SearchResult(BaseModel):
    """
    Individual search result model

    Attributes:
        title (str): Title of the search result
        url (str): URL of the search result
        snippet (str): Brief description of the result
        engine (str): Which search engine provided this result
    """
    title: str
    url: str
    snippet: str
    engine: str

class SearchResponse(BaseModel):
    """
    Response model for web search results

    Attributes:
        results (List[SearchResult]): List of search results
        total_results (int): Total number of results returned
    """
    results: List[SearchResult]
    total_results: int

class PageFetchRequest(BaseModel):
    """
    Request model for page content fetching

    Attributes:
        url (str): Full URL of the page to fetch content from (REQUIRED, must start with http:// or https://)
        max_length (int): Maximum character length of content to extract (default: 50000)
    """
    url: str
    max_length: int = PAGE_MAX_CONTENT_LENGTH

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "max_length": 10000
            }
        }

class PageFetchResponse(BaseModel):
    """
    Response model for page content fetching

    Attributes:
        url (str): The URL that was fetched
        title (str): Title of the page
        content (str): Extracted main content of the page
        content_length (int): Number of characters in the content
        truncated (bool): Whether the content was shortened due to length limits
    """
    url: str
    title: str
    content: str
    content_length: int
    truncated: bool

class RAGRequest(BaseModel):
    """
    Request model for retrieval-augmented generation functionality

    Attributes:
        query (str): Query to process
        urls (List[str]): List of URLs to retrieve information from
    """
    query: str
    urls: List[str]

class RAGResponse(BaseModel):
    """
    Response model for retrieval-augmented generation

    Attributes:
        text (str): Generated text response
        url (Optional[str]): URL associated with the response (optional)
    """
    text: str
    url: Optional[str] = None

def get_enabled_engines() -> List[str]:
    """Return list of enabled search engines"""
    engines = []
    if USE_GOOGLE_SEARCH and GOOGLE_PSE_API_KEY and GOOGLE_PSE_CX:
        engines.append("google")
    if USE_DUCKDUCKGO_SEARCH:
        engines.append("duckduckgo")
    if USE_SEARXNG_SEARCH and SEARXNG_URL:
        engines.append("searxng")
    return engines

async def extract_pdf_content_with_tika(pdf_url: str) -> str:
    """Extract text content from a PDF using Apache Tika server"""
    # Download the PDF content first, with redirects enabled
    async with httpx.AsyncClient(timeout=PAGE_FETCH_TIMEOUT, follow_redirects=True) as download_client:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = await download_client.get(pdf_url, headers=headers)
        response.raise_for_status()
        pdf_content = response.content

    # Use a separate client for Tika requests to avoid any potential conflicts
    async with httpx.AsyncClient(timeout=PAGE_FETCH_TIMEOUT) as tika_client:
        # The correct approach: PUT request to /tika endpoint with proper content-type
        try:
            tika_response = await tika_client.put(
                f"{TIKA_SERVER_URL}/tika",
                headers={"Accept": "text/plain", "Content-Type": "application/pdf"},
                content=pdf_content
            )
            tika_response.raise_for_status()
            return tika_response.text
        except httpx.HTTPStatusError as e:
            # Log the specific error for debugging
            print(f"Tika extraction failed for {pdf_url}, status: {e.response.status_code}, response: {e.response.text}")
            raise HTTPException(status_code=500, detail=f"Tika extraction failed for PDF: {pdf_url}, status: {e.response.status_code}")
        except Exception as e:
            print(f"Unexpected error during Tika extraction for {pdf_url}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error during Tika extraction for PDF: {pdf_url}: {str(e)}")


def is_pdf_url(url: str) -> bool:
    """Check if the URL points to a PDF file"""
    return url.lower().endswith('.pdf') or 'pdf' in url.lower()


def is_pdf_content_type(content_type: str) -> bool:
    """Check if the content type is PDF"""
    if content_type:
        return 'pdf' in content_type.lower()
    return False


async def fetch_html_content(url: str) -> str:
    """Fetch content from a URL, handling both HTML and PDF content with enhanced error handling and retry logic"""

    # Check if it's a PDF based on URL extension first (without making a request)
    if is_pdf_url(url):
        return await extract_pdf_content_with_tika(url)

    # Define multiple user agents and headers to try
    headers_options = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        },
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    ]

    # Try different configurations
    configs = [
        {"timeout": PAGE_FETCH_TIMEOUT, "follow_redirects": True},
        {"timeout": PAGE_FETCH_TIMEOUT, "follow_redirects": False},
        {"timeout": PAGE_FETCH_TIMEOUT + 5, "follow_redirects": True},  # Longer timeout
        {"timeout": PAGE_FETCH_TIMEOUT + 10, "follow_redirects": True},  # Even longer timeout
    ]

    # Try with different headers and client configurations
    for headers in headers_options:
        for config in configs:
            try:
                # Create a new client for each attempt with specific configuration
                async with httpx.AsyncClient(
                    timeout=config["timeout"],
                    follow_redirects=config["follow_redirects"]
                ) as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()

                    # Check the content type from the response
                    content_type = response.headers.get("content-type", "")
                    if is_pdf_content_type(content_type):
                        # If we got a PDF, process with Tika using the response content
                        # Use the same robust Tika extraction method as the dedicated function
                        pdf_content = response.content

                        # Use the dedicated function for Tika extraction
                        return await extract_pdf_content_with_tika(url)
                    else:
                        # It's HTML content, return as text
                        return response.text

            except httpx.TimeoutException:
                print(f"Timeout with headers {headers['User-Agent'][:50]}... and config {config}")
                continue  # Try next configuration
            except httpx.HTTPStatusError as e:
                print(f"HTTP {e.response.status_code} error with headers {headers['User-Agent'][:50]}... and config {config}")
                # Some sites return special codes to indicate they're blocking requests
                if e.response.status_code in [403, 429, 401]:  # Forbidden, Too Many Requests, Unauthorized
                    continue  # Try next configuration
                else:
                    # Re-raise if it's a different type of error (like 404, 500)
                    raise HTTPException(status_code=e.response.status_code, detail=f"Error fetching page {url}: {str(e)}")
            except httpx.RequestError as e:
                print(f"Request error '{str(e)}' with headers {headers['User-Agent'][:50]}... and config {config}")
                # Skip to next configuration if it's a connection error
                continue
            except Exception as e:
                print(f"General error '{str(e)}' with headers {headers['User-Agent'][:50]}... and config {config}")
                continue

    # If all attempts failed, raise a comprehensive error
    raise HTTPException(status_code=400, detail=f"All attempts to fetch page {url} failed. The site may be blocking automated requests, require specific authentication, or the URL may not exist.")

def clean_html_content(html: str, url: str) -> str:
    """
    Clean HTML content using trafilatura with readability fallback

    Args:
        html (str): Raw HTML content to clean
        url (str): URL of the page being processed (used for error reporting)

    Returns:
        str: Clean text content extracted from the HTML
    """
    try:
        # Try trafilatura first (more sophisticated extraction)
        cleaned_text = trafilatura.extract(html)

        # If trafilatura fails or returns empty/very short text, use readability
        if not cleaned_text or len(cleaned_text.strip()) < 100:
            doc = Document(html)
            # Get the summary from readability
            readability_result = doc.summary()

            # Extract text content from the summary
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(readability_result, 'html.parser')
            cleaned_text = soup.get_text(separator=' ')

        # Ensure we have text, even if minimal
        if not cleaned_text or len(cleaned_text.strip()) < 50:
            # Try a simple BeautifulSoup fallback
            soup = BeautifulSoup(html, 'html.parser')
            cleaned_text = soup.get_text(separator=' ', strip=True)

            # If still nothing substantial, return a descriptive message
            if not cleaned_text or len(cleaned_text.strip()) < 50:
                return f"Page content could not be extracted from URL: {url}. The page may contain mostly images, videos, or JavaScript that doesn't render as readable text."

        return cleaned_text.strip()

    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return f"Error extracting content from {url}. The page may contain complex formatting that couldn't be parsed. Error: {str(e)}"

async def process_single_url(url: str) -> str:
    """Process a single URL and return the full cleaned content"""
    html = await fetch_html_content(url)
    cleaned_text = clean_html_content(html, url)
    return cleaned_text

async def process_multiple_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """Process multiple URLs and return their content"""
    results = []
    
    for url in urls:
        try:
            html = await fetch_html_content(url)
            cleaned_text = clean_html_content(html, url)
            
            results.append({
                "url": url,
                "text": cleaned_text
            })
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            continue
    
    return results

async def search_google(query: str, num_results: int) -> List[SearchResult]:
    """Search using Google Programmable Search Engine"""
    if not GOOGLE_PSE_API_KEY or not GOOGLE_PSE_CX:
        return []
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_PSE_API_KEY,
        "cx": GOOGLE_PSE_CX,
        "q": query,
        "num": min(num_results, 10)  # Google PSE limit
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    engine="google"
                ))
            return results
        except Exception as e:
            print(f"Google search error: {e}")
            return []

async def search_duckduckgo(query: str, num_results: int) -> List[SearchResult]:
    """Search using DuckDuckGo HTML scraping"""
    url = "https://html.duckduckgo.com/html"

    # Prepare the headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Prepare form data for POST request
    data = {
        "q": query,
        "b": "",  # Leave blank to let DDG auto-fill
        "kl": "", # Leave blank to use default location
    }

    async with httpx.AsyncClient() as client:
        try:
            # Use POST request to search
            response = await client.post(url, data=data, headers=headers, timeout=30.0)
            response.raise_for_status()

            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')

            results = []
            # Find result containers - DuckDuckGo uses .result class for each search result
            result_elements = soup.select('.result')

            for idx, element in enumerate(result_elements):
                if len(results) >= num_results:
                    break

                # Extract title and link
                title_elem = element.select_one('.result__title a')
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                link = title_elem.get('href', '')

                # Clean DuckDuckGo redirect URLs
                if link.startswith('//duckduckgo.com/l/?uddg='):
                    link = urllib.parse.unquote(link.split('uddg=')[1].split('&')[0])
                elif link.startswith('/l/?'):
                    # Handle other redirect formats
                    href_parts = urllib.parse.urlparse(link)
                    query_params = urllib.parse.parse_qs(href_parts.query)
                    if 'uddg' in query_params:
                        link = urllib.parse.unquote(query_params['uddg'][0])

                # Skip if the link is an ad or invalid
                if 'y.js' in link or not link.startswith(('http://', 'https://')):
                    continue

                # Extract snippet
                snippet_elem = element.select_one('.result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                if title and link:
                    results.append(SearchResult(
                        title=title,
                        url=link,
                        snippet=snippet,
                        engine="duckduckgo"
                    ))

            return results
        except httpx.TimeoutException:
            print(f"DuckDuckGo search timeout: {url}")
            return []
        except httpx.HTTPError as e:
            print(f"DuckDuckGo HTTP error: {e}")
            return []
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []

async def search_searxng(query: str, num_results: int) -> List[SearchResult]:
    """Search using SearXNG JSON API"""
    if not SEARXNG_URL:
        return []
    
    url = f"{SEARXNG_URL}/search"
    params = {
        "q": query,
        "format": "json",
        "language": "en",
        "safesearch": "0",
        "categories": "general"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    async with httpx.AsyncClient(timeout=SEARXNG_TIMEOUT) as client:
        try:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", []):
                title = item.get("title", "")
                url = item.get("url", "")
                content = item.get("content", "")
                
                if title and url:
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=content,
                        engine="searxng"
                    ))
            
            return results[:num_results]
        except Exception as e:
            print(f"SearXNG search error: {e}")
            return []


async def fetch_page_content(url: str, max_length: int = PAGE_MAX_CONTENT_LENGTH) -> PageFetchResponse:
    """Fetch and extract text content from a web page"""
    try:
        html = await fetch_html_content(url)
        cleaned_text = clean_html_content(html, url)
        
        # Get title using BeautifulSoup for consistency
        soup = BeautifulSoup(html, 'html.parser')
        title = ""
        if soup.title:
            title = soup.title.get_text()
        elif soup.find('h1'):
            title = soup.find('h1').get_text()
        
        # Truncate if necessary
        truncated = len(cleaned_text) > max_length
        if truncated:
            cleaned_text = cleaned_text[:max_length] + "... [content truncated]"
        
        return PageFetchResponse(
            url=url,
            title=title,
            content=cleaned_text,
            content_length=len(cleaned_text),
            truncated=truncated
        )
    except Exception as e:
        raise e

@app.post("/search", response_model=SearchResponse,
          summary="Web Search Tool",
          description="""Performs web search across multiple search engines to find relevant results for your query.

          ## Purpose
          This tool searches the web for information based on your query using multiple search engines.

          ## Important Parameters
          - `query`: The search query - should be clear and specific (REQUIRED)
          - `num_results`: Number of search results to return, between 1-20 (default: 10)
          - `engines`: Which search engines to use - options: "google", "duckduckgo", "searxng" (default: uses all available)

          ## Usage Examples
          Good: `{"query": "machine learning applications healthcare", "num_results": 5}`
          Better for speed: `{"query": "renewable energy statistics 2024", "num_results": 3, "engines": ["google"]}`
          Private search: `{"query": "privacy focused search engine", "num_results": 5, "engines": ["searxng"]}`

          ## Common Errors
          - If you get an error about "no valid search engines", it means the server isn't configured with API keys for Google or has issues with DuckDuckGo or SearXNG
          - If query is too vague or empty, you'll get poor results""")
async def web_search(request: SearchRequest):
    """
    Perform web search across enabled engines.

    Args:
        request (SearchRequest): Contains query, number of results, and search engines to use

    Returns:
        SearchResponse: Contains list of search results with title, URL, snippet and engine
    """
    # Validate the query
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Query parameter is required and cannot be empty. Please provide a clear, specific search query."
        )

    # Validate number of results
    if request.num_results < 1 or request.num_results > 20:
        raise HTTPException(
            status_code=400,
            detail="num_results must be between 1 and 20. Use lower values for faster responses."
        )

    # If no engines specified, use enabled engines
    if request.engines is None:
        request.engines = get_enabled_engines()

        # If still no engines available after getting enabled ones
        if not request.engines:
            available_engines = []
            if GOOGLE_PSE_API_KEY and GOOGLE_PSE_CX:
                available_engines.append("google")
            if USE_DUCKDUCKGO_SEARCH:
                available_engines.append("duckduckgo")
            if USE_SEARXNG_SEARCH and SEARXNG_URL:
                available_engines.append("searxng")

            raise HTTPException(
                status_code=400,
                detail=f"""No search engines are available.
                Available: {available_engines}.
                Configured: Google API key={'SET' if GOOGLE_PSE_API_KEY else 'NOT SET'},
                Google CX={'SET' if GOOGLE_PSE_CX else 'NOT SET'},
                DuckDuckGo enabled={USE_DUCKDUCKGO_SEARCH},
                SearXNG enabled={USE_SEARXNG_SEARCH}, URL={'SET' if SEARXNG_URL else 'NOT SET'}.
                Please configure environment variables."""
            )

    # Filter to only enabled engines
    enabled_engines = get_enabled_engines()
    engines_to_use = [e for e in request.engines if e in enabled_engines]

    if not engines_to_use:
        raise HTTPException(
            status_code=400,
            detail=f"""No valid search engines specified.
            Requested engines: {request.engines},
            Available engines: {enabled_engines}.
            Valid options are: google, duckduckgo, searxng"""
        )

    search_tasks = []

    # Create tasks for each search engine
    if "google" in engines_to_use:
        search_tasks.append(search_google(request.query, request.num_results))

    if "duckduckgo" in engines_to_use:
        search_tasks.append(search_duckduckgo(request.query, request.num_results))

    if "searxng" in engines_to_use:
        search_tasks.append(search_searxng(request.query, request.num_results))

    # Execute searches concurrently
    try:
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # Process results
    all_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            engine_name = engines_to_use[i] if i < len(engines_to_use) else "unknown"
            print(f"Search engine {engine_name} error: {result}")
            continue
        if result:
            all_results.extend(result)

    # Limit to requested number of results
    limited_results = all_results[:request.num_results]

    return SearchResponse(
        results=limited_results,
        total_results=len(limited_results)
    )

@app.post("/fetch-page", response_model=PageFetchResponse,
          summary="Page Content Fetch Tool",
          description="""Gets the readable text content from a specific webpage URL.

          ## Purpose
          This tool loads a specific webpage and extracts the main content, removing navigation, ads, and other clutter.

          ## Important Parameters
          - `url`: The full web address of the page to extract content from (REQUIRED, must start with http:// or https://)
          - `max_length`: Maximum number of characters to extract (default: 50000)

          ## Usage Examples
          Basic: `{"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"}`
          Limited: `{"url": "https://example.com/article", "max_length": 10000}`

          ## Common Errors
          - "Timeout fetching page": The website took too long to respond (website might be slow/down)
          - "Error fetching page": Problems connecting to the URL (might not exist or blocked)
          - "Content could not be extracted": The page is mostly images, videos, or JavaScript that doesn't render as readable text""")
async def fetch_page(request: PageFetchRequest):
    """
    Fetch and extract content from a web page.

    Args:
        request (PageFetchRequest): Contains URL and max_length for content extraction

    Returns:
        PageFetchResponse: Contains URL, title, content, length and truncation info
    """
    # Validate URL format
    if not request.url or not request.url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="URL parameter is required and must start with 'http://' or 'https://' to be a valid web address."
        )

    # Validate max_length range
    if request.max_length < 1 or request.max_length > 100000:
        raise HTTPException(
            status_code=400,
            detail="max_length must be between 1 and 100,000 characters. Use lower values for shorter content."
        )

    try:
        return await fetch_page_content(request.url, request.max_length)
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch page content: {str(e)}. "
                   f"Make sure the URL is accessible and has readable content."
        )

@app.get("/health",
         summary="Service Status Tool",
         description="""Checks if the search service is working properly and sees what features are available.

         ## Purpose
         This tool tells you if the search service is running, which search engines are enabled,
         and confirms the configuration. Use this if your searches aren't working to understand what's available.

         ## What You Get Back
         - status: "ok" if running properly
         - enabled_engines: Which search engines are available (e.g., ["google", "duckduckgo", "searxng"])
         - Configuration details showing timeout values and other settings

         ## Common Errors
         - If you get an error here, the entire service might be down or misconfigured""")
async def health_check():
    """Check the health status of the service and report what features are available."""
    enabled_engines = get_enabled_engines()
    available_engines = []
    if GOOGLE_PSE_API_KEY and GOOGLE_PSE_CX:
        available_engines.append("google")
    if USE_DUCKDUCKGO_SEARCH:
        available_engines.append("duckduckgo")
    if USE_SEARXNG_SEARCH and SEARXNG_URL:
        available_engines.append("searxng")

    return {
        "status": "ok",
        "message": "Service is operational",
        "enabled_engines": enabled_engines,
        "available_engines": available_engines,
        "config": {
            "USE_GOOGLE_SEARCH": USE_GOOGLE_SEARCH,
            "USE_DUCKDUCKGO_SEARCH": USE_DUCKDUCKGO_SEARCH,
            "USE_SEARXNG_SEARCH": USE_SEARXNG_SEARCH,
            "SEARXNG_URL_SET": bool(SEARXNG_URL),
            "GOOGLE_PSE_API_KEY_SET": bool(GOOGLE_PSE_API_KEY),
            "GOOGLE_PSE_CX_SET": bool(GOOGLE_PSE_CX),
            "PAGE_FETCH_TIMEOUT": PAGE_FETCH_TIMEOUT,
            "PAGE_MAX_CONTENT_LENGTH": PAGE_MAX_CONTENT_LENGTH
        },
        "instructions": {
            "web_search": {
                "purpose": "Searches the web for information based on your query using multiple search engines",
                "parameters": {
                    "query": "The search query - should be clear and specific (REQUIRED)",
                    "num_results": "Number of search results to return (1-20, default: 10)",
                    "engines": "Which search engines to use - options: 'google', 'duckduckgo', 'searxng' (default: uses all available)"
                },
                "usage_examples": [
                    {"query": "machine learning applications healthcare", "num_results": 5},
                    {"query": "renewable energy statistics 2024", "num_results": 3, "engines": ["duckduckgo"]},
                    {"query": "privacy focused search", "num_results": 5, "engines": ["searxng"]}
                ]
            },
            "fetch_page": {
                "purpose": "Loads a specific webpage and extracts the main content, removing navigation, ads, and other clutter",
                "parameters": {
                    "url": "The full web address of the page to extract content from (REQUIRED, must start with http:// or https://)",
                    "max_length": "Maximum number of characters to extract (default: 50000)"
                },
                "usage_examples": [
                    {"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"},
                    {"url": "https://example.com/article", "max_length": 10000}
                ]
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
