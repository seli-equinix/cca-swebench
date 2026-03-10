# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""CCA extension providing web search and URL fetching tools.

Stateless utility tools — no user session or state dependency.
Follows CCA's ToolUseExtension pattern (same as UserToolsExtension).

Tools provided:
- web_search(query, ...) -> Search via SearXNG
- fetch_url_content(url, ...) -> Fetch and extract text from URLs
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import socket
from typing import Any
from urllib.parse import urlparse

import httpx

from ..core.analect import AnalectRunContext
from ..core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ..orchestrator.extensions.tool_use import ToolUseExtension

logger = logging.getLogger(__name__)

# Stopwords for query simplification
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "that", "this",
    "for", "to", "of", "in", "on", "with", "and", "or", "but",
    "use", "using", "used", "how", "what", "which", "can", "do",
    "does", "it", "i", "my", "we", "our", "their", "there",
})


def _simplify_query(query: str) -> str:
    """Strip stopwords, keep max 5 meaningful words."""
    words = [w for w in query.split() if w.lower() not in _STOPWORDS]
    return " ".join(words[:5])


# Category auto-detection — determines which SearXNG engine pool to use.
# The LLM must NOT pick categories: "it" returns Firefox for Python queries,
# "science" misses Stack Overflow, etc. We detect intent from the query.
_NEWS_PHRASES = frozenset({
    "latest news", "breaking news", "news today", "this week news",
    "current events", "what happened", "just announced", "just released",
    "just launched",
})
_SCIENCE_PHRASES = frozenset({
    "arxiv", "research paper", "scientific study", "published study",
    "peer reviewed", "journal article", "preprint",
})


def _auto_category(query: str) -> str | None:
    """Auto-detect the best SearXNG category from the query text.

    Returns a category string to pass to SearXNG, or None for the default
    general engines (Google/DuckDuckGo/Bing — best for most queries).
    """
    q = query.lower()
    if any(phrase in q for phrase in _NEWS_PHRASES):
        return "news"
    if any(phrase in q for phrase in _SCIENCE_PHRASES):
        return "science"
    # Default: general covers programming, tech, docs, and everything else
    return None


class UtilityToolsExtension(ToolUseExtension):
    """CCA extension providing web search and URL fetching tools."""

    name: str = "UtilityToolsExtension"
    enable_tool_use: bool = True
    included_in_system_prompt: bool = False

    _http_client: Any

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_http_client", None)
        object.__setattr__(self, "_search_blocked", False)

    def block_searches(self) -> None:
        """Block further web_search / fetch_url_content calls.

        Called by the orchestrator's search loop guard after the search limit
        is reached. Subsequent tool calls return a hard-stop result instead of
        executing, giving the model no new results to "justify" more searching.
        """
        object.__setattr__(self, "_search_blocked", True)
        logger.info("UtilityToolsExtension: searches blocked (loop guard)")

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy shared httpx client — reused across all tool calls."""
        if self._http_client is None:
            object.__setattr__(
                self,
                "_http_client",
                httpx.AsyncClient(follow_redirects=True),
            )
        return self._http_client

    @property
    async def tools(self) -> list[ant.ToolLike]:
        return [
            ant.Tool(
                name="web_search",
                description=(
                    "Search the internet via SearXNG for current information. "
                    "Use for: latest versions, release notes, documentation, "
                    "news, API references — anything time-sensitive.\n\n"
                    "Form a SHORT, SPECIFIC query (3-6 keywords). Bad: "
                    "'what is the latest version of vLLM llm inference engine' "
                    "— Good: 'vLLM 0.8 release notes' or 'vLLM latest version'.\n\n"
                    "Query syntax: 'site:github.com <query>', "
                    "'\"exact phrase\"', '-exclude_word'\n\n"
                    "Tips:\n"
                    "- Use time_range=\"week\" for very recent results\n"
                    "- Use engines=\"github\" to search only GitHub\n"
                    "- Stop after 2-3 searches — synthesize what you have\n"
                    "- Categories are selected automatically based on your query — "
                    "do NOT pass a categories parameter"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Short, specific search query — 3 to 6 keywords. "
                                "Do NOT write a full sentence or question."
                            ),
                        },
                        "n_results": {
                            "type": "integer",
                            "description": (
                                "Number of results (1-10, default 5)"
                            ),
                        },
                        "time_range": {
                            "type": "string",
                            "description": (
                                "Filter by recency: day, week, month, year"
                            ),
                        },
                        "engines": {
                            "type": "string",
                            "description": (
                                "Comma-separated engines: google, bing, "
                                "duckduckgo, brave, wikipedia, github"
                            ),
                        },
                        "language": {
                            "type": "string",
                            "description": (
                                "ISO 639-1 language code (default: en)"
                            ),
                        },
                    },
                    "required": ["query"],
                },
            ),
            ant.Tool(
                name="fetch_url_content",
                description=(
                    "Fetch a URL and extract readable text content. "
                    "Use after web_search to read full pages from the best "
                    "results. Strips HTML navigation, scripts, and styles "
                    "to return clean text.\n\n"
                    "Supports http/https URLs only. Internal/private IPs "
                    "are blocked for security. Content is truncated at 500KB."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch (http/https only)",
                        },
                        "extract_text": {
                            "type": "boolean",
                            "description": (
                                "Extract readable text from HTML "
                                "(default: true)"
                            ),
                        },
                        "timeout": {
                            "type": "integer",
                            "description": (
                                "Max seconds to wait (default 30, max 60)"
                            ),
                        },
                    },
                    "required": ["url"],
                },
            ),
        ]

    async def on_tool_use(
        self,
        tool_use: ant.MessageContentToolUse,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        """Dispatch tool calls to the appropriate handler."""
        name = tool_use.name
        inp = tool_use.input or {}

        # Hard block: loop guard fired — too many web_search calls.
        # Block web_search but ALLOW fetch_url_content: that's the correct tool
        # when the model wants full page content (not more snippets).
        if self._search_blocked and name == "web_search":
            query = (inp.get("query") or "").strip()
            logger.info(
                "UtilityToolsExtension: blocked web_search(%r) — use fetch_url_content instead",
                query,
            )
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=json.dumps({
                    "blocked": True,
                    "message": (
                        "web_search is no longer available — you have already searched enough. "
                        "web_search only returns 500-character snippets and cannot give full "
                        "page content no matter how many times it is called. "
                        "If you need the full text of a page you found earlier (such as "
                        "https://docs.python.org/3.13/whatsnew/3.13.html), call "
                        "fetch_url_content with that exact URL. "
                        "Otherwise write your final answer now using what you have gathered."
                    ),
                }),
            )

        try:
            if name == "web_search":
                result = await self._handle_web_search(inp)
            elif name == "fetch_url_content":
                result = await self._handle_fetch_url(inp)
            else:
                result = f"Unknown tool: {name}"

            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=result if isinstance(result, str) else json.dumps(result),
            )

        except Exception as e:
            logger.error(f"Utility tool '{name}' failed: {e}")
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=f"Error: {e}",
                is_error=True,
            )

    # ==================== Tool Handlers ====================

    async def _handle_web_search(self, inp: dict[str, Any]) -> str:
        """Search via SearXNG with progressive fallback.

        Category is auto-detected from the query by _auto_category() — the LLM
        does not control it. Retry cascade on empty results:
          1. Auto-detected category + original params
          2. Drop time_range (category engines often don't support it)
          3. Drop category (fall back to general Google/DDG/Bing)
          4. Simplified query (strip stopwords)
        """
        query = inp.get("query", "").strip()
        if not query:
            return json.dumps({"error": "query is required"})

        n_results = min(inp.get("n_results", 5), 10)
        # Auto-detect category from query — LLM does not control this
        category = _auto_category(query)
        time_range = inp.get("time_range")
        engines = inp.get("engines")
        language = inp.get("language", "en")
        logger.info(
            "web_search: query=%r n_results=%d category=%s",
            query, n_results, category or "general",
        )

        from ..core.config import get_services_config
        svc = get_services_config()
        searxng_url = os.getenv("SEARXNG_URL") or svc.searxng_url

        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "language": language,
        }
        if category:
            params["categories"] = category
        if time_range:
            params["time_range"] = time_range
        if engines:
            params["engines"] = engines

        try:
            results = await self._searxng_query(searxng_url, params, n_results)

            # Retry 1: drop time_range (category engines often don't support it)
            if not results and "time_range" in params:
                retry_params = {k: v for k, v in params.items() if k != "time_range"}
                results = await self._searxng_query(searxng_url, retry_params, n_results)

            # Retry 2: drop categories (fall back to general search)
            if not results and "categories" in params:
                retry_params = {k: v for k, v in params.items()
                                if k not in ("categories", "time_range")}
                results = await self._searxng_query(searxng_url, retry_params, n_results)

            # Retry 3: simplified query
            if not results:
                simplified = _simplify_query(query)
                if simplified and simplified != query:
                    simple_params = {"q": simplified, "format": "json", "language": language}
                    results = await self._searxng_query(searxng_url, simple_params, n_results)

        except httpx.HTTPError as e:
            return json.dumps({
                "error": f"Search request failed: {e}",
                "query": query,
            })

        formatted = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],
            }
            for r in results
        ]

        if formatted:
            return json.dumps({
                "results": formatted,
                "count": len(formatted),
            })

        return json.dumps({
            "results": [],
            "count": 0,
            "message": (
                f"No results for '{query}'. "
                "Try simpler keywords or different categories."
            ),
        })

    async def _searxng_query(
        self, searxng_url: str, params: dict[str, Any], n_results: int,
    ) -> list[dict[str, Any]]:
        """Execute a single SearXNG query and return results."""
        client = self._get_client()
        resp = await client.get(
            f"{searxng_url}/search", params=params, timeout=15.0
        )
        resp.raise_for_status()
        return resp.json().get("results", [])[:n_results]

    async def _handle_fetch_url(self, inp: dict[str, Any]) -> str:
        """Fetch URL content with SSRF protection."""
        url = inp.get("url", "").strip()
        extract_text = inp.get("extract_text", True)
        timeout = min(inp.get("timeout", 30), 60)

        if not url:
            return json.dumps({"error": "url is required"})

        # Validate scheme
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return json.dumps({"error": "Only http/https URLs allowed"})

        if not parsed.hostname:
            return json.dumps({"error": "Invalid URL: no hostname"})

        # SSRF protection
        try:
            ip = socket.gethostbyname(parsed.hostname)
            ip_obj = ipaddress.ip_address(ip)
            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
            ):
                return json.dumps({
                    "error": f"Blocked: internal/private IP ({ip})"
                })
        except socket.gaierror:
            return json.dumps({
                "error": f"Cannot resolve hostname: {parsed.hostname}"
            })

        # Fetch
        try:
            client = self._get_client()
            resp = await client.get(
                url,
                headers={"User-Agent": "CCA/1.0"},
                timeout=timeout,
            )
        except httpx.HTTPError as e:
            return json.dumps({
                "error": f"Fetch failed: {e}",
                "url": url,
            })

        content = resp.text
        content_type = resp.headers.get("content-type", "")

        # HTML text extraction
        if extract_text and "text/html" in content_type:
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(content, "html.parser")
                for tag in soup(
                    ["script", "style", "nav", "footer", "header", "aside"]
                ):
                    tag.decompose()
                content = soup.get_text(separator="\n", strip=True)
            except ImportError:
                pass  # bs4 not available, return raw

        # Truncate at 500KB (vLLM context is 1M tokens; 500KB ≈ 125K tokens per fetch)
        content = content[:500_000]

        return json.dumps({
            "url": str(resp.url),
            "status": resp.status_code,
            "content_type": content_type,
            "content": content,
        })
