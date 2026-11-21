"""
Search MCP Server
Provides tools for AI agents to search Google, Stack Overflow, GitHub, and Reddit
with minimal local compute overhead and automatic ripgrep-powered snippet extraction
to keep agent context windows clean.
"""

from __future__ import annotations

import asyncio
import html
import os
import re
import shutil
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load `.env` from the project directory (if present)
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

mcp = FastMCP(
    "search-mcp",
    instructions=(
        "Search Google, Stack Overflow, GitHub, and Reddit to quickly look up documentation, "
        "error messages, Q&A answers, repositories, issues, code, and community discussions. "
        "All tools automatically parse HTML/boilerplate and use ripgrep to extract only the "
        "relevant matching lines and surrounding context so your context window stays small."
    ),
)

HTTP_TIMEOUT = 15.0
PAGE_FETCH_TIMEOUT = 8.0
USER_AGENT = "python:search-mcp:0.2.0 (by /u/search-mcp)"
RG_BINARY = shutil.which("rg") or "rg"
RG_BLOCK_SEP = "__RG_CONTEXT_BLOCK_SEP__"

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "get",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "my",
        "not",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "this",
        "to",
        "use",
        "using",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
    }
)

# Qualifiers used in GitHub / Google / Reddit search queries that shouldn't be grepped literally
_QUALIFIER_RE = re.compile(
    r"\b(?:site|repo|org|user|language|lang|is|in|type|label|author|subreddit|flair|path|ext|extension):[^\s]+",
    re.IGNORECASE,
)


def _truncate(text: str | None, max_chars: int = 2000) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


# ==============================================================================
# HTML Parsing & Ripgrep Context Extraction Pipeline
# ==============================================================================
class _HTMLToMarkdownParser(HTMLParser):
    """Lightweight stdlib HTML-to-Markdown/text parser that strips boilerplate and preserves code blocks."""

    _SKIP_TAGS = frozenset(
        {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "iframe"}
    )
    _BLOCK_TAGS = frozenset(
        {"p", "div", "section", "article", "main", "blockquote", "tr", "ul", "ol", "dl", "dt", "dd", "table"}
    )
    _HEADING_TAGS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "##### ", "h6": "###### "}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._in_pre = False
        self._in_code = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return

        if tag == "pre":
            self._in_pre = True
            self._chunks.append("\n```\n")
        elif tag == "code":
            self._in_code = True
            if not self._in_pre:
                self._chunks.append("`")
        elif tag == "br" or tag == "hr":
            self._chunks.append("\n")
        elif tag == "li":
            self._chunks.append("\n- ")
        elif tag in self._HEADING_TAGS:
            self._chunks.append(f"\n{self._HEADING_TAGS[tag]}")
        elif tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return

        if tag == "pre":
            self._in_pre = False
            self._chunks.append("\n```\n")
        elif tag == "code":
            if not self._in_pre:
                self._chunks.append("`")
            self._in_code = False
        elif tag in self._HEADING_TAGS or tag in self._BLOCK_TAGS or tag == "li":
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0 or not data:
            return
        if self._in_pre:
            self._chunks.append(data)
        else:
            # Collapse internal runs of horizontal whitespace while keeping single spaces
            cleaned = re.sub(r"[ \t\r\f\v]+", " ", data)
            self._chunks.append(cleaned)

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        raw = html.unescape(raw)
        # Clean up spaces around newlines and collapse >2 consecutive newlines
        lines = [line.rstrip() for line in raw.splitlines()]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _clean_html_or_text(text: str | None) -> str:
    """Convert HTML into clean line-oriented Markdown/plain text, or normalize plain text."""
    if not text:
        return ""
    stripped = text.strip()
    if not stripped:
        return ""
    # Detect HTML tags
    if "<" in stripped and ">" in stripped and re.search(
        r"</?(?:p|div|span|pre|code|a|ul|ol|li|br|h[1-6]|blockquote|html|body|table|tr|td)\b",
        stripped,
        re.IGNORECASE,
    ):
        parser = _HTMLToMarkdownParser()
        try:
            parser.feed(stripped)
            parser.close()
            parsed = parser.get_text()
            if parsed:
                return parsed
        except Exception:
            pass
    # Plain text normalization
    unescaped = html.unescape(stripped)
    lines = [line.rstrip() for line in unescaped.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _extract_query_terms(query: str) -> list[str]:
    """
    Parse a natural-language or qualified search query into distinct searchable terms & phrases.
    Preserves quoted phrases, error identifiers, dotted/camelCase symbols, and meaningful keywords.
    """
    if not query or not query.strip():
        return []

    # Strip search qualifiers like repo:foo/bar, site:example.com, language:python
    cleaned = _QUALIFIER_RE.sub(" ", query)

    terms: list[str] = []
    # 1. Extract quoted phrases first
    for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', cleaned):
        phrase = (match.group(1) or match.group(2) or "").strip()
        if len(phrase) >= 2:
            terms.append(phrase)
    cleaned = re.sub(r'"[^"]+"|\'[^\']+\'', " ", cleaned)

    # 2. Extract tokens (keep technical symbols like foo_bar, Foo.bar, --flag, HTTP_404)
    raw_tokens = re.findall(r"[A-Za-z0-9_./:\-]+", cleaned)
    meaningful: list[str] = []
    for tok in raw_tokens:
        t_clean = tok.strip(".:/-_")
        if len(t_clean) < 2:
            continue
        if t_clean.lower() not in _STOPWORDS:
            meaningful.append(t_clean)

    # Fallback if all tokens were stopwords
    if not terms and not meaningful:
        meaningful = [t.strip(".:/-_") for t in raw_tokens if len(t.strip(".:/-_")) >= 2]

    seen: set[str] = set()
    for item in terms + meaningful:
        low = item.lower()
        if low not in seen:
            seen.add(low)
            terms.append(item) if item not in terms else None

    # Deduplicate preserving order
    deduped: list[str] = []
    seen_low: set[str] = set()
    for item in terms + meaningful:
        low = item.lower()
        if low not in seen_low:
            seen_low.add(low)
            deduped.append(item)
    return deduped


def _build_rg_pattern(query: str | None, filter_pattern: str | None = None) -> tuple[str, list[str], bool]:
    """
    Build a ripgrep regex pattern and a list of scoring terms from `query` and/or `filter_pattern`.
    Returns `(rg_pattern, scoring_terms, is_raw_regex)`.
    """
    if filter_pattern and filter_pattern.strip():
        pat = filter_pattern.strip()
        terms = _extract_query_terms(pat) or [pat]
        return pat, terms, True

    if not query or not query.strip():
        return "", [], False

    terms = _extract_query_terms(query)
    if not terms:
        escaped = re.escape(query.strip())
        return escaped, [query.strip()], False

    # Build alternation pattern of escaped terms, longest first
    sorted_terms = sorted(terms, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in sorted_terms)
    return pattern, terms, False


async def _run_ripgrep(
    text: str,
    pattern: str,
    context_lines: int = 3,
    line_numbers: bool = False,
    fixed_strings: bool = False,
) -> str | None:
    """
    Run `ripgrep` (`rg`) asynchronously over `text` via stdin and return matching blocks.
    Returns None if no matches are found or if `rg` fails.
    """
    if not text or not pattern:
        return None

    ctx = max(0, min(20, context_lines))
    cmd = [
        RG_BINARY,
        "--ignore-case",
        "--no-heading",
        "--color=never",
        f"--context={ctx}",
        f"--context-separator={RG_BLOCK_SEP}",
    ]
    if line_numbers:
        cmd.append("--line-number")
    if fixed_strings:
        cmd.append("--fixed-strings")
    cmd.extend(["--regexp", pattern, "-"])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, _ = await asyncio.wait_for(
            proc.communicate(input=text.encode("utf-8", errors="replace")),
            timeout=5.0,
        )
    except (FileNotFoundError, asyncio.TimeoutError, OSError):
        return None

    # Exit code 0 = matches found, 1 = no matches, 2 = error (e.g. invalid regex)
    if proc.returncode == 2 and not fixed_strings:
        # Retry with --fixed-strings if user passed an invalid regex in filter_pattern
        return await _run_ripgrep(
            text,
            pattern,
            context_lines=context_lines,
            line_numbers=line_numbers,
            fixed_strings=True,
        )

    if proc.returncode != 0 or not stdout_bytes:
        return None

    return stdout_bytes.decode("utf-8", errors="replace").strip()


def _rank_and_trim_blocks(
    rg_output: str,
    terms: list[str],
    max_chars: int,
) -> str:
    """
    Split ripgrep output into context blocks, score each block by how many distinct
    query terms and code/technical signals it contains, select the top blocks that fit
    within `max_chars`, and restore their original document order.
    """
    raw_blocks = [b.strip() for b in rg_output.split(RG_BLOCK_SEP) if b.strip()]
    if not raw_blocks:
        return ""

    joined_all = "\n...\n".join(raw_blocks)
    if len(joined_all) <= max_chars:
        return joined_all

    low_terms = [t.lower() for t in terms if t]
    scored_blocks: list[tuple[float, int, str]] = []
    for idx, block in enumerate(raw_blocks):
        block_low = block.lower()
        distinct_hits = sum(1 for t in low_terms if t in block_low)
        total_hits = sum(block_low.count(t) for t in low_terms)
        has_code = 1.5 if ("```" in block or "    " in block or "def " in block or "function " in block) else 0.0
        # Slight preference for earlier blocks when scores tie
        position_bonus = max(0.0, 0.5 - (idx * 0.02))
        score = (distinct_hits * 10.0) + min(total_hits, 10) * 1.5 + has_code + position_bonus
        scored_blocks.append((score, idx, block))

    # Pick highest-scoring blocks until budget is reached
    scored_blocks.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[int, str]] = []
    current_chars = 0

    for _score, idx, block in scored_blocks:
        sep_len = 5 if selected else 0  # len("\n...\n")
        if current_chars + sep_len + len(block) <= max_chars:
            selected.append((idx, block))
            current_chars += sep_len + len(block)
        elif not selected:
            # First block alone exceeds max_chars; truncate it
            selected.append((idx, _truncate(block, max_chars)))
            break

    # Restore document order so excerpts read top-to-bottom
    selected.sort(key=lambda item: item[0])
    return "\n...\n".join(block for _idx, block in selected)


async def _extract_relevant_text(
    raw_text: str | None,
    query: str | None = None,
    filter_pattern: str | None = None,
    context_lines: int = 3,
    max_chars: int = 1500,
    line_numbers: bool = False,
    clean_html: bool = True,
    always_filter_if_pattern: bool = True,
) -> str:
    """
    Clean raw text/HTML and use `ripgrep` to extract only relevant matching blocks + context.
    Falls back to clean truncated text if no query/pattern is given or if ripgrep finds no literal match.
    """
    if not raw_text:
        return ""

    cleaned = _clean_html_or_text(raw_text) if clean_html else raw_text.strip()
    if not cleaned:
        return ""

    has_explicit_filter = bool(filter_pattern and filter_pattern.strip())
    # If text is already very concise and no explicit filter_pattern or line_numbers was requested, return it directly
    if len(cleaned) <= min(max_chars, 400) and not has_explicit_filter and not line_numbers:
        return cleaned

    pattern, terms, _is_raw = _build_rg_pattern(query, filter_pattern)
    if pattern and (has_explicit_filter or len(cleaned) > max_chars or always_filter_if_pattern):
        rg_out = await _run_ripgrep(
            cleaned,
            pattern=pattern,
            context_lines=context_lines,
            line_numbers=line_numbers,
        )
        if rg_out:
            extracted = _rank_and_trim_blocks(rg_out, terms=terms, max_chars=max_chars)
            if extracted:
                return extracted
        elif has_explicit_filter:
            return f"_(No lines matched filter pattern `{filter_pattern}`; showing lead excerpt)_\n{_truncate(cleaned, min(max_chars, 600))}"

    return _truncate(cleaned, max_chars)


async def _fetch_and_extract_webpage(
    client: httpx.AsyncClient,
    url: str,
    query: str,
    filter_pattern: str | None,
    context_lines: int,
    max_chars: int = 1000,
) -> str:
    """
    Fetch a web page, strip HTML/boilerplate, and use ripgrep to extract the most relevant snippets
    matching the agent's search query.
    """
    try:
        resp = await client.get(url, timeout=PAGE_FETCH_TIMEOUT, follow_redirects=True)
        if resp.status_code != 200:
            return ""
        content_type = resp.headers.get("content-type", "").lower()
        if not any(t in content_type for t in ("text/html", "text/plain", "application/xhtml", "text/markdown")):
            return ""
        raw_html = resp.text
    except Exception:
        return ""

    if not raw_html:
        return ""

    return await _extract_relevant_text(
        raw_html,
        query=query,
        filter_pattern=filter_pattern,
        context_lines=context_lines,
        max_chars=max_chars,
        clean_html=True,
    )


# ==============================================================================
# 1. Google Custom Search JSON API
# ==============================================================================
@mcp.tool()
async def search_google(
    query: str,
    num_results: int = 5,
    site_search: str | None = None,
    fetch_and_extract: bool = True,
    filter_pattern: str | None = None,
    context_lines: int = 2,
) -> str:
    """
    Search the web using the official Google Custom Search JSON API, and automatically fetch
    and parse the top results with ripgrep so only relevant matching snippets enter your context.

    Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` in the `.env` file.

    Args:
        query: The search query string.
        num_results: Number of results to return (1 to 10, default 5).
        site_search: Optional domain to restrict results to (e.g., "docs.python.org").
        fetch_and_extract: If True (default), fetches the top result pages, strips HTML boilerplate,
                           and runs ripgrep to extract only lines matching your query/filter_pattern.
        filter_pattern: Optional explicit ripgrep regex/pattern to filter fetched page content.
                        Defaults to keywords parsed from `query`.
        context_lines: Number of surrounding context lines around each ripgrep match (default 2).
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    cse_id = os.getenv("GOOGLE_CSE_ID", "").strip()

    if not api_key or not cse_id:
        return (
            "Error: Google Custom Search requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` "
            "to be set in your `.env` file.\n"
            "See `.env.example` for links to create a free Google API key and Programmable Search Engine ID."
        )

    num_results = max(1, min(10, num_results))
    params: dict[str, str | int] = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results,
    }
    if site_search:
        params["siteSearch"] = site_search
        params["siteSearchFilter"] = "i"

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        try:
            resp = await client.get("https://www.googleapis.com/customsearch/v1", params=params)
            if resp.status_code != 200:
                return f"Google Search API Error (HTTP {resp.status_code}): {resp.text}"
            data = resp.json()
        except httpx.HTTPError as exc:
            return f"Network error querying Google Search API: {exc}"

        items = data.get("items", [])
        if not items:
            return f"No Google search results found for: {query!r}"

        extracted_pages: list[str] = [""] * len(items)
        if fetch_and_extract:
            # Concurrently fetch & ripgrep-extract up to top 5 result pages
            fetch_count = min(len(items), 5)
            tasks = [
                _fetch_and_extract_webpage(
                    client,
                    items[i].get("link", ""),
                    query=query,
                    filter_pattern=filter_pattern,
                    context_lines=context_lines,
                    max_chars=900,
                )
                for i in range(fetch_count)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, res in enumerate(results):
                if isinstance(res, str) and res:
                    extracted_pages[i] = res

    lines = [f"### Google Search Results for `{query}`\n"]
    for idx, item in enumerate(items, start=1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        snippet = item.get("snippet", "").replace("\n", " ").strip()
        entry = [f"{idx}. **[{title}]({link})**", f"   - **Snippet**: {snippet}"]
        page_extract = extracted_pages[idx - 1]
        if page_extract:
            indented = "\n".join(f"     {line}" for line in page_extract.splitlines())
            entry.append(f"   - **Extracted Page Matches (via `rg`)**:\n{indented}")
        lines.append("\n".join(entry) + "\n")

    return "\n".join(lines)


# ==============================================================================
# 2. Stack Overflow (Stack Exchange API v2.3)
# ==============================================================================
STACKEXCHANGE_API_BASE = "https://api.stackexchange.com/2.3"


@mcp.tool()
async def search_stackoverflow(
    query: str,
    tagged: str | None = None,
    sort: Literal["relevance", "votes", "creation", "activity"] = "relevance",
    num_results: int = 5,
    include_top_answers: bool = True,
    filter_pattern: str | None = None,
    context_lines: int = 3,
) -> str:
    """
    Search Stack Overflow questions, retrieve their accepted/top-voted answers, strip HTML,
    and use ripgrep to extract the relevant problem statement, code blocks, and solution lines.
    Works without an API key (300 requests/day free); set `STACKEXCHANGE_KEY` in `.env` for 10,000/day.

    Args:
        query: Search query (error message, concept, or keywords).
        tagged: Optional semicolon-separated tags (e.g., "python;asyncio" or "typescript").
        sort: Sort order ("relevance", "votes", "creation", "activity"). Default "relevance".
        num_results: Number of questions to return (1 to 10, default 5).
        include_top_answers: If True, fetches the accepted or highest-voted answer body for the returned questions.
        filter_pattern: Optional explicit ripgrep regex/pattern to extract specific lines from questions/answers.
                        Defaults to keywords parsed from `query`.
        context_lines: Number of surrounding context lines around each ripgrep match (default 3).
    """
    num_results = max(1, min(10, num_results))
    se_key = os.getenv("STACKEXCHANGE_KEY", "").strip()

    params: dict[str, str | int] = {
        "order": "desc",
        "sort": sort,
        "q": query,
        "site": "stackoverflow",
        "pagesize": num_results,
        "filter": "withbody",
    }
    if tagged:
        params["tagged"] = tagged
    if se_key:
        params["key"] = se_key

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        try:
            resp = await client.get(f"{STACKEXCHANGE_API_BASE}/search/advanced", params=params)
            if resp.status_code != 200:
                return f"Stack Exchange API Error (HTTP {resp.status_code}): {resp.text}"
            data = resp.json()
        except httpx.HTTPError as exc:
            return f"Network error querying Stack Exchange API: {exc}"

        items = data.get("items", [])
        if not items:
            return f"No Stack Overflow results found for: {query!r}"

        # Optionally batch-fetch answers for the found questions in a single API call
        answers_by_question: dict[int, list[dict]] = {}
        if include_top_answers:
            q_ids = [str(item["question_id"]) for item in items if item.get("answer_count", 0) > 0]
            if q_ids:
                ans_params: dict[str, str | int] = {
                    "order": "desc",
                    "sort": "votes",
                    "site": "stackoverflow",
                    "pagesize": min(30, len(q_ids) * 2),
                    "filter": "withbody",
                }
                if se_key:
                    ans_params["key"] = se_key
                try:
                    ans_resp = await client.get(
                        f"{STACKEXCHANGE_API_BASE}/questions/{';'.join(q_ids)}/answers",
                        params=ans_params,
                    )
                    if ans_resp.status_code == 200:
                        for ans in ans_resp.json().get("items", []):
                            qid = ans.get("question_id")
                            if qid is not None:
                                answers_by_question.setdefault(qid, []).append(ans)
                except httpx.HTTPError:
                    pass

    lines = [
        f"### Stack Overflow Results for `{query}` "
        f"(quota remaining: {data.get('quota_remaining', 'unknown')})\n"
    ]

    for idx, q in enumerate(items, start=1):
        qid = q.get("question_id")
        title = html.unescape(q.get("title", "Untitled"))
        link = q.get("link", f"https://stackoverflow.com/q/{qid}")
        score = q.get("score", 0)
        ans_count = q.get("answer_count", 0)
        is_answered = q.get("is_answered", False)
        tags = ", ".join(q.get("tags", []))
        q_body = await _extract_relevant_text(
            q.get("body", ""),
            query=query,
            filter_pattern=filter_pattern,
            context_lines=context_lines,
            max_chars=500,
            clean_html=True,
        )

        status = "Answered" if is_answered else "Unanswered"
        lines.append(
            f"#### {idx}. [{title}]({link})\n"
            f"- **Question ID**: `{qid}` | **Score**: {score} | **Answers**: {ans_count} ({status}) | **Tags**: `{tags}`\n"
            f"- **Relevant Question Excerpt**:\n{q_body}\n"
        )

        q_answers = answers_by_question.get(qid, [])
        if q_answers:
            # Prefer accepted answer first, then highest score
            q_answers.sort(key=lambda a: (bool(a.get("is_accepted")), a.get("score", 0)), reverse=True)
            best = q_answers[0]
            acc_badge = " (Accepted Answer)" if best.get("is_accepted") else ""
            a_score = best.get("score", 0)
            a_body = await _extract_relevant_text(
                best.get("body", ""),
                query=query,
                filter_pattern=filter_pattern,
                context_lines=context_lines,
                max_chars=1200,
                clean_html=True,
            )
            lines.append(
                f"- **Top Answer{acc_badge} (Score: {a_score})**:\n{a_body}\n"
            )

    return "\n".join(lines)


@mcp.tool()
async def get_stackoverflow_question(
    question_id: int,
    max_answers: int = 3,
    query: str | None = None,
    context_lines: int = 4,
) -> str:
    """
    Fetch details and top answers for a specific Stack Overflow question by its numeric ID,
    converting HTML to clean Markdown and optionally filtering with ripgrep for a specific query.
    Works without an API key; uses `STACKEXCHANGE_KEY` from `.env` if available.

    Args:
        question_id: The numeric Stack Overflow question ID.
        max_answers: Maximum number of top/accepted answers to include (default 3).
        query: Optional search query or regex pattern to filter the question and answers via ripgrep
               so only matching sections + surrounding context lines are returned.
        context_lines: Number of surrounding context lines around each ripgrep match (default 4).
    """
    se_key = os.getenv("STACKEXCHANGE_KEY", "").strip()
    params: dict[str, str | int] = {
        "site": "stackoverflow",
        "filter": "withbody",
    }
    if se_key:
        params["key"] = se_key

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        try:
            q_resp = await client.get(f"{STACKEXCHANGE_API_BASE}/questions/{question_id}", params=params)
            if q_resp.status_code != 200:
                return f"Stack Exchange API Error (HTTP {q_resp.status_code}): {q_resp.text}"
            q_items = q_resp.json().get("items", [])
            if not q_items:
                return f"Question ID {question_id} not found on Stack Overflow."
            q = q_items[0]

            ans_params = {
                **params,
                "order": "desc",
                "sort": "votes",
                "pagesize": max(1, min(10, max_answers)),
            }
            a_resp = await client.get(
                f"{STACKEXCHANGE_API_BASE}/questions/{question_id}/answers",
                params=ans_params,
            )
            answers = a_resp.json().get("items", []) if a_resp.status_code == 200 else []
        except httpx.HTTPError as exc:
            return f"Network error querying Stack Exchange API: {exc}"

    title = html.unescape(q.get("title", "Untitled"))
    link = q.get("link", f"https://stackoverflow.com/q/{question_id}")
    score = q.get("score", 0)
    tags = ", ".join(q.get("tags", []))
    body = await _extract_relevant_text(
        q.get("body", ""),
        query=query,
        filter_pattern=query,
        context_lines=context_lines,
        max_chars=1800 if query else 3000,
        clean_html=True,
        always_filter_if_pattern=bool(query),
    )

    lines = [
        f"### [{title}]({link})",
        f"**Score**: {score} | **Tags**: `{tags}`" + (f" | **Filtered by**: `{query}`" if query else "") + "\n",
        "#### Question",
        body,
        "\n#### Answers",
    ]

    if not answers:
        lines.append("_No answers found._")
    else:
        answers.sort(key=lambda a: (bool(a.get("is_accepted")), a.get("score", 0)), reverse=True)
        for idx, ans in enumerate(answers[:max_answers], start=1):
            acc = " [ACCEPTED]" if ans.get("is_accepted") else ""
            a_score = ans.get("score", 0)
            a_body = await _extract_relevant_text(
                ans.get("body", ""),
                query=query,
                filter_pattern=query,
                context_lines=context_lines,
                max_chars=1800 if query else 3000,
                clean_html=True,
                always_filter_if_pattern=bool(query),
            )
            lines.append(f"##### Answer {idx}{acc} (Score: {a_score})\n{a_body}\n")

    return "\n".join(lines)


# ==============================================================================
# 3. GitHub REST API
# ==============================================================================
GITHUB_API_BASE = "https://api.github.com"


def _github_headers(text_match: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github.text-match+json" if text_match else "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@mcp.tool()
async def search_github(
    query: str,
    search_type: Literal["repositories", "issues", "code"] = "repositories",
    sort: str | None = None,
    num_results: int = 5,
    filter_pattern: str | None = None,
    context_lines: int = 3,
) -> str:
    """
    Search GitHub for repositories, issues/PRs, or code, using ripgrep to extract relevant
    matching sections from issue bodies and code snippets.
    Works without an API key for `repositories` and `issues` (10 requests/min free).
    Setting `GITHUB_TOKEN` in `.env` increases the limit to 30 requests/min and enables `code` search.

    Args:
        query: GitHub search query (supports qualifiers like `language:python`, `repo:owner/name`, `is:issue`).
        search_type: One of "repositories", "issues", or "code". Default "repositories".
        sort: Optional sort field:
              - For repositories: "stars", "forks", "help-wanted-issues", "updated"
              - For issues: "comments", "reactions", "created", "updated"
              - For code: "indexed"
              Leave None for best match relevance.
        num_results: Number of results to return (1 to 10, default 5).
        filter_pattern: Optional explicit ripgrep regex/pattern to filter issue bodies or code fragments.
                        Defaults to keywords parsed from `query` (with GitHub qualifiers stripped).
        context_lines: Number of surrounding context lines around each ripgrep match (default 3).
    """
    num_results = max(1, min(10, num_results))
    token = os.getenv("GITHUB_TOKEN", "").strip()

    if search_type == "code" and not token:
        return (
            "Error: GitHub's Code Search API endpoint requires authentication.\n"
            "Either add a free `GITHUB_TOKEN` (Personal Access Token with no special scopes needed for public repos) "
            "to your `.env` file, or use `search_type='repositories'` / `search_type='issues'` which work without a key."
        )

    params: dict[str, str | int] = {
        "q": query,
        "per_page": num_results,
    }
    if sort:
        params["sort"] = sort

    headers = _github_headers(text_match=(search_type == "code"))

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers) as client:
        try:
            resp = await client.get(f"{GITHUB_API_BASE}/search/{search_type}", params=params)
            if resp.status_code != 200:
                return f"GitHub API Error (HTTP {resp.status_code}): {resp.text}"
            data = resp.json()
        except httpx.HTTPError as exc:
            return f"Network error querying GitHub API: {exc}"

    items = data.get("items", [])
    total = data.get("total_count", 0)
    if not items:
        return f"No GitHub {search_type} found for: {query!r}"

    lines = [f"### GitHub {search_type.capitalize()} Results for `{query}` (total matches: {total})\n"]

    if search_type == "repositories":
        for idx, repo in enumerate(items, start=1):
            full_name = repo.get("full_name", "")
            html_url = repo.get("html_url", "")
            desc = repo.get("description") or "No description."
            stars = repo.get("stargazers_count", 0)
            lang = repo.get("language") or "Unknown"
            topics = ", ".join(repo.get("topics", [])[:6])
            updated = repo.get("updated_at", "")[:10]
            lines.append(
                f"{idx}. **[{full_name}]({html_url})** ⭐ {stars} | Language: `{lang}` | Updated: {updated}\n"
                f"   {desc}\n"
                + (f"   Topics: `{topics}`\n" if topics else "")
            )
    elif search_type == "issues":
        for idx, issue in enumerate(items, start=1):
            title = issue.get("title", "Untitled")
            html_url = issue.get("html_url", "")
            state = issue.get("state", "unknown")
            comments = issue.get("comments", 0)
            body = await _extract_relevant_text(
                issue.get("body") or "",
                query=query,
                filter_pattern=filter_pattern,
                context_lines=context_lines,
                max_chars=500,
                clean_html=True,
            )
            kind = "PR" if "pull_request" in issue else "Issue"
            lines.append(
                f"{idx}. **[{title}]({html_url})** ({kind} • `{state}` • {comments} comments)\n"
                + (f"   {body}\n" if body else "")
            )
    elif search_type == "code":
        for idx, code_item in enumerate(items, start=1):
            name = code_item.get("name", "")
            path = code_item.get("path", "")
            html_url = code_item.get("html_url", "")
            repo_name = code_item.get("repository", {}).get("full_name", "")
            fragments = [
                m.get("fragment", "")
                for m in code_item.get("text_matches", [])
                if m.get("fragment")
            ]
            raw_fragments = "\n---\n".join(fragments)
            snippet = await _extract_relevant_text(
                raw_fragments,
                query=query,
                filter_pattern=filter_pattern,
                context_lines=context_lines,
                max_chars=800,
                clean_html=False,
            )
            lines.append(
                f"{idx}. **`{repo_name}`: [{path}]({html_url})** (`{name}`)\n"
                + (f"```\n{snippet}\n```\n" if snippet else "")
            )

    return "\n".join(lines)


@mcp.tool()
async def get_github_file(
    repo: str,
    path: str,
    ref: str | None = None,
    query: str | None = None,
    context_lines: int = 5,
    max_chars: int = 6000,
) -> str:
    """
    Fetch a file from a public GitHub repository and optionally filter it with ripgrep (`rg -n -C`)
    so only matching lines (with exact line numbers and surrounding context) enter your context window.
    Works without an API key; uses `GITHUB_TOKEN` from `.env` if available.

    Args:
        repo: Repository in "owner/name" format (e.g., "modelcontextprotocol/python-sdk").
        path: File path within the repository (e.g., "README.md" or "src/mcp/server/fastmcp/server.py").
        ref: Optional branch, tag, or commit SHA (defaults to the repository's default branch).
        query: Optional search query or regex pattern to run through ripgrep against the file.
               When provided, returns only matching lines + `context_lines` with line numbers.
        context_lines: Number of surrounding context lines around each ripgrep match (default 5).
        max_chars: Maximum characters to return (default 6000).
    """
    headers = _github_headers()
    headers["Accept"] = "application/vnd.github.raw+json"

    params: dict[str, str] = {}
    if ref:
        params["ref"] = ref

    url = f"{GITHUB_API_BASE}/repos/{repo.strip('/')}/contents/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers) as client:
        try:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return f"GitHub API Error (HTTP {resp.status_code}) fetching `{repo}/{path}`: {resp.text}"
            content = resp.text
        except httpx.HTTPError as exc:
            return f"Network error fetching file from GitHub: {exc}"

    max_chars = max(500, min(20000, max_chars))
    total_lines = len(content.splitlines())

    if query and query.strip():
        extracted = await _extract_relevant_text(
            content,
            query=query,
            filter_pattern=query,
            context_lines=context_lines,
            max_chars=max_chars,
            line_numbers=True,
            clean_html=False,
            always_filter_if_pattern=True,
        )
        return (
            f"### `{repo}` — `{path}` ({total_lines} lines total, ripgrep matches for `{query}`)\n\n"
            f"```\n{extracted}\n```"
        )

    return f"### `{repo}` — `{path}` ({total_lines} lines)\n\n```\n{_truncate(content, max_chars)}\n```"


# ==============================================================================
# 4. Reddit API (Public JSON & Optional OAuth2)
# ==============================================================================
REDDIT_PUBLIC_BASE = "https://www.reddit.com"
REDDIT_OAUTH_BASE = "https://oauth.reddit.com"
_reddit_token_cache: dict[str, str | float] = {"token": "", "expires_at": 0.0}


async def _get_reddit_auth(client: httpx.AsyncClient) -> tuple[str, dict[str, str]]:
    """
    Return (base_url, headers) for Reddit requests.
    Uses OAuth2 client_credentials if `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` are set,
    otherwise falls back to public `.json` endpoints.
    """
    user_agent = os.getenv("REDDIT_USER_AGENT", "").strip() or USER_AGENT
    headers = {"User-Agent": user_agent, "Accept": "application/json"}

    client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return REDDIT_PUBLIC_BASE, headers

    now = time.time()
    cached_token = str(_reddit_token_cache.get("token", ""))
    expires_at = float(_reddit_token_cache.get("expires_at", 0.0))
    if cached_token and now < expires_at:
        headers["Authorization"] = f"Bearer {cached_token}"
        return REDDIT_OAUTH_BASE, headers

    try:
        resp = await client.post(
            f"{REDDIT_PUBLIC_BASE}/api/v1/access_token",
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            headers=headers,
        )
        if resp.status_code == 200:
            token_data = resp.json()
            access_token = token_data.get("access_token", "")
            expires_in = int(token_data.get("expires_in", 3600))
            if access_token:
                _reddit_token_cache["token"] = access_token
                _reddit_token_cache["expires_at"] = now + max(60, expires_in - 60)
                headers["Authorization"] = f"Bearer {access_token}"
                return REDDIT_OAUTH_BASE, headers
    except httpx.HTTPError:
        pass

    return REDDIT_PUBLIC_BASE, headers


@mcp.tool()
async def search_reddit(
    query: str,
    subreddit: str | None = None,
    sort: Literal["relevance", "hot", "top", "new", "comments"] = "relevance",
    time_filter: Literal["all", "year", "month", "week", "day", "hour"] = "all",
    num_results: int = 5,
    filter_pattern: str | None = None,
    context_lines: int = 2,
) -> str:
    """
    Search Reddit posts across all of Reddit or within a specific subreddit, using ripgrep
    to extract only the relevant matching sections from post bodies.
    Works without an API key via Reddit's public JSON endpoints; optionally set
    `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` in `.env` for higher OAuth rate limits.

    Args:
        query: Search query string.
        subreddit: Optional subreddit name to restrict search to (e.g., "python" or "LocalLLaMA", without "r/").
        sort: Sort order ("relevance", "hot", "top", "new", "comments"). Default "relevance".
        time_filter: Time window for results ("all", "year", "month", "week", "day", "hour"). Default "all".
        num_results: Number of posts to return (1 to 15, default 5).
        filter_pattern: Optional explicit ripgrep regex/pattern to extract matching lines from post bodies.
                        Defaults to keywords parsed from `query`.
        context_lines: Number of surrounding context lines around each ripgrep match (default 2).
    """
    num_results = max(1, min(15, num_results))
    clean_sub = subreddit.strip().lstrip("/").removeprefix("r/").strip("/") if subreddit else None

    params: dict[str, str | int] = {
        "q": query,
        "sort": sort,
        "t": time_filter,
        "limit": num_results,
        "raw_json": 1,
    }
    if clean_sub:
        params["restrict_sr"] = "on"

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        base_url, headers = await _get_reddit_auth(client)
        is_oauth = base_url == REDDIT_OAUTH_BASE
        suffix = "" if is_oauth else ".json"

        if clean_sub:
            url = f"{base_url}/r/{clean_sub}/search{suffix}"
        else:
            url = f"{base_url}/search{suffix}"

        try:
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                return f"Reddit API Error (HTTP {resp.status_code}): {resp.text}"
            data = resp.json()
        except httpx.HTTPError as exc:
            return f"Network error querying Reddit: {exc}"
        except ValueError:
            return "Error decoding Reddit JSON response."

    children = data.get("data", {}).get("children", [])
    if not children:
        scope = f" in r/{clean_sub}" if clean_sub else ""
        return f"No Reddit results found for: {query!r}{scope}"

    scope_title = f" in `r/{clean_sub}`" if clean_sub else ""
    lines = [f"### Reddit Search Results for `{query}`{scope_title}\n"]

    for idx, child in enumerate(children, start=1):
        post = child.get("data", {})
        post_id = post.get("id", "")
        title = post.get("title", "Untitled")
        sub_prefixed = post.get("subreddit_name_prefixed", f"r/{post.get('subreddit', '')}")
        author = post.get("author", "unknown")
        score = post.get("score", 0)
        num_comments = post.get("num_comments", 0)
        permalink = post.get("permalink", "")
        link = f"https://www.reddit.com{permalink}" if permalink else post.get("url", "")
        external_url = post.get("url", "")
        is_self = post.get("is_self", True)
        selftext = await _extract_relevant_text(
            post.get("selftext", ""),
            query=query,
            filter_pattern=filter_pattern,
            context_lines=context_lines,
            max_chars=500,
            clean_html=False,
        )

        meta = (
            f"- **ID**: `{post_id}` | **Subreddit**: `{sub_prefixed}` | "
            f"**Score**: {score} | **Comments**: {num_comments} | **Author**: `u/{author}`"
        )
        entry = [f"#### {idx}. [{title}]({link})", meta]
        if not is_self and external_url and external_url != link:
            entry.append(f"- **External Link**: {external_url}")
        if selftext:
            entry.append(f"- **Relevant Excerpt**:\n{selftext}")
        lines.append("\n".join(entry) + "\n")

    return "\n".join(lines)


@mcp.tool()
async def get_reddit_post(
    post_id_or_url: str,
    max_comments: int = 5,
    query: str | None = None,
    context_lines: int = 3,
) -> str:
    """
    Fetch a Reddit post and its comments by post ID or URL, optionally using ripgrep to filter
    the post body and comments down to only the lines relevant to `query`.
    Works without an API key; uses `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` from `.env` if available.

    Args:
        post_id_or_url: Reddit post ID (e.g., "1a2b3c4" or "t3_1a2b3c4") or full Reddit post URL.
        max_comments: Maximum number of comments to include (1 to 15, default 5).
        query: Optional search query or regex pattern to filter the post body and comments via ripgrep
               so only matching comments and surrounding context lines are returned.
        context_lines: Number of surrounding context lines around each ripgrep match (default 3).
    """
    max_comments = max(1, min(15, max_comments))
    raw = post_id_or_url.strip()

    if raw.startswith(("http://", "https://")):
        parsed = urlparse(raw)
        path = parsed.path.rstrip("/")
        if path.endswith(".json"):
            path = path[:-5]
    else:
        clean_id = raw.removeprefix("t3_").strip("/")
        path = f"/comments/{clean_id}"

    # Fetch a wider comment pool when filtering by `query` so ripgrep can surface matching comments
    fetch_limit = min(50, max_comments * 4) if query else max_comments
    params: dict[str, str | int] = {
        "sort": "top",
        "limit": fetch_limit,
        "raw_json": 1,
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        base_url, headers = await _get_reddit_auth(client)
        is_oauth = base_url == REDDIT_OAUTH_BASE
        suffix = "" if is_oauth else ".json"
        url = f"{base_url}{path}{suffix}"

        try:
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                return f"Reddit API Error (HTTP {resp.status_code}): {resp.text}"
            data = resp.json()
        except httpx.HTTPError as exc:
            return f"Network error fetching Reddit post: {exc}"
        except ValueError:
            return "Error decoding Reddit JSON response."

    if not isinstance(data, list) or not data:
        return f"Reddit post {post_id_or_url!r} not found."

    post_children = data[0].get("data", {}).get("children", [])
    if not post_children:
        return f"Reddit post {post_id_or_url!r} not found."

    post = post_children[0].get("data", {})
    title = post.get("title", "Untitled")
    sub_prefixed = post.get("subreddit_name_prefixed", f"r/{post.get('subreddit', '')}")
    author = post.get("author", "unknown")
    score = post.get("score", 0)
    num_comments = post.get("num_comments", 0)
    permalink = post.get("permalink", "")
    link = f"https://www.reddit.com{permalink}" if permalink else post.get("url", "")
    raw_selftext = post.get("selftext", "")
    selftext = await _extract_relevant_text(
        raw_selftext,
        query=query,
        filter_pattern=query,
        context_lines=context_lines,
        max_chars=1500 if query else 3000,
        clean_html=False,
        always_filter_if_pattern=bool(query),
    )
    external_url = post.get("url", "")
    is_self = post.get("is_self", True)

    lines = [
        f"### [{title}]({link})",
        f"**Subreddit**: `{sub_prefixed}` | **Score**: {score} | **Comments**: {num_comments} | **Author**: `u/{author}`"
        + (f" | **Filtered by**: `{query}`" if query else "")
        + "\n",
    ]
    if not is_self and external_url and external_url != link:
        lines.append(f"**Link**: {external_url}\n")
    if selftext:
        lines.append(f"#### Post Body\n{selftext}\n")

    lines.append("#### Top Relevant Comments" if query else "#### Top Comments")
    comment_children = data[1].get("data", {}).get("children", []) if len(data) > 1 else []
    comments = [
        c.get("data", {})
        for c in comment_children
        if c.get("kind") == "t1" and c.get("data", {}).get("body")
    ]

    if not comments:
        lines.append("_No comments found._")
    elif query and query.strip():
        pattern, terms, _ = _build_rg_pattern(query, query)
        matched_comments: list[tuple[int, dict, str]] = []
        for c in comments:
            c_body_raw = c.get("body", "")
            rg_out = await _run_ripgrep(c_body_raw, pattern=pattern, context_lines=context_lines)
            if rg_out:
                trimmed = _rank_and_trim_blocks(rg_out, terms=terms, max_chars=1000)
                if trimmed:
                    matched_comments.append((c.get("score", 0), c, trimmed))

        if matched_comments:
            for idx, (_score, c, excerpt) in enumerate(matched_comments[:max_comments], start=1):
                c_author = c.get("author", "unknown")
                c_score = c.get("score", 0)
                lines.append(f"##### Comment {idx} by `u/{c_author}` (Score: {c_score})\n{excerpt}\n")
        else:
            lines.append(f"_(No comments matched `{query}`; showing top comments)_\n")
            for idx, c in enumerate(comments[:max_comments], start=1):
                c_author = c.get("author", "unknown")
                c_score = c.get("score", 0)
                c_body = _truncate(c.get("body", ""), 800)
                lines.append(f"##### Comment {idx} by `u/{c_author}` (Score: {c_score})\n{c_body}\n")
    else:
        for idx, c in enumerate(comments[:max_comments], start=1):
            c_author = c.get("author", "unknown")
            c_score = c.get("score", 0)
            c_body = _truncate(c.get("body", ""), 1500)
            lines.append(f"##### Comment {idx} by `u/{c_author}` (Score: {c_score})\n{c_body}\n")

    return "\n".join(lines)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
