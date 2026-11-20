"""
Search MCP Server
Provides tools for AI agents to search Google, Stack Overflow, GitHub, and Reddit
with minimal local compute overhead.
"""

from __future__ import annotations

import os
import time
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
        "error messages, Q&A answers, repositories, issues, code, and community discussions."
    ),
)

HTTP_TIMEOUT = 15.0
USER_AGENT = "python:search-mcp:0.1.0 (by /u/search-mcp)"


def _truncate(text: str | None, max_chars: int = 2000) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


# ==============================================================================
# 1. Google Custom Search JSON API
# ==============================================================================
@mcp.tool()
async def search_google(
    query: str,
    num_results: int = 5,
    site_search: str | None = None,
) -> str:
    """
    Search the web using the official Google Custom Search JSON API.

    Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` in the `.env` file.

    Args:
        query: The search query string.
        num_results: Number of results to return (1 to 10, default 5).
        site_search: Optional domain to restrict results to (e.g., "docs.python.org").
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

    lines = [f"### Google Search Results for `{query}`\n"]
    for idx, item in enumerate(items, start=1):
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        snippet = item.get("snippet", "").replace("\n", " ").strip()
        lines.append(f"{idx}. **[{title}]({link})**\n   {snippet}\n")

    return "\n".join(lines)


# ==============================================================================
# 2. Stack Overflow (Stack Exchange API v2.3)
# ==============================================================================
# Custom filter built via Stack Exchange API filter methods or standard `withbody` filter.
# `!nNPvSNdWme` / `withbody` includes question and answer `.body_markdown` / `.body`.
STACKEXCHANGE_API_BASE = "https://api.stackexchange.com/2.3"


@mcp.tool()
async def search_stackoverflow(
    query: str,
    tagged: str | None = None,
    sort: Literal["relevance", "votes", "creation", "activity"] = "relevance",
    num_results: int = 5,
    include_top_answers: bool = True,
) -> str:
    """
    Search Stack Overflow questions and optionally retrieve their accepted/top-voted answers.
    Works without an API key (300 requests/day free); set `STACKEXCHANGE_KEY` in `.env` for 10,000/day.

    Args:
        query: Search query (error message, concept, or keywords).
        tagged: Optional semicolon-separated tags (e.g., "python;asyncio" or "typescript").
        sort: Sort order ("relevance", "votes", "creation", "activity"). Default "relevance".
        num_results: Number of questions to return (1 to 10, default 5).
        include_top_answers: If True, fetches the accepted or highest-voted answer body for the returned questions.
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
        title = q.get("title", "Untitled")
        link = q.get("link", f"https://stackoverflow.com/q/{qid}")
        score = q.get("score", 0)
        ans_count = q.get("answer_count", 0)
        is_answered = q.get("is_answered", False)
        tags = ", ".join(q.get("tags", []))
        q_body = _truncate(q.get("body", ""), 600)

        status = "Answered" if is_answered else "Unanswered"
        lines.append(
            f"#### {idx}. [{title}]({link})\n"
            f"- **Question ID**: `{qid}` | **Score**: {score} | **Answers**: {ans_count} ({status}) | **Tags**: `{tags}`\n"
            f"- **Question Excerpt**:\n{q_body}\n"
        )

        q_answers = answers_by_question.get(qid, [])
        if q_answers:
            # Prefer accepted answer first, then highest score
            q_answers.sort(key=lambda a: (bool(a.get("is_accepted")), a.get("score", 0)), reverse=True)
            best = q_answers[0]
            acc_badge = " (Accepted Answer)" if best.get("is_accepted") else ""
            a_score = best.get("score", 0)
            a_body = _truncate(best.get("body", ""), 1500)
            lines.append(
                f"- **Top Answer{acc_badge} (Score: {a_score})**:\n{a_body}\n"
            )

    return "\n".join(lines)


@mcp.tool()
async def get_stackoverflow_question(question_id: int, max_answers: int = 3) -> str:
    """
    Fetch full details and top answers for a specific Stack Overflow question by its numeric ID.
    Works without an API key; uses `STACKEXCHANGE_KEY` from `.env` if available.

    Args:
        question_id: The numeric Stack Overflow question ID.
        max_answers: Maximum number of top/accepted answers to include (default 3).
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

    title = q.get("title", "Untitled")
    link = q.get("link", f"https://stackoverflow.com/q/{question_id}")
    score = q.get("score", 0)
    tags = ", ".join(q.get("tags", []))
    body = _truncate(q.get("body", ""), 3000)

    lines = [
        f"### [{title}]({link})",
        f"**Score**: {score} | **Tags**: `{tags}`\n",
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
            a_body = _truncate(ans.get("body", ""), 3000)
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
) -> str:
    """
    Search GitHub for repositories, issues/PRs, or code.
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
            body = _truncate(issue.get("body") or "", 500)
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
            snippet = _truncate("\n---\n".join(fragments), 800)
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
) -> str:
    """
    Fetch the raw text content of a file from a public GitHub repository (e.g., a README or source file).
    Works without an API key; uses `GITHUB_TOKEN` from `.env` if available.

    Args:
        repo: Repository in "owner/name" format (e.g., "modelcontextprotocol/python-sdk").
        path: File path within the repository (e.g., "README.md" or "src/mcp/server/fastmcp/server.py").
        ref: Optional branch, tag, or commit SHA (defaults to the repository's default branch).
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

    return f"### `{repo}` — `{path}`\n\n```\n{_truncate(content, 8000)}\n```"


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
) -> str:
    """
    Search Reddit posts across all of Reddit or within a specific subreddit.
    Works without an API key via Reddit's public JSON endpoints; optionally set
    `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` in `.env` for higher OAuth rate limits.

    Args:
        query: Search query string.
        subreddit: Optional subreddit name to restrict search to (e.g., "python" or "LocalLLaMA", without "r/").
        sort: Sort order ("relevance", "hot", "top", "new", "comments"). Default "relevance".
        time_filter: Time window for results ("all", "year", "month", "week", "day", "hour"). Default "all".
        num_results: Number of posts to return (1 to 15, default 5).
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
        selftext = _truncate(post.get("selftext", ""), 600)

        meta = (
            f"- **ID**: `{post_id}` | **Subreddit**: `{sub_prefixed}` | "
            f"**Score**: {score} | **Comments**: {num_comments} | **Author**: `u/{author}`"
        )
        entry = [f"#### {idx}. [{title}]({link})", meta]
        if not is_self and external_url and external_url != link:
            entry.append(f"- **External Link**: {external_url}")
        if selftext:
            entry.append(f"- **Excerpt**:\n{selftext}")
        lines.append("\n".join(entry) + "\n")

    return "\n".join(lines)


@mcp.tool()
async def get_reddit_post(
    post_id_or_url: str,
    max_comments: int = 5,
) -> str:
    """
    Fetch the full text and top-voted comments of a Reddit post by its post ID or URL.
    Works without an API key; uses `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` from `.env` if available.

    Args:
        post_id_or_url: Reddit post ID (e.g., "1a2b3c4" or "t3_1a2b3c4") or full Reddit post URL.
        max_comments: Maximum number of top comments to include (1 to 15, default 5).
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

    params: dict[str, str | int] = {
        "sort": "top",
        "limit": max_comments,
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
    selftext = _truncate(post.get("selftext", ""), 3000)
    external_url = post.get("url", "")
    is_self = post.get("is_self", True)

    lines = [
        f"### [{title}]({link})",
        f"**Subreddit**: `{sub_prefixed}` | **Score**: {score} | **Comments**: {num_comments} | **Author**: `u/{author}`\n",
    ]
    if not is_self and external_url and external_url != link:
        lines.append(f"**Link**: {external_url}\n")
    if selftext:
        lines.append(f"#### Post Body\n{selftext}\n")

    lines.append("#### Top Comments")
    comment_children = data[1].get("data", {}).get("children", []) if len(data) > 1 else []
    comments = [c.get("data", {}) for c in comment_children if c.get("kind") == "t1" and c.get("data", {}).get("body")]

    if not comments:
        lines.append("_No comments found._")
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
