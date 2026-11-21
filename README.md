# Search MCP Server (`search-mcp`)

A containerized Python [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that gives AI agents direct search access to **Google**, **Stack Overflow**, **GitHub**, and **Reddit**—combining HTML boilerplate stripping with **`ripgrep` (`rg`)** context extraction so only relevant matching lines and code blocks enter the agent's context window.

## How Context Window Protection Works

Instead of dumping full web pages, raw HTML bodies, or blind top-of-file truncations into the LLM context:
1. **HTML & Boilerplate Stripping**: Strips `<script>`, `<style>`, `<nav>`, `<footer>`, and raw HTML tags from Stack Overflow posts and fetched web pages while preserving fenced Markdown code blocks.
2. **Smart Query Parsing**: Strips search qualifiers (`site:`, `repo:`, `language:`, `is:`), extracts quoted phrases and technical identifiers, filters out stopwords, and builds an alternation regex pattern for `ripgrep`.
3. **Ripgrep Context Slicing & Ranking**: Pipes fetched content through `rg --ignore-case --context=<N>`, scores matching context blocks by distinct keyword hits and code density, and returns only the highest-signal blocks in document order.

## Features & API Key Requirements

| Platform | Tool(s) | Key Required? | Free Tier & Details |
| :--- | :--- | :--- | :--- |
| **Stack Overflow** | `search_stackoverflow`, `get_stackoverflow_question` | **No** (Optional) | **300 requests/day** without a key. Add `STACKEXCHANGE_KEY` in `.env` for **10,000 requests/day**. |
| **GitHub** | `search_github`, `get_github_file` | **No** for repos, issues & files (Optional for code search) | **10 search requests/min** without a key. Add `GITHUB_TOKEN` in `.env` for **30 requests/min** and to enable `search_type="code"`. |
| **Reddit** | `search_reddit`, `get_reddit_post` | **No** (Optional) | Works without a key via Reddit's public JSON endpoints. Add `REDDIT_CLIENT_ID` & `REDDIT_CLIENT_SECRET` in `.env` for **100 requests/min** via OAuth2. |
| **Google** | `search_google` | **Yes** | **100 queries/day** free via Official Google Custom Search JSON API (`GOOGLE_API_KEY` + `GOOGLE_CSE_ID`). |

## Quick Start (Docker — Recommended)

### 1. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` to add any keys you want to use (`.env` is gitignored).

### 2. Build the Container Image (Includes `ripgrep`)

```bash
docker build -t search-mcp .
```

### 3. Add to Your MCP Client Configuration (`stdio`)

Configure your MCP client (Claude Desktop, Cursor, or Antigravity `mcp_config.json`) to run the container over `stdio`:

```json
{
  "mcpServers": {
    "search-mcp": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "--env-file",
        "/Users/null/Desktop/search-mcp/.env",
        "search-mcp"
      ]
    }
  }
}
```

## Quick Start (Local with `mise` or Virtual Environment)

With [`mise`](https://mise.jdx.dev) (automatically manages `python`, `ripgrep`, `.venv`, and `.env`):

```bash
mise install
mise run install
mise run run        # Or `mise run docker:build` / `mise run docker:run`
```

Or manually with `venv` (ensure `ripgrep` (`rg`) is installed locally via `brew install ripgrep` or `apt install ripgrep`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Local MCP client configuration:

```json
{
  "mcpServers": {
    "search-mcp": {
      "command": "/Users/null/Desktop/search-mcp/.venv/bin/python",
      "args": ["/Users/null/Desktop/search-mcp/server.py"]
    }
  }
}
```

## Available Tools

- **`search_google(query, num_results=5, site_search=None, fetch_and_extract=True, filter_pattern=None, context_lines=2)`** — Search the web via Google Custom Search JSON API, fetch top result pages concurrently, strip HTML boilerplate, and extract matching lines via `ripgrep`.
- **`search_stackoverflow(query, tagged=None, sort="relevance", num_results=5, include_top_answers=True, filter_pattern=None, context_lines=3)`** — Search Stack Overflow questions and top/accepted answers, converting HTML to clean Markdown and extracting relevant lines/code blocks with `ripgrep`.
- **`get_stackoverflow_question(question_id, max_answers=3, query=None, context_lines=4)`** — Retrieve a Stack Overflow question and top answers by ID; pass `query` to filter long question and answer bodies with `ripgrep`.
- **`search_github(query, search_type="repositories", sort=None, num_results=5, filter_pattern=None, context_lines=3)`** — Search GitHub `repositories`, `issues` (including PRs), or `code`, extracting relevant sections from issue bodies and code matches via `ripgrep`.
- **`get_github_file(repo, path, ref=None, query=None, context_lines=5, max_chars=6000)`** — Fetch a file from any public GitHub repository; pass `query` to run `ripgrep` with line numbers (`rg -n -C`) and return only matching lines and surrounding context.
- **`search_reddit(query, subreddit=None, sort="relevance", time_filter="all", num_results=5, filter_pattern=None, context_lines=2)`** — Search Reddit posts and extract matching excerpts from post bodies via `ripgrep`.
- **`get_reddit_post(post_id_or_url, max_comments=5, query=None, context_lines=3)`** — Fetch a Reddit post and top comments by ID or URL; pass `query` to filter both the post body and the comment thread using `ripgrep`.

## License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](LICENSE).
