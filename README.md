# Search MCP Server (`search-mcp`)

A lightweight Python [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that gives AI agents direct search access to **Google**, **Stack Overflow**, **GitHub**, and **Reddit**—offloading web and code lookups from local compute.

## Features & API Key Requirements

| Platform | Tool(s) | Key Required? | Free Tier & Details |
| :--- | :--- | :--- | :--- |
| **Stack Overflow** | `search_stackoverflow`, `get_stackoverflow_question` | **No** (Optional) | **300 requests/day** without a key. Add `STACKEXCHANGE_KEY` in `.env` for **10,000 requests/day**. |
| **GitHub** | `search_github`, `get_github_file` | **No** for repos, issues & files (Optional for code search) | **10 search requests/min** without a key. Add `GITHUB_TOKEN` in `.env` for **30 requests/min** and to enable `search_type="code"`. |
| **Reddit** | `search_reddit`, `get_reddit_post` | **No** (Optional) | Works without a key via Reddit's public JSON endpoints. Add `REDDIT_CLIENT_ID` & `REDDIT_CLIENT_SECRET` in `.env` for **100 requests/min** via OAuth2. |
| **Google** | `search_google` | **Yes** | **100 queries/day** free via Official Google Custom Search JSON API (`GOOGLE_API_KEY` + `GOOGLE_CSE_ID`). |

## Quick Start

### 1. Create a Virtual Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env` (Optional for Stack Overflow & GitHub, Required for Google)

Copy the template `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to add any keys you want to use (`.env` is gitignored):

- **`GOOGLE_API_KEY` & `GOOGLE_CSE_ID`**:
  1. Get an API key from the [Google Custom Search JSON API page](https://developers.google.com/custom-search/v1/overview).
  2. Create a search engine at [Programmable Search Engine](https://programmablesearchengine.google.com/) (enable "Search the entire web") and copy its **Search engine ID (`cx`)**.
- **`GITHUB_TOKEN`** *(optional)*: Create a [GitHub Personal Access Token](https://github.com/settings/tokens) (no scopes needed for public search) to unlock code search and higher rate limits.
- **`STACKEXCHANGE_KEY`** *(optional)*: Register on [Stack Apps](https://stackapps.com/apps/oauth/register) to raise the daily quota from 300 to 10,000 requests.
- **`REDDIT_CLIENT_ID` & `REDDIT_CLIENT_SECRET`** *(optional)*: Create an app at [Reddit App Preferences](https://www.reddit.com/prefs/apps) to use OAuth2 (`oauth.reddit.com`) instead of unauthenticated public JSON endpoints.

### 3. Add to Your MCP Client Configuration

Add the server to your MCP client config (e.g., Claude Desktop, Cursor, or Antigravity `mcp_config.json`):

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

- **`search_google(query, num_results=5, site_search=None)`** — Search the web via Google Custom Search JSON API.
- **`search_stackoverflow(query, tagged=None, sort="relevance", num_results=5, include_top_answers=True)`** — Search Stack Overflow questions and automatically include accepted/top-voted answers.
- **`get_stackoverflow_question(question_id, max_answers=3)`** — Retrieve full question body and top answers by Stack Overflow question ID.
- **`search_github(query, search_type="repositories", sort=None, num_results=5)`** — Search GitHub `repositories`, `issues` (including PRs), or `code`.
- **`get_github_file(repo, path, ref=None)`** — Fetch raw file contents (such as a `README.md` or source file) from any public GitHub repository.
- **`search_reddit(query, subreddit=None, sort="relevance", time_filter="all", num_results=5)`** — Search Reddit posts across all of Reddit or within a specific subreddit.
- **`get_reddit_post(post_id_or_url, max_comments=5)`** — Fetch the full body and top-voted comments of a Reddit post by ID or URL.

## License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](LICENSE).
