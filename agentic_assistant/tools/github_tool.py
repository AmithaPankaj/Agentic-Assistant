"""Minimal GitHub REST API tool (read-mostly, one safe write action).

Input format (pipe-delimited):
    "search_repos|langgraph agents"
    "repo_info|owner/repo"
    "list_issues|owner/repo"
    "create_issue|owner/repo|Title text|Body text"

Requires GITHUB_TOKEN in the environment for anything beyond public
search (GitHub's anonymous rate limit is very low). `create_issue` is
included deliberately as the *only* write action, to demonstrate safe
tool-scoping: an agent tool should expose the smallest useful surface,
not the entire API.
"""

import re

import requests

from agentic_assistant.config import settings

API_ROOT = "https://api.github.com"
_REPO_SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


def _headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


def _clean_repo_slug(raw: str) -> str:
    """Extract a bare 'owner/repo' from text that may carry extra words,
    punctuation, or star counts (e.g. 'TauricResearch/TradingAgents.' or
    'TauricResearch/TradingAgents (93,230 stars)'). The LLM resolving this
    input is working from a results paragraph, not a clean value, so this
    tool must not assume it received exactly 'owner/repo' with nothing else.
    """
    raw = raw.strip().strip("`").strip('"').strip("'")
    match = _REPO_SLUG_RE.search(raw)
    if not match:
        return raw
    return match.group(0).rstrip(".,;:")


def run(instruction: str) -> str:
    parts = instruction.split("|", maxsplit=3)
    action = parts[0].strip().lower() if parts else ""

    if action == "search_repos":
        query = parts[1].strip() if len(parts) > 1 else ""
        # GitHub search qualifiers are space-separated; a literal '+' (old-style
        # AND syntax) gets percent-encoded by requests into a literal plus
        # character instead of being treated as a separator, silently zeroing
        # out results. Normalize defensively since the query text originates
        # from an LLM that may not know this.
        query = query.replace("+", " ")
        resp = requests.get(
            f"{API_ROOT}/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 5},
            headers=_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            return f"No repositories found for '{query}'"
        lines = [f"Top repos for '{query}':"]
        for r in items:
            lines.append(f"- {r['full_name']} ({r['stargazers_count']} stars): {r.get('description', '')}")
        return "\n".join(lines)

    if action == "repo_info":
        repo = _clean_repo_slug(parts[1]) if len(parts) > 1 else ""
        resp = requests.get(f"{API_ROOT}/repos/{repo}", headers=_headers(), timeout=10)
        resp.raise_for_status()
        d = resp.json()
        return (
            f"{d['full_name']}: {d.get('description', '')}\n"
            f"Stars: {d['stargazers_count']}, Forks: {d['forks_count']}, "
            f"Open issues: {d['open_issues_count']}, Language: {d.get('language')}"
        )

    if action == "list_issues":
        repo = _clean_repo_slug(parts[1]) if len(parts) > 1 else ""
        resp = requests.get(f"{API_ROOT}/repos/{repo}/issues", headers=_headers(), timeout=10)
        resp.raise_for_status()
        issues = resp.json()
        if not issues:
            return f"No open issues on {repo}"
        lines = [f"Open issues on {repo}:"]
        for i in issues[:10]:
            lines.append(f"#{i['number']} {i['title']}")
        return "\n".join(lines)

    if action == "create_issue":
        if len(parts) < 3:
            return "Error: 'create_issue' requires 'create_issue|owner/repo|Title|Body'"
        repo, title = _clean_repo_slug(parts[1]), parts[2].strip()
        body = parts[3].strip() if len(parts) > 3 else ""
        if not settings.github_token:
            return "Error: GITHUB_TOKEN not set; cannot create issues."
        resp = requests.post(
            f"{API_ROOT}/repos/{repo}/issues",
            json={"title": title, "body": body},
            headers=_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return f"Created issue #{resp.json()['number']} on {repo}"

    return f"Error: unknown github action '{action}'. Use search_repos|repo_info|list_issues|create_issue."