"""Web search tool backed by DuckDuckGo (no API key required).

Returns a compact, LLM-friendly digest of the top results rather than
raw HTML/JSON, since the executor feeds this straight back into a
prompt and token budget matters.
"""

from duckduckgo_search import DDGS


def run(query: str, max_results: int = 5) -> str:
    """Search the web and return a formatted list of title/snippet/url."""
    query = query.strip()
    if not query:
        return "Error: empty search query."

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return f"No results found for query: '{query}'"

    lines = [f"Search results for '{query}':"]
    for i, r in enumerate(results, start=1):
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()
        href = r.get("href", "").strip()
        lines.append(f"{i}. {title}\n   {body}\n   Source: {href}")
    return "\n".join(lines)
