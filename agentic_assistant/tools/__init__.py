"""Tool implementations available to the executor agent.

Every tool exposes a single `run(input_str: str) -> str` function so the
executor can dispatch on a tool name string without importing each
tool's internals directly.
"""

from agentic_assistant.tools import code_exec, file_ops, github_tool, web_search

TOOL_REGISTRY = {
    "web_search": web_search.run,
    "code_exec": code_exec.run,
    "file_ops": file_ops.run,
    "github": github_tool.run,
}


def run_tool(name: str, tool_input: str) -> str:
    """Dispatch to the named tool, returning a stringified result.

    Unknown tool names or runtime errors are captured and returned as
    text (never raised) so a single bad tool call can't crash the graph
    — the critic node gets to see the failure and can replan instead.
    """
    if name in ("none", "", None):
        return "No tool executed for this step."
    tool_fn = TOOL_REGISTRY.get(name)
    if tool_fn is None:
        return f"Error: unknown tool '{name}'. Available: {list(TOOL_REGISTRY)}"
    try:
        return tool_fn(tool_input)
    except Exception as exc:  # noqa: BLE001 - tool failures must surface, not crash the graph
        return f"Tool '{name}' raised an error: {exc}"
