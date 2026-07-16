"""File read/write tool, sandboxed to a single workspace directory.

Input format (pipe-delimited so the LLM can produce it reliably):
    "read|path/to/file.txt"
    "write|path/to/file.txt|file contents here"
    "list|subdir"

All paths are resolved relative to `settings.workspace_dir` and any
attempt to escape it (e.g. via `../..`) is rejected — the whole point
of an agent tool is that its blast radius is bounded.
"""

from pathlib import Path

from agentic_assistant.config import settings

WORKSPACE = Path(settings.workspace_dir).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)


def _safe_path(relative: str) -> Path:
    candidate = (WORKSPACE / relative).resolve()
    if WORKSPACE not in candidate.parents and candidate != WORKSPACE:
        raise PermissionError(f"Path '{relative}' escapes the sandboxed workspace.")
    return candidate


def run(instruction: str) -> str:
    parts = instruction.split("|", maxsplit=2)
    action = parts[0].strip().lower() if parts else ""

    try:
        if action == "read":
            if len(parts) < 2:
                return "Error: 'read' requires a path, e.g. 'read|notes.txt'"
            path = _safe_path(parts[1].strip())
            if not path.exists():
                return f"Error: file '{parts[1].strip()}' does not exist."
            return path.read_text(encoding="utf-8")

        if action == "write":
            if len(parts) < 3:
                return "Error: 'write' requires 'write|path|content'"
            path = _safe_path(parts[1].strip())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(parts[2], encoding="utf-8")
            return f"Wrote {len(parts[2])} chars to {parts[1].strip()}"

        if action == "list":
            sub = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "."
            path = _safe_path(sub)
            if not path.exists():
                return f"Error: directory '{sub}' does not exist."
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
            return "\n".join(entries) if entries else "(empty directory)"
    except PermissionError as exc:
        return f"Error: {exc}"

    return f"Error: unknown file_ops action '{action}'. Use read|write|list."
