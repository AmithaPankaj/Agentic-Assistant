"""Sandboxed Python code execution tool.

Runs untrusted (LLM-generated) code as a subprocess rather than via
exec()/eval() in-process, with a hard timeout and no shell=True, so a
runaway or malicious snippet can't hang or compromise the host process.
This is a deliberate design choice worth calling out in interviews:
in-process eval() is a common (and dangerous) shortcut.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from agentic_assistant.config import settings


def run(code: str) -> str:
    """Execute a Python snippet in an isolated subprocess and return stdout/stderr."""
    code = code.strip()
    if not code:
        return "Error: no code provided."

    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "snippet.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=settings.code_exec_timeout,
                cwd=tmp_dir,
            )
        except subprocess.TimeoutExpired:
            return f"Error: execution exceeded {settings.code_exec_timeout}s timeout and was killed."

        output = []
        if proc.stdout:
            output.append(f"stdout:\n{proc.stdout.strip()}")
        if proc.stderr:
            output.append(f"stderr:\n{proc.stderr.strip()}")
        output.append(f"exit_code: {proc.returncode}")
        return "\n".join(output) if output else "Code executed with no output."
