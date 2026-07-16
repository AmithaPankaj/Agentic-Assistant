"""Smoke tests that don't require GROQ_API_KEY or network access.

These check the parts of the system that are pure logic (tools, graph
wiring, plan parsing) so CI/graders can verify the project works
without needing secrets.
"""

import json

from agentic_assistant.agents.planner import _parse_plan
from agentic_assistant.graph import build_graph
from agentic_assistant.tools import code_exec, file_ops


def test_code_exec_runs_and_captures_stdout():
    result = code_exec.run("print(2 + 2)")
    assert "stdout:" in result
    assert "4" in result


def test_code_exec_captures_errors_without_crashing():
    result = code_exec.run("raise ValueError('boom')")
    assert "stderr:" in result
    assert "ValueError" in result


def test_code_exec_timeout_is_enforced():
    result = code_exec.run("import time; time.sleep(60)")
    assert "timeout" in result.lower()


def test_file_ops_write_then_read_roundtrip():
    write_result = file_ops.run("write|smoke_test.txt|hello agent")
    assert "Wrote" in write_result
    read_result = file_ops.run("read|smoke_test.txt")
    assert read_result == "hello agent"


def test_file_ops_blocks_path_escape():
    result = file_ops.run("read|../../etc/passwd")
    assert "escapes the sandboxed workspace" in result or "Error" in result


def test_plan_parser_handles_clean_json():
    raw = json.dumps(
        {"steps": [{"id": 1, "description": "search", "tool": "web_search", "tool_input": "test query"}]}
    )
    steps = _parse_plan(raw)
    assert len(steps) == 1
    assert steps[0]["tool"] == "web_search"
    assert steps[0]["status"] == "pending"


def test_plan_parser_strips_markdown_fences():
    raw = "```json\n" + json.dumps({"steps": [{"id": 1, "description": "x", "tool": "none", "tool_input": ""}]}) + "\n```"
    steps = _parse_plan(raw)
    assert len(steps) == 1


def test_graph_compiles():
    app = build_graph()
    assert app is not None
