"""Planner node: turns the task (and any critique) into an ordered list
of steps, each tagged with the tool it needs.

The LLM is asked to return strict JSON so we can deterministically load
it into `Step` dicts. If parsing fails, we retry once with an explicit
"your last output was not valid JSON" correction — this is the single
biggest reliability lever for structured-output agents.
"""

import json

from agentic_assistant.llm import get_llm
from agentic_assistant.state import AgentState, Step

SYSTEM_PROMPT = """You are the planning module of an autonomous agent.
Break the user's task into a short ordered list of concrete steps (2-6 steps).
Each step must specify which single tool it needs:
  - "web_search": look something up online
  - "code_exec": ONLY for steps that genuinely require running Python (e.g. calculations,
    data transformations too complex to do in prose, testing a code snippet). Do NOT use
    code_exec to extract, restate, sort, or summarize information that a prior step's
    result already contains in text form — use "none" for that instead.
  - "file_ops": read, write, or list files in the workspace (format: "read|path", "write|path|content", "list|dir")
  - "github": query or act on GitHub (format: "search_repos|q", "repo_info|owner/repo", "list_issues|owner/repo", "create_issue|owner/repo|title|body")
  - "none": pure reasoning, extraction from already-known text, or writing — no tool call

Keep the plan minimal: don't add a step whose only job is to reformat or re-derive
something already produced by an earlier step's result.

Respond with ONLY valid JSON, no prose, no markdown fences, matching this schema:
{"steps": [{"id": 1, "description": "...", "tool": "web_search", "tool_input": "..."}]}
"""


def _parse_plan(raw: str) -> list[Step]:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(cleaned)
    steps: list[Step] = []
    for s in data["steps"]:
        steps.append(
            Step(
                id=int(s["id"]),
                description=str(s["description"]),
                tool=str(s.get("tool", "none")),
                tool_input=str(s.get("tool_input", "")),
                status="pending",
                result="",
            )
        )
    return steps


def plan_node(state: AgentState) -> dict:
    llm = get_llm()
    critique_note = f"\n\nPrevious attempt's critique to address: {state['critique']}" if state.get("critique") else ""
    user_prompt = f"Task: {state['task']}{critique_note}"

    raw = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    ).content

    try:
        steps = _parse_plan(raw)
    except (json.JSONDecodeError, KeyError, TypeError):
        # one retry with an explicit correction, matching how a human would nudge a bad response
        retry_raw = llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That was not valid JSON matching the schema. Reply with ONLY the JSON object."},
            ]
        ).content
        steps = _parse_plan(retry_raw)

    return {
        "plan": steps,
        "current_step": 0,
        "needs_replan": False,
        "critique": "",
        "messages": [{"role": "assistant", "content": f"Plan created with {len(steps)} steps."}],
    }