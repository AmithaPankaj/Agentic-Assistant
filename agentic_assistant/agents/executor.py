"""Executor node: resolves the current step's tool input against real
prior results, runs the tool, then asks the LLM to interpret the raw
output in the context of the step's goal.

Why "resolve" is a separate pass: the planner writes the *entire* plan
— including every step's tool_input — before any step has executed.
That means a later step (e.g. "get info for the repo found in step 1")
can only be written as a guess or placeholder at planning time, since
the real repo name doesn't exist yet. Resolving each tool_input against
the actual results of prior steps, right before that step runs, is what
turns a rigid pre-written plan into something that behaves correctly
across dependent, chained steps.
"""

from agentic_assistant.llm import get_llm
from agentic_assistant.state import AgentState
from agentic_assistant.tools import run_tool

TOOL_INPUT_FORMATS = {
    "web_search": "plain text search query, e.g. 'LLM agent framework python'",
    "code_exec": "raw Python source code to execute",
    "file_ops": "'read|path', 'write|path|content', or 'list|dir'",
    "github": (
        "'search_repos|query' (space-separated qualifiers, no '+' signs), "
        "'repo_info|owner/repo', 'list_issues|owner/repo', "
        "or 'create_issue|owner/repo|title|body'"
    ),
    "none": "(not used)",
}

RESOLVE_PROMPT = """You are the execution module of an autonomous agent.
You are about to run ONE tool call for ONE step of a plan. Your job is to produce
the exact, concrete input string for that tool call — grounded in the REAL results
of steps that already ran, not placeholders like {{owner}} or {{repo_name}}.

Rules:
- If a prior step's result already contains the concrete value you need (e.g. a real
  repository name, URL, or filename), use that exact value.
- Never invent placeholder brackets like {{owner1}} or <repo>. If you don't know a
  required concrete value, keep the input generic/broad rather than using a placeholder.
- Match the tool's required input format exactly.
- Respond with ONLY the resolved input string. No prose, no quotes, no explanation.
"""


def _resolve_tool_input(state: AgentState, step: dict) -> str:
    if step["tool"] == "none":
        return step["tool_input"]

    prior_results = "\n".join(
        f"Step {s['id']} ({s['tool']}) [{s['status']}]: {s['result']}"
        for s in state["plan"]
        if s["status"] in ("done", "failed") and s["result"]
    )
    llm = get_llm(temperature=0.0)
    resolved = llm.invoke(
        [
            {"role": "system", "content": RESOLVE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Overall task: {state['task']}\n"
                    f"Tool: {step['tool']}\n"
                    f"Required format: {TOOL_INPUT_FORMATS.get(step['tool'], 'free text')}\n"
                    f"Step goal: {step['description']}\n"
                    f"Planner's draft input (may contain placeholders — do not copy them verbatim): {step['tool_input']}\n\n"
                    f"Results from prior steps:\n{prior_results or '(none yet)'}\n\n"
                    "Resolved input string:"
                ),
            },
        ]
    ).content.strip()
    # guard against the model wrapping its answer in quotes or fences anyway
    return resolved.strip("`").strip('"').strip("'")


INTERPRET_PROMPT = """You are the execution module of an autonomous agent.
A tool was just run for one step of a plan. Summarize the outcome in 2-4 sentences,
focused on what matters for completing the overall task. Note clearly if the tool failed.
"""


def execute_node(state: AgentState) -> dict:
    plan = state["plan"]
    idx = state["current_step"]
    step = plan[idx]

    resolved_input = _resolve_tool_input(state, step)
    tool_output = run_tool(step["tool"], resolved_input)

    llm = get_llm()
    summary = llm.invoke(
        [
            {"role": "system", "content": INTERPRET_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Overall task: {state['task']}\n"
                    f"Step goal: {step['description']}\n"
                    f"Tool used: {step['tool']}\n"
                    f"Resolved tool input: {resolved_input}\n"
                    f"Raw tool output:\n{tool_output}"
                ),
            },
        ]
    ).content

    failed = "error" in tool_output.lower()[:200] or "raised an error" in tool_output.lower()
    updated_step = {
        **step,
        "tool_input": resolved_input,
        "status": "failed" if failed else "done",
        "result": summary,
    }
    new_plan = [updated_step if i == idx else s for i, s in enumerate(plan)]

    return {
        "plan": new_plan,
        "current_step": idx + 1,
        "iterations": state["iterations"] + 1,
        "messages": [
            {"role": "assistant", "content": f"Step {step['id']} ({step['tool']}): {summary}"}
        ],
    }