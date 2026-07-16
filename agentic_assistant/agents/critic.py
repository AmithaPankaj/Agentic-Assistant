"""Critic node: reviews completed-plan results, decides whether the
task is genuinely done or needs another planning pass, and — when
done — synthesizes the final answer.

This is what turns a linear pipeline into an *agent*: the loop can
detect its own failure (a failed tool step, an incomplete answer) and
route back to the planner instead of confidently returning garbage.
"""

import json

from agentic_assistant.config import settings
from agentic_assistant.llm import get_llm
from agentic_assistant.state import AgentState

SYSTEM_PROMPT = """You are the critic/reflection module of an autonomous agent.
Given the original task and the results of every executed step, decide:
1. Is the task fully and correctly accomplished? (true/false)
2. If not, what specifically is missing or wrong (used to replan)?
3. If yes, write the final answer to give the user, synthesizing the step results.

Respond with ONLY valid JSON:
{"done": true/false, "critique": "...", "final_answer": "..."}
"""


def critic_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.0)
    step_summary = "\n".join(
        f"Step {s['id']} [{s['status']}] {s['description']} -> {s['result']}" for s in state["plan"]
    )
    user_prompt = f"Task: {state['task']}\n\nStep results:\n{step_summary}"

    raw = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    ).content

    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        verdict = json.loads(cleaned)
    except json.JSONDecodeError:
        # fail safe: if the critic itself can't produce JSON, don't loop forever — finalize as-is
        return {
            "needs_replan": False,
            "final_answer": step_summary,
            "critique": "",
        }

    done = bool(verdict.get("done", True))
    out_of_iterations = state["iterations"] >= settings.max_iterations

    if done or out_of_iterations:
        return {
            "needs_replan": False,
            "final_answer": verdict.get("final_answer") or step_summary,
            "critique": "",
        }

    return {
        "needs_replan": True,
        "critique": verdict.get("critique", "Plan was incomplete."),
        "final_answer": "",
    }
