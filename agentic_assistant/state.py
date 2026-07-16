"""Shared state that flows through the LangGraph graph.

Every node reads from and writes to this state. Keeping it as a single
TypedDict (rather than passing ad-hoc args between nodes) is what lets
LangGraph checkpoint, resume, and visualize the graph.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class Step(TypedDict):
    """A single step in the agent's plan."""

    id: int
    description: str
    tool: str          # which tool to call: web_search | code_exec | file_ops | github | none
    tool_input: str
    status: str         # pending | done | failed
    result: str


class AgentState(TypedDict):
    """The full state carried across the graph."""

    task: str                                   # original user task
    messages: Annotated[list, add_messages]      # running conversation / scratchpad
    plan: list[Step]                             # ordered steps produced by the planner
    current_step: int                            # index into plan
    iterations: int                              # loop guard
    final_answer: str                            # populated once the task is complete
    needs_replan: bool                           # set by critic if plan is off track
    critique: str                                # critic's most recent feedback
