"""Wires the planner / executor / critic nodes into a LangGraph graph.

Flow:
    plan -> execute (loop until plan exhausted) -> critic -> (done? END : plan again)

The executor loops over its own steps via a conditional edge rather
than the critic doing it, so the critic only ever sees a *complete*
plan's results — it reflects on outcomes, not on individual tool calls.
"""

from langgraph.graph import END, StateGraph

from agentic_assistant.agents.critic import critic_node
from agentic_assistant.agents.executor import execute_node
from agentic_assistant.agents.planner import plan_node
from agentic_assistant.state import AgentState


def _has_more_steps(state: AgentState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "execute"
    return "critic"


def _route_after_critic(state: AgentState) -> str:
    return "plan" if state["needs_replan"] else "end"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", _has_more_steps, {"execute": "execute", "critic": "critic"})
    graph.add_conditional_edges("execute", _has_more_steps, {"execute": "execute", "critic": "critic"})
    graph.add_conditional_edges("critic", _route_after_critic, {"plan": "plan", "end": END})

    return graph.compile()


def run_task(task: str) -> AgentState:
    """Convenience entry point: runs a task to completion and returns final state."""
    app = build_graph()
    initial_state: AgentState = {
        "task": task,
        "messages": [],
        "plan": [],
        "current_step": 0,
        "iterations": 0,
        "final_answer": "",
        "needs_replan": False,
        "critique": "",
    }
    return app.invoke(initial_state)
