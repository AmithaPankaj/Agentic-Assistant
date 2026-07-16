# Agentic Assistant — Autonomous Multi-Agent Task Executor

A LangGraph-based agent that takes a natural-language task, **plans** a
sequence of steps, **executes** them using real tools (web search, sandboxed
code execution, file I/O, GitHub API), and **critiques its own output** —
looping back to replan if the result is incomplete, instead of confidently
returning a wrong answer.

Built to demonstrate agentic system design beyond a single prompt-response
chatbot: explicit state, tool-calling, self-correction, and a bounded
execution loop.

## Architecture

```
        ┌────────┐        ┌─────────┐        ┌────────┐
 task → │  Plan  │ ─────→ │ Execute │ ─────→ │ Critic │
        └────────┘  loop  └─────────┘        └────────┘
             ▲          each step                 │
             │                                     │
             └─────────── needs_replan? ───────────┤
                                                     │
                                              done → END
```

- **Planner** — asks the LLM to break the task into 2–6 steps, each tagged
  with the exact tool it needs. Returns strict JSON, parsed into a typed
  `Step`; one automatic retry if the JSON is malformed.
- **Executor** — runs the tool for the current step, then asks the LLM to
  summarize the raw tool output in the context of the step's goal, before
  moving to the next step. Loops via a conditional edge until the plan is
  exhausted.
- **Critic** — reviews all step results against the original task. If
  something is missing or wrong, it writes a critique and the graph routes
  back to the planner (bounded by `AGENT_MAX_ITERATIONS`, default 6). If the
  task is complete, it synthesizes the final answer.

All state (task, plan, message history, iteration count) lives in a single
`AgentState` TypedDict passed through the graph — this is what lets
LangGraph checkpoint/resume/visualize the run.

## Tools

| Tool | What it does | Safety notes |
|---|---|---|
| `web_search` | DuckDuckGo search, no API key needed | — |
| `code_exec` | Runs LLM-generated Python | **Subprocess, not `exec()`** — isolated temp dir, hard timeout, no shell |
| `file_ops` | Read / write / list files | Sandboxed to `AGENT_WORKSPACE`; path-escape attempts are rejected |
| `github` | Search repos, get repo info, list issues, create an issue | Read-only by default; `create_issue` is the one write action, gated behind `GITHUB_TOKEN` |

Tool failures never crash the graph — `run_tool()` catches exceptions and
returns them as text, so the critic can see the failure and decide whether
to replan.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your GROQ_API_KEY (free at https://console.groq.com)
```

## Usage

```bash
python -m agentic_assistant.cli run "Find the 3 most-starred Python LLM agent frameworks on GitHub and write a one-paragraph comparison to workspace/comparison.txt"
```

This prints the generated plan as a table (with per-step tool + status),
then the final synthesized answer.

## Tests

Smoke tests cover everything that doesn't require a live API key —
sandboxing, JSON plan parsing, timeout enforcement, and graph compilation:

```bash
pip install pytest
PYTHONPATH=. pytest tests/ -v
```

## Design decisions worth calling out

- **Plan/execute/critic separation** rather than a single ReAct loop, so
  each concern (decomposition, tool use, self-evaluation) is independently
  testable and debuggable.
- **Subprocess-isolated code execution** instead of in-process `eval()` —
  LLM-generated code is untrusted by definition.
- **Bounded iteration count** so a stubborn critic can't loop forever; after
  the cap it finalizes with whatever it has rather than hanging.
- **Tool failures are data, not exceptions** — they flow back into the
  state so the agent can reason about and recover from them.

## Possible extensions

- Swap the DuckDuckGo tool for Tavily/SerpAPI for higher-quality search
- Add a `human_approval` node before `create_issue` or other write actions
- Persist `AgentState` via LangGraph's checkpointer for resumable runs
- Add a Streamlit/FastAPI front end to visualize the plan graph live
