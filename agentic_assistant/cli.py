"""Command-line entry point.

Usage:
    python -m agentic_assistant.cli run Find the 3 most starred Python LLM agent frameworks on GitHub

Quotes around the task are optional — any words after `run` are joined
into a single task string. This avoids shell-specific quoting issues
(e.g. PowerShell mangling straight quotes into smart quotes on paste).
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_assistant.graph import run_task

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def run(
    task_words: list[str] = typer.Argument(..., help="The task for the agent to complete (quotes optional)")
):
    """Run the agent end-to-end on TASK and print the plan, step results, and final answer."""
    task = " ".join(task_words)
    console.print(Panel(task, title="Task", style="bold cyan"))

    with console.status("[bold green]Agent thinking..."):
        result = run_task(task)

    table = Table(title="Execution Plan")
    table.add_column("#", justify="right")
    table.add_column("Step")
    table.add_column("Tool")
    table.add_column("Status")
    for step in result["plan"]:
        style = "green" if step["status"] == "done" else "red"
        table.add_row(str(step["id"]), step["description"], step["tool"], f"[{style}]{step['status']}[/{style}]")
    console.print(table)

    console.print(Panel(result["final_answer"], title="Final Answer", style="bold magenta"))


if __name__ == "__main__":
    app()