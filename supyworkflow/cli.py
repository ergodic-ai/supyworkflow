"""CLI entry point for supyworkflow."""

from __future__ import annotations

import json
import sys

import click


@click.group()
@click.version_option()
def app() -> None:
    """supyworkflow — Workflow-as-code runtime."""


@app.command()
@click.argument("script", type=click.Path(exists=True))
@click.option("--api-key", envvar="SUPYAGENT_API_KEY", required=True, help="Cardamon API key")
@click.option("--user-id", envvar="SUPYAGENT_USER_ID", required=True, help="User ID")
@click.option("--base-url", envvar="SUPYAGENT_BASE_URL", default="https://app.supyagent.com")
@click.option("--input", "-i", "inputs", multiple=True, help="Input as key=value pairs")
@click.option("--from-cell", default=0, help="Cell index to start from")
@click.option("--dry-run", is_flag=True, help="Analyze without executing")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def run(
    script: str,
    api_key: str,
    user_id: str,
    base_url: str,
    inputs: tuple[str, ...],
    from_cell: int,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Execute a workflow script."""
    import logging

    from supyworkflow.runtime import SupyWorkflow

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    with open(script) as f:
        source = f.read()

    # Parse inputs
    input_dict = {}
    for item in inputs:
        key, _, value = item.partition("=")
        # Try to parse as JSON, fall back to string
        try:
            input_dict[key] = json.loads(value)
        except json.JSONDecodeError:
            input_dict[key] = value

    runtime = SupyWorkflow(api_key=api_key, user_id=user_id, base_url=base_url)

    if dry_run:
        analysis = runtime.dry_run(source)
        click.echo(json.dumps(analysis, indent=2))
        return

    result = runtime.run(source, inputs=input_dict, from_cell=from_cell)

    # Print cell results
    for cell in result.cells:
        status_icon = {"completed": "ok", "failed": "FAIL", "skipped": "skip"}.get(
            cell.status, "?"
        )
        label = cell.label or f"cell {cell.index}"
        click.echo(f"  [{status_icon}] {label} ({cell.duration_ms:.0f}ms)")
        if cell.error:
            click.echo(f"       error: {cell.error}")

    # Print summary
    if result.trace:
        summary = result.trace.summary()
        click.echo(
            f"\n  {summary['cells']} cells, "
            f"{summary['tool_calls']} tool calls, "
            f"{summary['llm_calls']} llm calls, "
            f"{summary['total_duration_ms']:.0f}ms total"
        )

    if result.status == "failed":
        sys.exit(1)


@app.command()
@click.argument("script", type=click.Path(exists=True))
def parse(script: str) -> None:
    """Parse a workflow script and show cell structure."""
    from supyworkflow.parser import build_dependency_graph, parse_cells

    with open(script) as f:
        source = f.read()

    cells = parse_cells(source)
    graph = build_dependency_graph(cells)

    for cell in cells:
        deps = graph.get(cell.index, set())
        click.echo(f"Cell {cell.index}: {cell.label or '(unlabeled)'}")
        click.echo(f"  reads:  {sorted(cell.reads)}")
        click.echo(f"  writes: {sorted(cell.writes)}")
        if deps:
            click.echo(f"  depends on cells: {sorted(deps)}")
        click.echo()
