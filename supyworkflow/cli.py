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
@click.option("--user-id", envvar="SUPYAGENT_USER_ID", default="default", help="User ID")
@click.option("--base-url", envvar="SUPYAGENT_BASE_URL", default="https://app.supyagent.com")
@click.option("--input", "-i", "inputs", multiple=True, help="Input as key=value pairs")
@click.option("--from-cell", default=0, help="Cell index to start from")
@click.option("--dry-run", is_flag=True, help="Analyze without executing")
@click.option("--output-format", type=click.Choice(["text", "json"]), default="text")
@click.option("--tools-gateway-url", envvar="SUPYWORKFLOW_TOOLS_GATEWAY_URL", default=None,
              help="URL of a tools gateway. When set, ALL tool calls route through this gateway "
                   "instead of directly to supyagent.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def run(
    script: str,
    api_key: str,
    user_id: str,
    base_url: str,
    inputs: tuple[str, ...],
    from_cell: int,
    dry_run: bool,
    output_format: str,
    tools_gateway_url: str | None,
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
        try:
            input_dict[key] = json.loads(value)
        except json.JSONDecodeError:
            input_dict[key] = value

    # Build providers if gateway URL is set
    providers = None
    if tools_gateway_url:
        from supyworkflow.providers.http_gateway import HttpGatewayToolProvider
        providers = [HttpGatewayToolProvider(
            gateway_url=tools_gateway_url,
            api_key=api_key,
            user_id=user_id,
        )]

    runtime = SupyWorkflow(
        api_key=api_key, user_id=user_id, base_url=base_url,
        providers=providers,
    )

    if dry_run:
        analysis = runtime.dry_run(source)
        click.echo(json.dumps(analysis, indent=2))
        return

    result = runtime.run(source, inputs=input_dict, from_cell=from_cell)

    if output_format == "json":
        _output_json(result)
    else:
        _output_text(result)

    if result.status == "failed":
        sys.exit(1)


@app.command()
@click.option("--prompt", required=True, help="What the workflow should do")
@click.option("--api-key", envvar="SUPYAGENT_API_KEY", required=True, help="Cardamon API key")
@click.option("--user-id", default=None, help="User ID for X-Account-Id scoping")
@click.option("--base-url", envvar="SUPYAGENT_BASE_URL", default="https://app.supyagent.com")
@click.option("--context", default=None, help="Additional context for the generator")
@click.option("--max-turns", default=20, help="Maximum agent exploration turns")
@click.option("--progress-file", default=None, help="File to write progress updates to (for polling)")
@click.option("--output-format", type=click.Choice(["text", "json"]), default="text")
@click.option("--tools-gateway-url", envvar="SUPYWORKFLOW_TOOLS_GATEWAY_URL", default=None,
              help="URL of a tools gateway. When set, tool discovery and execution route "
                   "through this gateway instead of directly to supyagent.")
@click.option("--job-id", default=None, help="External job ID (for state persistence). Auto-generated if omitted.")
@click.option("--state-dir", default=None, help="Directory for full state checkpoints (enables resume on restart).")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def generate(
    prompt: str,
    api_key: str,
    user_id: str | None,
    base_url: str,
    context: str | None,
    max_turns: int,
    progress_file: str | None,
    output_format: str,
    tools_gateway_url: str | None,
    job_id: str | None,
    state_dir: str | None,
    verbose: bool,
) -> None:
    """Generate a workflow script from a natural language prompt (agentic)."""
    import logging

    from supyworkflow.agent_generator import generate_workflow_agentic

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Build providers if gateway URL is set
    providers = None
    if tools_gateway_url:
        from supyworkflow.providers.http_gateway import HttpGatewayToolProvider
        providers = [HttpGatewayToolProvider(
            gateway_url=tools_gateway_url,
            api_key=api_key,
            user_id=user_id,
        )]

    session = generate_workflow_agentic(
        prompt=prompt,
        api_key=api_key,
        base_url=base_url,
        context=context,
        max_turns=max_turns,
        progress_file=progress_file,
        user_id=user_id,
        providers=providers,
        job_id=job_id,
        state_dir=state_dir,
    )

    if output_format == "json":
        output = {
            "script": session.script,
            "session_id": session.session_id,
            "turns": session.turns,
            "tool_calls_made": session.tool_calls_made,
            "total_tokens": session.total_tokens,
            "duration_ms": session.duration_ms,
        }
        click.echo(json.dumps(output, default=str))
    else:
        if session.script:
            click.echo(session.script)
        else:
            click.echo("No script generated.", err=True)
            sys.exit(1)

        click.echo(
            f"\n# Generated in {session.turns} turns, "
            f"{len(session.tool_calls_made)} tool calls, "
            f"{session.total_tokens} tokens, "
            f"{session.duration_ms:.0f}ms",
            err=True,
        )


@app.command()
@click.argument("script", type=click.Path(exists=True))
@click.option("--output-format", type=click.Choice(["text", "json"]), default="text")
def parse(script: str, output_format: str) -> None:
    """Parse a workflow script and show cell structure."""
    from supyworkflow.parser import build_dependency_graph, parse_cells

    with open(script) as f:
        source = f.read()

    cells = parse_cells(source)
    graph = build_dependency_graph(cells)

    if output_format == "json":
        # Detect tool calls and llm calls per cell
        import re
        tool_pattern = re.compile(r'\b(\w+_\w+)\s*\(')
        llm_pattern = re.compile(r'\bllm\s*\(')

        result = {
            "cells": [
                {
                    "index": c.index,
                    "label": c.label or f"cell_{c.index}",
                    "reads": sorted(c.reads),
                    "writes": sorted(c.writes),
                    "depends_on": sorted(graph.get(c.index, set())),
                    "tool_calls": [m for m in tool_pattern.findall(c.source) if '_' in m and m != '__build_class__'],
                    "has_llm_call": bool(llm_pattern.search(c.source)),
                    "source_lines": len(c.source.strip().splitlines()),
                }
                for c in cells
            ],
            "dependency_graph": {
                str(k): sorted(v) for k, v in graph.items()
            },
        }
        click.echo(json.dumps(result, indent=2))
    else:
        for cell in cells:
            deps = graph.get(cell.index, set())
            click.echo(f"Cell {cell.index}: {cell.label or '(unlabeled)'}")
            click.echo(f"  reads:  {sorted(cell.reads)}")
            click.echo(f"  writes: {sorted(cell.writes)}")
            if deps:
                click.echo(f"  depends on cells: {sorted(deps)}")
            click.echo()


def _output_json(result) -> None:
    """Output run result as JSON (for subprocess consumption)."""
    output = {
        "status": result.status,
        "run_id": result.trace.run_id if result.trace else None,
        "outputs": {
            k: v for k, v in result.outputs.items()
            if not callable(v) and not isinstance(v, type)
        },
        "cells": [
            {
                "index": c.index,
                "label": c.label,
                "status": c.status,
                "duration_ms": round(c.duration_ms, 1),
            }
            for c in result.cells
        ],
        "healed_cells": {
            str(idx): {
                "healed": h.healed,
                "attempts": h.attempts,
                "patched_source": h.patched_source if h.healed else None,
            }
            for idx, h in result.healed_cells.items()
        } if result.healed_cells else {},
        "trace": result.trace.to_dict() if result.trace else None,
        "error": str(result.error) if result.error else None,
    }
    click.echo(json.dumps(output, default=str))


def _output_text(result) -> None:
    """Output run result as human-readable text."""
    for cell in result.cells:
        status_icon = {"completed": "ok", "failed": "FAIL", "skipped": "skip"}.get(
            cell.status, "?"
        )
        label = cell.label or f"cell {cell.index}"
        click.echo(f"  [{status_icon}] {label} ({cell.duration_ms:.0f}ms)")
        if cell.error:
            click.echo(f"       error: {cell.error}")

    if result.trace:
        summary = result.trace.summary()
        click.echo(
            f"\n  {summary['cells']} cells, "
            f"{summary['tool_calls']} tool calls, "
            f"{summary['llm_calls']} llm calls, "
            f"${summary['total_cost']:.4f}, "
            f"{summary['total_duration_ms']:.0f}ms total"
        )
