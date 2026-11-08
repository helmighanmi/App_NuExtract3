"""Command line interface: serve, extract, markdown, templates, health."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from nuextract_app.config import get_settings
from nuextract_app.core.client import build_client
from nuextract_app.core.pipeline import ExtractionPipeline, ExtractionResult
from nuextract_app.core.templates import (
    TemplateRegistry,
    convert_json_schema,
)

console = Console()
app = typer.Typer(
    help="NuExtract3 Studio - document extraction from the command line.",
    no_args_is_help=True,
    add_completion=False,
)
templates_app = typer.Typer(help="Inspect and generate extraction templates.", no_args_is_help=True)
app.add_typer(templates_app, name="templates")


def _pipeline() -> ExtractionPipeline:
    settings = get_settings()
    return ExtractionPipeline(
        settings, build_client(settings), TemplateRegistry(settings.templates_dir)
    )


async def _run_extract(
    data: bytes,
    filename: str,
    template: Any,
    instructions: str | None,
    thinking: bool,
    force_vision: bool,
) -> ExtractionResult:
    pipeline = _pipeline()
    try:
        return await pipeline.extract(
            data,
            filename,
            template=template,
            instructions=instructions,
            thinking=thinking,
            force_vision=force_vision,
        )
    finally:
        await pipeline.close()


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind address (default from settings)."),
    port: int = typer.Option(None, help="Port (default from settings)."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload for development."),
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    settings = get_settings()
    console.print(
        f"[bold]Starting API on http://{host or settings.api_host}:{port or settings.api_port}[/bold]"
    )
    uvicorn.run(
        "nuextract_app.api.main:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


@app.command()
def extract(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    template: str = typer.Option(
        None, "--template", "-t", help="Template name, .json path or inline JSON."
    ),
    schema: Path = typer.Option(
        None, "--schema", exists=True, dir_okay=False, help="JSON Schema to convert to a template."
    ),
    instructions: str = typer.Option(None, "--instructions", help="Extraction instructions."),
    thinking: bool = typer.Option(False, "--thinking", help="Enable reasoning mode."),
    force_vision: bool = typer.Option(
        False, "--force-vision", help="Send pages as images even when a text layer exists."
    ),
    output: Path = typer.Option(None, "--output", "-o", help="Write JSON result to a file."),
    include_raw: bool = typer.Option(False, "--raw", help="Include raw model output."),
) -> None:
    """Extract structured data from a PDF or image."""
    settings = get_settings()
    registry = TemplateRegistry(settings.templates_dir)

    try:
        if schema is not None:
            template_obj, dropped = convert_json_schema(json.loads(schema.read_text("utf-8")))
            for warning in dropped:
                console.print(f"[yellow]schema: {warning}[/yellow]")
        else:
            template_obj = registry.resolve(template or settings.default_template)
    except (ValueError, OSError) as exc:
        console.print(f"[red]Invalid template:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        result = asyncio.run(
            _run_extract(
                file.read_bytes(),
                file.name,
                template_obj,
                instructions,
                thinking,
                force_vision,
            )
        )
    except Exception as exc:
        console.print(f"[red]Extraction failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    payload = result.to_dict(include_raw=include_raw)
    if output:
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote {output}[/green]")
    else:
        console.print(JSON.from_data(payload))

    console.print(
        f"[dim]{result.page_count} page(s) - {result.prompt_tokens + result.completion_tokens} tokens "
        f"- {result.latency_s:.2f}s[/dim]"
    )
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


@app.command()
def markdown(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    thinking: bool = typer.Option(False, "--thinking", help="Enable reasoning mode."),
    output: Path = typer.Option(None, "--output", "-o", help="Write Markdown to a file."),
) -> None:
    """Convert a PDF or image to Markdown (OCR / RAG preprocessing)."""
    pipeline = _pipeline()

    async def run() -> Any:
        try:
            return await pipeline.markdown(file.read_bytes(), file.name, thinking=thinking)
        finally:
            await pipeline.close()

    try:
        result = asyncio.run(run())
    except Exception as exc:
        console.print(f"[red]Markdown conversion failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    if output:
        output.write_text(result.markdown, encoding="utf-8")
        console.print(f"[green]Wrote {output}[/green]")
    else:
        console.print(result.markdown)

    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


@templates_app.command("list")
def list_templates() -> None:
    """List bundled extraction templates."""
    settings = get_settings()
    registry = TemplateRegistry(settings.templates_dir)
    table = Table(title="NuExtract templates", show_lines=False)
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    table.add_column("Fields", justify="right")
    for info in registry.list_templates():
        fields = len(info.template) if isinstance(info.template, dict) else 1
        table.add_row(info.name, info.description or "-", str(fields))
    console.print(table)


@templates_app.command("show")
def show_template(name: str = typer.Argument(..., help="Template name.")) -> None:
    """Print a bundled template as JSON."""
    settings = get_settings()
    registry = TemplateRegistry(settings.templates_dir)
    try:
        info = registry.get_info(name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(JSON.from_data(info.template))


@templates_app.command("generate")
def generate_template(
    prompt: str = typer.Argument(..., help="Describe what you want to extract."),
    thinking: bool = typer.Option(False, "--thinking"),
) -> None:
    """Ask NuExtract3 to generate a template from a description."""
    pipeline = _pipeline()

    async def run() -> Any:
        try:
            return await pipeline.generate_template(prompt, thinking=thinking)
        finally:
            await pipeline.close()

    try:
        result = asyncio.run(run())
    except Exception as exc:
        console.print(f"[red]Template generation failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(JSON.from_data(result.template))


@app.command()
def health() -> None:
    """Check that the inference backend is reachable."""
    settings = get_settings()
    pipeline = _pipeline()

    async def run() -> bool:
        try:
            return await pipeline.health()
        finally:
            await pipeline.close()

    healthy = asyncio.run(run())
    if healthy:
        console.print(
            f"[green]OK[/green] {settings.provider} backend at {settings.openai_base_url} "
            f"serving {settings.model_name}"
        )
    else:
        console.print(f"[red]UNREACHABLE[/red] {settings.openai_base_url}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
