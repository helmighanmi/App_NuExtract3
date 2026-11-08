#!/usr/bin/env python3
"""Run an extraction evaluation set against the API and report tree scores."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import score_prediction

console = Console()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return items


def call_extract(
    client: httpx.Client,
    item: dict[str, Any],
    dataset_dir: Path,
    default_template: str,
    thinking: bool,
) -> dict[str, Any]:
    file_path = (dataset_dir / item["file"]).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")

    template = item.get("template", default_template)
    data: dict[str, str] = {"thinking": str(thinking).lower()}
    if isinstance(template, (dict, list)):
        data["template"] = json.dumps(template)
    else:
        data["template_name"] = str(template)
    if item.get("instructions"):
        data["instructions"] = item["instructions"]

    with file_path.open("rb") as handle:
        response = client.post(
            "/v1/extract",
            files={"file": (file_path.name, handle, "application/octet-stream")},
            data=data,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    items = load_dataset(dataset_path)
    if not items:
        raise SystemExit("Dataset is empty")
    if args.limit:
        items = items[: args.limit]

    console.print(f"Evaluating [bold]{len(items)}[/bold] document(s) against {args.api_url}")

    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    tokens = 0
    field_scores: dict[str, list[float]] = {}

    with httpx.Client(base_url=args.api_url, timeout=600.0) as client:
        for index, item in enumerate(items, start=1):
            identifier = item.get("id", f"item-{index}")
            try:
                payload = call_extract(
                    client,
                    item,
                    dataset_path.parent,
                    args.template,
                    args.thinking,
                )
            except Exception as exc:
                console.print(f"[red]{identifier}: {exc}[/red]")
                records.append({"id": identifier, "ok": False, "error": str(exc)})
                continue

            score = score_prediction(payload.get("result"), item["ground_truth"])
            latency = float(payload.get("latency_s", 0.0))
            usage = payload.get("usage") or {}
            latencies.append(latency)
            tokens += int(usage.get("total_tokens", 0) or 0)
            for key, value in score.per_field.items():
                field_scores.setdefault(key, []).append(value)

            records.append(
                {
                    "id": identifier,
                    "ok": True,
                    "score": score.to_dict(),
                    "latency_s": latency,
                    "usage": usage,
                    "warnings": payload.get("warnings", []),
                    "prediction": payload.get("result"),
                    "ground_truth": item["ground_truth"],
                }
            )
            console.print(
                f"[{index}/{len(items)}] {identifier}: score={score.score:.3f} ({latency:.1f}s)"
            )

    successful = [record for record in records if record["ok"]]
    summary = {
        "documents": len(items),
        "successful": len(successful),
        "failed": len(items) - len(successful),
        "mean_score": round(statistics.fmean(record["score"]["score"] for record in successful), 4)
        if successful
        else 0.0,
        "median_latency_s": round(statistics.median(latencies), 3) if latencies else None,
        "total_tokens": tokens,
        "per_field": {
            key: round(statistics.fmean(values), 4) for key, values in sorted(field_scores.items())
        },
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_path),
        "api_url": args.api_url,
        "template": args.template,
        "thinking": args.thinking,
    }

    table = Table(title="Field scores")
    table.add_column("Field", style="cyan")
    table.add_column("Mean score", justify="right")
    for key, value in summary["per_field"].items():
        table.add_row(key, f"{value:.3f}")
    console.print(table)

    console.print(
        f"[bold]Mean score: {summary['mean_score']:.3f}[/bold] · "
        f"success: {summary['successful']}/{summary['documents']} · "
        f"median latency: {summary['median_latency_s']}s"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"eval-{timestamp}.json"
    report_path.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    console.print(f"Report: [green]{report_path}[/green]")
    return 0 if summary["failed"] == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="eval/datasets/example.jsonl")
    parser.add_argument("--api-url", default="http://localhost:8080")
    parser.add_argument("--template", default="invoice")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
