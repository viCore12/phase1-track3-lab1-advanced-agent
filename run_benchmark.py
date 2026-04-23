"""
Main benchmark runner for the Reflexion Agent Lab.

Features:
    - Supports real LLM providers (NVIDIA NIM, Groq, Ollama, OpenAI).
    - Supports mock mode for offline testing.
    - Incremental saving: saves every record immediately to JSONL.
    - Resume capability: skips already processed questions.
    - Periodic report update: refreshes report.json/md every 10 samples.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Set

import typer
from rich import print

from src.reflexion_lab.schemas import RunRecord

app = typer.Typer(add_completion=False)


def _load_existing_qids(path: Path) -> Set[str]:
    """Load QIDs from an existing JSONL file to support resuming."""
    if not path.exists():
        return set()
    qids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    qids.add(data["qid"])
                except:
                    continue
    return qids


def _load_all_records(path: Path) -> list[RunRecord]:
    """Load all records from a JSONL file."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(RunRecord.model_validate_json(line))
    return records


def _append_record(path: Path, record: RunRecord) -> None:
    """Append a single record to a JSONL file immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


@app.command()
def main(
    dataset: str = "data/hotpot_100.json",
    out_dir: str = "outputs/real_run",
    reflexion_attempts: int = 3,
    mode: str = "real",
    batch_size: int = 10,
) -> None:
    """
    Run ReAct and Reflexion agents with incremental saving and resume.
    """
    out_path = Path(out_dir)
    react_file = out_path / "react_runs.jsonl"
    reflexion_file = out_path / "reflexion_runs.jsonl"

    # Dynamically import runtime based on mode
    if mode == "mock":
        print("[yellow]Running in MOCK mode (no API calls)[/yellow]")
        import src.reflexion_lab.agents as agents_mod
        import src.reflexion_lab.mock_runtime as mock_rt
        agents_mod.actor_answer = lambda ex, aid, at, rm: (mock_rt.actor_answer(ex, aid, at, rm), 320 + aid * 65, 160 + aid * 40)
        agents_mod.evaluator = mock_rt.evaluator
        agents_mod.reflector = mock_rt.reflector
        agents_mod.FAILURE_MODE_BY_QID = mock_rt.FAILURE_MODE_BY_QID
    else:
        print(f"[cyan]Running in REAL mode (incremental save, batch_size={batch_size})[/cyan]")

    from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
    from src.reflexion_lab.reporting import build_report, save_report
    from src.reflexion_lab.utils import load_dataset

    examples = load_dataset(dataset)
    print(f"Loaded [bold]{len(examples)}[/bold] examples from {dataset}")

    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)

    # 1. Run ReAct Agent
    print("[cyan]Running ReAct agent...[/cyan]")
    existing_react = _load_existing_qids(react_file)
    if existing_react:
        print(f"  Found {len(existing_react)} existing records. Resuming...")

    for i, example in enumerate(examples):
        if example.qid in existing_react:
            continue
        
        print(f"  ReAct [{i+1}/{len(examples)}] qid={example.qid} (processing...)")
        record = react.run(example)
        _append_record(react_file, record)
        
        # Periodic report update
        if (len(_load_existing_qids(react_file))) % batch_size == 0:
            all_rec = _load_all_records(react_file) + _load_all_records(reflexion_file)
            if all_rec:
                report = build_report(all_rec, dataset_name=Path(dataset).name, mode=mode)
                save_report(report, out_path)
                print(f"  [dim]Updated report after ReAct batch...[/dim]")

    # 2. Run Reflexion Agent
    print("[cyan]Running Reflexion agent...[/cyan]")
    existing_reflexion = _load_existing_qids(reflexion_file)
    if existing_reflexion:
        print(f"  Found {len(existing_reflexion)} existing records. Resuming...")

    for i, example in enumerate(examples):
        if example.qid in existing_reflexion:
            continue
        
        print(f"  Reflexion [{i+1}/{len(examples)}] qid={example.qid} (processing...)")
        record = reflexion.run(example)
        _append_record(reflexion_file, record)
        
        # Periodic report update
        if (len(_load_existing_qids(reflexion_file))) % batch_size == 0:
            all_rec = _load_all_records(react_file) + _load_all_records(reflexion_file)
            if all_rec:
                report = build_report(all_rec, dataset_name=Path(dataset).name, mode=mode)
                save_report(report, out_path)
                print(f"  [dim]Updated report after Reflexion batch...[/dim]")

    # Final Report
    all_records = _load_all_records(react_file) + _load_all_records(reflexion_file)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)

    print(f"\n[green]Benchmark Complete![/green]")
    print(f"Report saved to: {json_path}")
    print(f"\n[bold]Summary:[/bold]")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
