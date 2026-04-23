"""
Script to build a 50-sample HotpotQA dataset in the QAExample schema format.

Usage:
    python scripts/build_hotpot_dataset.py [--out data/hotpot_100.json] [--n 50]

Requires:
    pip install datasets

The script pulls from the HotpotQA 'distractor' validation split on HuggingFace,
selects `n` diverse samples (mixed difficulty), and writes them as a JSON array
matching the QAExample schema:
  {qid, difficulty, question, gold_answer, context: [{title, text}]}

Why 50 samples?  50 QA × 2 agents (ReAct + Reflexion) = 100 RunRecords,
which satisfies autograde.py's requirement: num_records >= 100.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import typer
from rich import print

app = typer.Typer(add_completion=False)

# Difficulty mapping based on HotpotQA 'level' field
_LEVEL_MAP = {"easy": "easy", "medium": "medium", "hard": "hard"}


def _build_example(idx: int, item: dict) -> dict:
    """Convert a HotpotQA item to the QAExample dict format."""
    # context is list of [title, sentences_list]
    context_chunks = [
        {"title": title, "text": " ".join(sentences)}
        for title, sentences in item["context"]["sentences"]
        # zip with titles
    ]
    # HotpotQA structure: context = {"title": [...], "sentences": [[...]]}
    titles = item["context"]["title"]
    sentences = item["context"]["sentences"]
    context_chunks = [
        {"title": titles[i], "text": " ".join(sentences[i])}
        for i in range(len(titles))
    ]

    difficulty = _LEVEL_MAP.get(item.get("level", "medium"), "medium")
    return {
        "qid": f"hpq{idx:04d}",
        "difficulty": difficulty,
        "question": item["question"],
        "gold_answer": item["answer"],
        "context": context_chunks,
    }


@app.command()
def main(
    out: str = "data/hotpot_100.json",
    n: int = 50,
    seed: int = 42,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("[red]Error:[/red] 'datasets' package not found. Run: pip install datasets")
        raise typer.Exit(1)

    print(f"[cyan]Loading HotpotQA (distractor / validation)...[/cyan]")
    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)

    random.seed(seed)
    total = len(ds)
    indices = random.sample(range(total), min(n, total))

    examples = []
    for rank, idx in enumerate(indices):
        item = ds[idx]
        try:
            ex = _build_example(rank, item)
            examples.append(ex)
        except Exception as e:
            print(f"[yellow]Skipping item {idx}: {e}[/yellow]")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[green]Saved {len(examples)} samples → {out_path}[/green]")
    print(f"Expected RunRecords after benchmark: {len(examples) * 2} (≥ 100 required)")


if __name__ == "__main__":
    app()
