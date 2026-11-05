"""Compare eval runs side by side, from their results files. Makes no API calls.

Run from the project root, oldest run first:

    python -m evals.diff_runs evals/results/<run_a>.jsonl evals/results/<run_b>.jsonl

Any number of runs can be compared, but they must use the same questions. They
can have different repeat counts, so compare the per-attempt numbers, not totals.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def load_run(path: Path) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty (a run saves its first line after its first question)")
    return rows[0], [r for r in rows if r["type"] == "result"]


def short_model(name: str) -> str:
    return name.split("/")[-1]


def models_used(meta: dict, records: list[dict]) -> str:
    # Older results files have no "models" field, only the configured model in the meta line
    used = sorted({m for r in records for m in r.get("models", [])}) or [meta["model"]]
    return " + ".join(short_model(m) for m in used)


def tokens(record: dict) -> int:
    return record["input_tokens"] + record["output_tokens"]


def summary_rows(meta: dict, records: list[dict]) -> dict[str, str]:
    n = len(records)
    passed = sum(r["outcome"] == "pass" for r in records)
    failures = Counter(r["outcome"] for r in records if r["outcome"] != "pass")
    total_tokens = sum(tokens(r) for r in records)
    expected = meta["repeats"] * len(meta["question_ids"])
    average = lambda value: f"{value / n:,.1f}" if n else "-"
    return {
        "Models used": models_used(meta, records),
        "Prompt": meta["prompt_hash"],
        "Attempts": f"{n}/{expected}" + ("" if n == expected else " (incomplete)"),
        "Passed": f"{passed}/{n} ({passed / n:.0%})" if n else "-",
        "Failures": ", ".join(f"{count} {kind}" for kind, count in failures.most_common()) or "none",
        "Tokens, total": f"{total_tokens:,}",
        "Tokens per attempt": f"{total_tokens / n:,.0f}" if n else "-",
        # Failed attempts can stop early and look cheap; this charges their tokens to the passes
        "Tokens per pass": f"{total_tokens / passed:,.0f}" if passed else "-",
        "LLM calls per attempt": average(sum(r["llm_calls"] for r in records)),
        "Tool calls per attempt": average(sum(r["tool_calls"] for r in records)),
        "Median seconds (with waits)": f"{median(r['latency_s'] for r in records):.1f}" if n else "-",
    }


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, *rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


def main():
    parser = argparse.ArgumentParser(description="Compare eval runs side by side.")
    parser.add_argument("files", nargs="+", type=Path, help="Results files, oldest first")
    args = parser.parse_args()
    if len(args.files) < 2:
        parser.error("give at least two results files")

    runs = [load_run(path) for path in args.files]
    question_ids = runs[0][0]["question_ids"]
    for path, (meta, _) in zip(args.files, runs):
        if set(meta["question_ids"]) != set(question_ids):
            raise SystemExit(f"Can't compare: {path.name} uses different questions from {args.files[0].name}")

    headers = [""] + [path.stem for path in args.files]
    summaries = [summary_rows(meta, records) for meta, records in runs]
    print_table(headers, [[label] + [s[label] for s in summaries] for label in summaries[0]])

    # One row per question: how often it passed and what it cost
    per_question = []
    for qid in question_ids:
        row = [qid]
        for _, records in runs:
            attempts = [r for r in records if r["id"] == qid]
            if attempts:
                passed = sum(r["outcome"] == "pass" for r in attempts)
                avg_tokens = sum(tokens(r) for r in attempts) / len(attempts)
                avg_tools = sum(r["tool_calls"] for r in attempts) / len(attempts)
                row.append(f"{passed}/{len(attempts)}  {avg_tokens:>6,.0f} tok  {avg_tools:.1f} tools")
            else:
                row.append("not run yet")
        per_question.append(row)
    print()
    print_table(["id"] + [path.stem for path in args.files], per_question)

    failed = defaultdict(list)
    for path, (_, records) in zip(args.files, runs):
        for r in records:
            if r["outcome"] != "pass":
                failed[path.stem].append(f"{r.get('run_id', r['id'])} {r['outcome']}")
    if failed:
        print("\nFailed attempts:")
        for name, attempts in failed.items():
            print(f"  {name}: {', '.join(attempts)}")


if __name__ == "__main__":
    main()
