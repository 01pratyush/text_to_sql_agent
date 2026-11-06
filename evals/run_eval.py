"""Run the eval dataset through the agent and score it by execution accuracy.

Run from the project root:

    python -m evals.run_eval                    # every question once
    python -m evals.run_eval --ids c01,t05      # only these questions
    python -m evals.run_eval --repeats 3        # each question 3 times

Each run is saved to evals/results/<timestamp>.jsonl: one "meta" line describing
the agent version, then one "result" line per question attempt.

Every log event of a question attempt is tagged with run_id "<id>-r<repeat>"
(e.g. c02-r1), which is also stored in its result line. The console shows
WARNING and above by default, so the progress lines stay readable; set
LOG_LEVEL=INFO to watch every LLM and tool call. logs/agent.jsonl always has everything.
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from evals.agent_runner import run_agent
from evals.compare import execution_match
from log_config import run_context, setup_logging
from main import LLM_CONFIG, RECURSION_LIMIT, SYSTEM_PROMPT
from tracing import setup_tracing

log = logging.getLogger("evals.run_eval")

DATASET = Path("evals") / "questions.jsonl"
RESULTS_DIR = Path("evals") / "results"


def load_questions(ids: list[str] | None) -> list[dict]:
    with open(DATASET, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    if ids:
        unknown = set(ids) - {q["id"] for q in questions}
        if unknown:
            raise SystemExit(f"Unknown question ids: {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q["id"] in ids]
    return questions


def git_version() -> str:
    """Current commit, marked "-dirty" if there are uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def grade(question: dict, run: dict) -> tuple[str, str]:
    """Return (outcome, reason). outcome is "pass" or one failure category."""
    if run["status"] == "recursion_limit":
        return "recursion_limit", run["error"]
    if run["status"] == "error":
        return "api_error", run["error"]
    if run["sql"] is None:
        if run["sql_attempts"]:
            return "sql_error", "every run_sql attempt failed"
        return "no_sql", "agent answered without running a query"
    matched, reason = execution_match(run["sql"], question["reference_sql"], question["ordered"])
    return ("pass" if matched else "wrong_result"), reason


def print_summary(records: list[dict]) -> None:
    total = len(records)
    passed = sum(r["outcome"] == "pass" for r in records)
    print(f"\nExecution accuracy: {passed}/{total} ({passed / total:.1%})")

    # The fallback model may score differently, so a mixed run isn't a clean result for the main model
    fallback_runs = sum(LLM_CONFIG.fallback_model in r["models"] for r in records)
    if fallback_runs:
        print(f"  WARNING: {fallback_runs}/{total} attempts used {LLM_CONFIG.fallback_model} for at least one call")

    by_difficulty = defaultdict(lambda: [0, 0])
    for r in records:
        by_difficulty[r["difficulty"]][0] += r["outcome"] == "pass"
        by_difficulty[r["difficulty"]][1] += 1
    order = ["easy", "medium", "hard"]
    parts = [f"{d} {p}/{n}" for d, (p, n) in sorted(by_difficulty.items(),
             key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order))]
    print("  by difficulty: " + " · ".join(parts))

    failures = Counter(r["outcome"] for r in records if r["outcome"] != "pass")
    if failures:
        print("  failures: " + ", ".join(f"{n} {kind}" for kind, n in failures.most_common()))

    avg = lambda key: sum(r[key] for r in records) / total
    print(f"  per question: {avg('tool_calls'):.1f} tool calls · {avg('llm_calls'):.1f} LLM calls · "
          f"{avg('input_tokens') + avg('output_tokens'):,.0f} tokens · {avg('latency_s'):.1f}s")
    if LLM_CONFIG.requests_per_min:
        print(f"  NOTE: latency includes waits from the {LLM_CONFIG.requests_per_min:g} requests/min throttle, "
              "so it isn't comparable across providers")

    # Per-question view; with --repeats this shows how consistent each question is
    print("\n  id    passed  outcomes")
    by_id = defaultdict(list)
    for r in records:
        by_id[r["id"]].append(r)
    for qid, runs in by_id.items():
        n_pass = sum(r["outcome"] == "pass" for r in runs)
        outcomes = ", ".join(r["outcome"] for r in runs)
        print(f"  {qid:<5} {n_pass}/{len(runs):<5} {outcomes}")


def main():
    parser = argparse.ArgumentParser(description="Run the text-to-SQL eval.")
    parser.add_argument("--ids", help="Comma-separated question ids to run (default: all)")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per question (default: 1)")
    parser.add_argument("--pause", type=float, default=2.0,
                        help="Seconds to wait between questions, to ease provider rate limits (default: 2)")
    args = parser.parse_args()
    setup_logging(console_level=os.environ.get("LOG_LEVEL", "WARNING"))
    setup_tracing("text-to-sql-evals")

    questions = load_questions(args.ids.split(",") if args.ids else None)
    meta = {
        "type": "meta",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        **LLM_CONFIG.describe(),
        # A short fingerprint of the system prompt, so runs with different prompts can be told apart
        "prompt_hash": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8],
        "git": git_version(),
        "recursion_limit": RECURSION_LIMIT,
        "repeats": args.repeats,
        "question_ids": [q["id"] for q in questions],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}.jsonl"
    print(f"Running {len(questions)} question(s) x {args.repeats} on {LLM_CONFIG.name}/{LLM_CONFIG.model} "
          f"(prompt {meta['prompt_hash']}, git {meta['git']})")
    print(f"Saving to {out_path}\n")
    log.info("eval_start", extra={**{k: v for k, v in meta.items() if k != "type"}, "results_file": str(out_path)})

    records = []
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(json.dumps(meta) + "\n")
        for repeat in range(1, args.repeats + 1):
            for q in questions:
                # Readable run_id, so the log file can be filtered by question attempt
                with run_context(f"{q['id']}-r{repeat}") as run_id:
                    log.info("question_start", extra={"question_id": q["id"], "repeat": repeat,
                                                      "question": q["question"]})
                    run = run_agent(q["question"])
                    outcome, reason = grade(q, run)
                    log.info("question_end", extra={"question_id": q["id"], "outcome": outcome,
                                                    "reason": reason})
                record = {
                    "type": "result",
                    "id": q["id"],
                    "run_id": run_id,
                    "repeat": repeat,
                    "difficulty": q["difficulty"],
                    "source": q["source"],
                    "outcome": outcome,
                    "reason": reason,
                    **{k: v for k, v in run.items() if k != "status"},
                }
                records.append(record)
                # Written as we go, so a run stopped partway still keeps its results
                out.write(json.dumps(record, default=str) + "\n")
                out.flush()

                mark = "✓" if outcome == "pass" else "✗"
                fallback_note = " [fallback]" if LLM_CONFIG.fallback_model in run["models"] else ""
                print(f"{mark} {q['id']:<5} run {repeat}  {outcome:<16} "
                      f"{run['tool_calls']:>2} tools  {run['latency_s']:>6.1f}s  {reason[:70]}{fallback_note}")
                time.sleep(args.pause)

    passed = sum(r["outcome"] == "pass" for r in records)
    log.info("eval_end", extra={"passed": passed, "total": len(records),
                                "accuracy": round(passed / len(records), 3), "results_file": str(out_path)})
    print_summary(records)
    print(f"\nFull results: {out_path}")
    print(f"Logs: logs/agent.jsonl (filter by run_id, e.g. \"{records[0]['run_id']}\")")


if __name__ == "__main__":
    main()
