"""Run the agent on one question and collect what it did.

Run from the project root as a module, so `main` can be imported:

    python -m evals.agent_runner "How many customers are there in Germany?"
"""

import logging
import sys
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai.chat_models import GoogleRateLimitError
from langgraph.errors import GraphRecursionError

from log_config import current_run_id
from main import RECURSION_LIMIT, graph, make_config

log = logging.getLogger("evals.agent_runner")

RATE_LIMIT_RETRIES = 3
RATE_LIMIT_WAIT_S = 20  # first wait; doubles on each retry

# Prefixes the tools in sql_tools.py use when they return an error as text
TOOL_ERROR_PREFIXES = ("Error", "SQL error", "SQL syntax error")


def _tool_failed(msg: ToolMessage) -> bool:
    # status="error" is set when ToolNode itself catches a problem (e.g. bad arguments)
    return msg.status == "error" or msg.content.startswith(TOOL_ERROR_PREFIXES)


def _summarize(messages: list) -> dict:
    """Pull the SQL attempts, final answer and usage numbers out of a run's messages."""
    results_by_id = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]

    sql_attempts = []
    tool_calls = 0
    for m in ai_messages:
        for tc in m.tool_calls:
            tool_calls += 1
            if tc["name"] != "run_sql":
                continue
            result = results_by_id.get(tc["id"])
            sql_attempts.append({
                "sql": tc["args"].get("query", ""),
                "ok": result is not None and not _tool_failed(result),
                "result_preview": result.content[:300] if result else None,
            })

    # The SQL we grade is the last one that ran successfully, not the SQL shown
    # in the answer text, which may differ from what was actually executed
    successful = [a["sql"] for a in sql_attempts if a["ok"]]
    final = ai_messages[-1] if ai_messages and not ai_messages[-1].tool_calls else None

    usage = [m.usage_metadata or {} for m in ai_messages]
    return {
        "sql": successful[-1] if successful else None,
        # .text, because Gemini 3 returns .content as a list of blocks
        "answer": final.text if final else None,
        "sql_attempts": sql_attempts,
        "models": sorted({m.response_metadata["model_name"] for m in ai_messages
                          if "model_name" in m.response_metadata}),
        "llm_calls": len(ai_messages),
        "tool_calls": tool_calls,
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
    }


def run_agent(question: str) -> dict:
    """Run the agent on one question in a fresh conversation and report what happened.

    Returns a dict with:
      status: "ok", "recursion_limit" or "error"
      sql: the last successfully executed run_sql query (None if there was none)
      answer: the final answer text (None if the agent never finished)
      models: every model that answered during the run (shows if the fallback was used)
      sql_attempts, llm_calls, tool_calls, input_tokens, output_tokens, latency_s, error
    """
    # A new thread_id per question, so no question sees another's history.
    # make_config also attaches the logging callbacks, as in the chat CLI
    config = make_config(f"eval-{uuid.uuid4()}")
    # Tags the Phoenix trace with the same run_id as the log events (e.g. c02-r1)
    config["metadata"] = {"run_id": current_run_id()}
    inputs = {"messages": [HumanMessage(content=question)]}

    status, error = "ok", None
    wait = RATE_LIMIT_WAIT_S
    start = time.perf_counter()
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            graph.invoke(inputs, config)
            break
        except GraphRecursionError:
            status, error = "recursion_limit", f"hit recursion_limit={RECURSION_LIMIT}"
            break
        except GoogleRateLimitError as e:
            if attempt == RATE_LIMIT_RETRIES:
                status, error = "error", f"rate limited: {e}"
                break
            log.warning("rate_limited", extra={"attempt": attempt + 1, "wait_s": wait})
            time.sleep(wait)
            wait *= 2
            # Retry in a fresh thread so a half-finished run doesn't leak into the next one
            config["configurable"]["thread_id"] = f"eval-{uuid.uuid4()}"
        except Exception as e:
            log.exception("agent_run_error")
            status, error = "error", f"{type(e).__name__}: {e}"
            break
    latency = time.perf_counter() - start

    # Read the messages from the checkpointer rather than invoke()'s return
    # value, so we still get them when the run stopped with an exception
    messages = graph.get_state(config).values.get("messages", [])
    summary = {"status": status, **_summarize(messages), "latency_s": round(latency, 2), "error": error}
    log.info("agent_run_end", extra={
        "status": status,
        "has_sql": summary["sql"] is not None,
        "sql_attempts": len(summary["sql_attempts"]),
        "models": summary["models"],
        "llm_calls": summary["llm_calls"],
        "tool_calls": summary["tool_calls"],
        "latency_s": summary["latency_s"],
    })
    return summary


if __name__ == "__main__":
    import json

    from log_config import run_context, setup_logging
    from tracing import setup_tracing

    setup_logging()
    setup_tracing("text-to-sql-evals")
    question = " ".join(sys.argv[1:]) or "How many customers are there in Germany?"
    with run_context():
        print(json.dumps(run_agent(question), indent=2, default=str))
