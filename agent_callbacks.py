"""LangChain callback handler that logs every LLM call, tool call and graph node.

Attach it through the run config ({"callbacks": [LoggingCallbackHandler()]});
LangChain passes it down to every nested run, so the node and tool code
doesn't need its own logging for these events.

INFO events carry counts, timings and short previews; DEBUG events (file only,
by default) carry full message and tool-output content.
"""

import logging
import time
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

log = logging.getLogger("agent")

PREVIEW_CHARS = 200


def _preview(text: str) -> str:
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"


class LoggingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        # Start time and tool name per callback run_id, to compute durations on *_end
        self._started: dict[UUID, float] = {}
        self._tool_names: dict[UUID, str] = {}
        # Requested model per callback run_id, because some errors (e.g. 503) don't name it
        self._models: dict[UUID, str | None] = {}

    def _elapsed_ms(self, run_id: UUID) -> int | None:
        start = self._started.pop(run_id, None)
        return round((time.perf_counter() - start) * 1000) if start else None

    # --- graph nodes ------------------------------------------------------

    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        # LangGraph reports each node (and some internal steps) as a chain
        log.debug("node_start", extra={"node": kwargs.get("name")})

    def on_chain_error(self, error, *, run_id, **kwargs):
        log.debug("node_error", extra={"error": f"{type(error).__name__}: {error}"})

    # --- LLM calls --------------------------------------------------------

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._started[run_id] = time.perf_counter()
        model = (kwargs.get("metadata") or {}).get("ls_model_name")
        self._models[run_id] = model
        log.debug("llm_start", extra={"model": model, "message_count": len(messages[0])})

    def on_llm_end(self, response, *, run_id, **kwargs):
        duration = self._elapsed_ms(run_id)
        self._models.pop(run_id, None)
        message = response.generations[0][0].message
        usage = message.usage_metadata or {}
        tool_calls = [tc["name"] for tc in message.tool_calls]
        log.info("llm_end", extra={
            "model": message.response_metadata.get("model_name"),
            "duration_ms": duration,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            # Either the tools the model asked for, or a final answer
            "decision": f"call {', '.join(tool_calls)}" if tool_calls else "final answer",
        })
        # .text leaves out Gemini's content blocks and thought signatures
        log.debug("llm_output", extra={"content": message.text, "tool_calls": message.tool_calls})

    def on_llm_error(self, error, *, run_id, **kwargs):
        log.error("llm_error", extra={
            "model": self._models.pop(run_id, None),
            "duration_ms": self._elapsed_ms(run_id),
            "error": f"{type(error).__name__}: {error}",
        })

    # --- tool calls -------------------------------------------------------

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
        self._started[run_id] = time.perf_counter()
        self._tool_names[run_id] = name
        # Not "args": that name (like "name" and "msg") is a built-in LogRecord attribute
        log.info("tool_start", extra={"tool": name, "tool_args": inputs if inputs is not None else input_str})

    def on_tool_end(self, output, *, run_id, **kwargs):
        # Inside ToolNode the output is a ToolMessage; called directly it's the raw return value
        content = str(getattr(output, "content", output))
        log.info("tool_end", extra={
            "tool": self._tool_names.pop(run_id, "unknown"),
            "duration_ms": self._elapsed_ms(run_id),
            "output_lines": len(content.splitlines()),
            "preview": _preview(content),
        })
        log.debug("tool_output", extra={"content": content})

    def on_tool_error(self, error, *, run_id, **kwargs):
        log.error("tool_error", extra={
            "tool": self._tool_names.pop(run_id, "unknown"),
            "duration_ms": self._elapsed_ms(run_id),
            "error": f"{type(error).__name__}: {error}",
        })
