"""Structured logging for the agent, tools and evals.

Every log call uses an event name as the message plus structured fields:

    log.info("tool_end", extra={"tool": "run_sql", "duration_ms": 41})

Two outputs:
  - logs/agent.jsonl : every event at DEBUG and above, one JSON object per line
  - console          : readable lines at LOG_LEVEL (env var, default WARNING)

Every event carries the current run_id (one per chat turn or eval question),
so the file can be filtered to everything that happened in one run.
Call setup_logging() once, from an entry point (CLI, eval runner), never from library code.
"""

import json
import logging
import os
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILE = Path("logs") / "agent.jsonl"
MAX_LOG_BYTES = 5_000_000  # rotate after ~5 MB, keeping 3 old files
CONSOLE_VALUE_CHARS = 160  # long field values are shortened on the console only

# Set per chat turn / eval question; contextvars also reach the threads
# ToolNode runs tools in, so tool events get the right run_id too
_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)

# Attributes every LogRecord has; anything else came from extra={...}
_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "run_id"}


@contextmanager
def run_context(run_id: str | None = None):
    """Tag every log event inside the block with a run_id (generated if not given)."""
    token = _run_id.set(run_id or uuid.uuid4().hex[:8])
    try:
        yield _run_id.get()
    finally:
        _run_id.reset(token)


def current_run_id() -> str | None:
    return _run_id.get()


def _fields(record: logging.LogRecord) -> dict:
    return {k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS}


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "run_id": record.run_id,
            **_fields(record),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record):
        def short(value):
            text = str(value).replace("\n", " ⏎ ")
            return text if len(text) <= CONSOLE_VALUE_CHARS else text[:CONSOLE_VALUE_CHARS] + "…"

        time = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        fields = " ".join(f"{k}={short(v)}" for k, v in _fields(record).items())
        line = f"  {time} {record.levelname:<5} [{record.run_id or '-'}] {record.getMessage()} {fields}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(console_level: str | None = None) -> None:
    """Configure the root logger. Safe to call more than once."""
    root = logging.getLogger()
    if getattr(root, "_text_to_sql_configured", False):
        return
    root.setLevel(logging.DEBUG)

    LOG_FILE.parent.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_LOG_BYTES, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())

    # The console handler writes to stderr, which on Windows defaults to cp1252
    # and can't print characters like "⏎" or ones in LLM output
    sys.stderr.reconfigure(encoding="utf-8")
    console = logging.StreamHandler()
    console.setLevel((console_level or os.environ.get("LOG_LEVEL", "WARNING")).upper())
    console.setFormatter(ConsoleFormatter())

    for handler in (file_handler, console):
        handler.addFilter(_RunIdFilter())
        root.addHandler(handler)

    # Libraries that log every HTTP request; their warnings still come through
    for noisy in ("httpx", "httpcore", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._text_to_sql_configured = True
