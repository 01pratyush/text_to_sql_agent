"""Read-only SQL tools over the TPC-H DuckDB database.

Safety is enforced in code, not just in the prompt:
  1. The connection is opened read_only, so DuckDB itself rejects writes.
  2. run_sql parses the query and only allows a single SELECT statement.
  3. Results are capped at MAX_ROWS and queries are cancelled after QUERY_TIMEOUT_S.

The logging callback (agent_callbacks.py) already logs each tool call and its
output; this module logs only what happens inside: rejections, database
timings, row counts and truncation.
"""

import logging
import threading
import time

import duckdb
from langchain_core.tools import tool

from setup_db import DB_PATH

log = logging.getLogger("sql_tools")

# Rows returned to the model; more would flood the context. The number 50 is
# also written in run_sql's docstring and main.SYSTEM_PROMPT; keep them in sync
MAX_ROWS = 50
MAX_CELL_CHARS = 100   # long text values are truncated
QUERY_TIMEOUT_S = 30   # long-running queries are interrupted
SAMPLE_ROWS = 3        # example rows shown by describe_table

if not DB_PATH.exists():
    raise FileNotFoundError(f"{DB_PATH} not found. Create it with: python setup_db.py")

# One shared read-only connection. Each tool call works on its own cursor,
# because ToolNode runs multiple tool calls from one reply in parallel threads
_con = duckdb.connect(str(DB_PATH), read_only=True)


def _format_rows(columns: list[str], rows: list[tuple]) -> str:
    """Render rows as a compact pipe-separated table the model can read."""
    def cell(value) -> str:
        text = "NULL" if value is None else str(value)
        return text if len(text) <= MAX_CELL_CHARS else text[:MAX_CELL_CHARS] + "..."

    lines = [" | ".join(columns)]
    lines += [" | ".join(cell(v) for v in row) for row in rows]
    return "\n".join(lines)


def _table_names(cur) -> list[str]:
    return [name for (name,) in cur.execute("SHOW TABLES").fetchall()]


def schema_summary() -> str:
    """Compact schema for the system prompt: one line per table, with column names and types."""
    cur = _con.cursor()
    lines = []
    for name in _table_names(cur):
        columns = cur.execute(f'DESCRIBE "{name}"').fetchall()
        lines.append(f"{name}: " + ", ".join(f"{col[0]} {col[1]}" for col in columns))
    return "\n".join(lines)


@tool
def list_tables() -> str:
    """List all tables in the database with their row counts. Call this first to see what data exists."""
    cur = _con.cursor()
    lines = []
    for name in _table_names(cur):
        (count,) = cur.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()
        lines.append(f"{name} ({count:,} rows)")
    return "\n".join(lines)


@tool(parse_docstring=True)
def describe_table(table_name: str) -> str:
    """Show a table's columns, their types, and a few example rows.

    Args:
        table_name: Name of the table, exactly as returned by list_tables
    """
    cur = _con.cursor()
    tables = _table_names(cur)
    # Only accept known table names, since the name is put into SQL below
    if table_name not in tables:
        log.warning("unknown_table", extra={"table": table_name})
        return f"Error: unknown table '{table_name}'. Available tables: {', '.join(tables)}"

    columns = cur.execute(f'DESCRIBE "{table_name}"').fetchall()
    schema = "\n".join(f"  {col[0]} {col[1]}" for col in columns)

    sample = cur.execute(f'SELECT * FROM "{table_name}" LIMIT {SAMPLE_ROWS}')
    sample_cols = [d[0] for d in sample.description]
    sample_text = _format_rows(sample_cols, sample.fetchall())

    return f"Table {table_name}\nColumns:\n{schema}\n\nExample rows:\n{sample_text}"


@tool(parse_docstring=True)
def run_sql(query: str) -> str:
    """Run a read-only SQL query (DuckDB dialect) and return the result rows.

    Only a single SELECT statement is allowed. At most 50 rows are returned,
    so aggregate or add ORDER BY ... LIMIT when a result could be large.

    Args:
        query: One DuckDB SELECT statement
    """
    try:
        statements = duckdb.extract_statements(query)
    except duckdb.Error as e:
        log.warning("sql_rejected", extra={"reason": "syntax_error", "error": str(e)})
        return f"SQL syntax error: {e}"
    if len(statements) != 1:
        log.warning("sql_rejected", extra={"reason": "multiple_statements", "statements": len(statements)})
        return "Error: provide exactly one SQL statement."
    if statements[0].type != duckdb.StatementType.SELECT:
        log.warning("sql_rejected", extra={"reason": "not_select", "statement_type": statements[0].type.name})
        return f"Error: only SELECT queries are allowed (got {statements[0].type.name})."

    cur = _con.cursor()
    # Cancel the query if it runs too long
    timer = threading.Timer(QUERY_TIMEOUT_S, cur.interrupt)
    timer.start()
    start = time.perf_counter()
    try:
        result = cur.execute(query)
        columns = [d[0] for d in result.description]
        rows = result.fetchmany(MAX_ROWS + 1)
    except duckdb.InterruptException:
        log.warning("sql_timeout", extra={"timeout_s": QUERY_TIMEOUT_S})
        return f"Error: query cancelled after {QUERY_TIMEOUT_S}s. Simplify it or add filters."
    except duckdb.Error as e:
        # The model sees this text and can fix its SQL; the log keeps the error type
        log.warning("sql_failed", extra={"error_type": type(e).__name__, "error": str(e)})
        return f"SQL error: {e}"
    finally:
        timer.cancel()

    truncated = len(rows) > MAX_ROWS
    log.info("sql_executed", extra={
        "db_ms": round((time.perf_counter() - start) * 1000),
        "columns": len(columns),
        "rows_returned": min(len(rows), MAX_ROWS),
        "truncated": truncated,
    })
    if not rows:
        return _format_rows(columns, []) + "\n(0 rows)"
    text = _format_rows(columns, rows[:MAX_ROWS])
    if truncated:
        text += f"\n(showing first {MAX_ROWS} rows; more exist)"
    return text


SQL_TOOLS = [list_tables, describe_table, run_sql]
