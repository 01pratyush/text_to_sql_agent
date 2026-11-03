"""Execution-accuracy check: does the agent's SQL return the same data as the reference?

Matching rules:
  - Both queries are run in full (no 50-row cap) on a read-only connection.
  - Values are normalised: all numbers become floats rounded to FLOAT_DECIMALS,
    dates become ISO strings. Column names are ignored.
  - Lenient columns: every reference column must appear somewhere in the agent's
    result (in any position); extra agent columns are allowed.
  - Row count must match exactly, so a missing LIMIT or a wrong filter fails.
  - Rows are compared as a multiset, or as an ordered list when ordered=True.
"""

import datetime
import decimal
import logging
from collections import Counter

import duckdb

from setup_db import DB_PATH

log = logging.getLogger("evals.compare")

FLOAT_DECIMALS = 2

_con = duckdb.connect(str(DB_PATH), read_only=True)


def _normalize(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, decimal.Decimal)):
        # 596 (COUNT) and 596.0 (SUM) should compare equal, and tiny float
        # differences from summing in a different order should not matter
        return round(float(value), FLOAT_DECIMALS)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    return value


def _run(sql: str) -> tuple[list[str], list[tuple]]:
    result = _con.cursor().execute(sql)
    columns = [d[0] for d in result.description]
    return columns, [tuple(_normalize(v) for v in row) for row in result.fetchall()]


def _same(a: list, b: list, ordered: bool) -> bool:
    return a == b if ordered else Counter(a) == Counter(b)


def _column_mappings(agent_rows, ref_rows, ordered):
    """Yield every way to assign each reference column to a distinct agent column
    whose values match it, as a tuple of agent column indexes in reference order."""
    agent_cols = list(zip(*agent_rows))
    ref_cols = list(zip(*ref_rows))
    candidates = [
        [i for i, a_col in enumerate(agent_cols) if _same(list(a_col), list(r_col), ordered)]
        for r_col in ref_cols
    ]

    def assign(j, used):
        if j == len(candidates):
            yield ()
            return
        for i in candidates[j]:
            if i not in used:
                for rest in assign(j + 1, used | {i}):
                    yield (i, *rest)

    return candidates, assign(0, frozenset())


def execution_match(agent_sql: str | None, reference_sql: str, ordered: bool = False) -> tuple[bool, str]:
    """Compare the results of the agent's SQL and the reference SQL.

    Returns (matched, reason); reason explains a mismatch, or is "match".
    """
    # Filled in by _compare with whatever it got to see (row and column counts)
    shape: dict = {}
    matched, reason = _compare(agent_sql, reference_sql, ordered, shape)
    log.info("compare_result", extra={"matched": matched, "reason": reason, "ordered": ordered, **shape})
    log.debug("compare_sql", extra={"agent_sql": agent_sql, "reference_sql": reference_sql})
    return matched, reason


def _compare(agent_sql, reference_sql, ordered, shape: dict) -> tuple[bool, str]:
    if not agent_sql:
        return False, "no SQL to compare"

    ref_names, ref_rows = _run(reference_sql)  # a broken reference is a dataset bug, so let it raise
    shape.update(ref_rows=len(ref_rows), ref_cols=len(ref_names))
    try:
        agent_names, agent_rows = _run(agent_sql)
    except duckdb.Error as e:
        return False, f"agent SQL failed: {e}"
    shape.update(agent_rows=len(agent_rows), agent_cols=len(agent_names))

    if len(agent_rows) != len(ref_rows):
        return False, f"row count differs: agent {len(agent_rows)} vs reference {len(ref_rows)}"
    if not ref_rows:
        return True, "match (both empty)"

    n_agent_cols, n_ref_cols = len(agent_rows[0]), len(ref_rows[0])
    if n_agent_cols < n_ref_cols:
        return False, f"too few columns: agent {n_agent_cols} vs reference {n_ref_cols}"

    candidates, mappings = _column_mappings(agent_rows, ref_rows, ordered)
    missing = [j for j, c in enumerate(candidates) if not c]
    if missing:
        # When ordered, retry unordered to tell "wrong order" apart from "wrong values"
        if ordered and not [j for j, c in enumerate(_column_mappings(agent_rows, ref_rows, False)[0]) if not c]:
            return False, "same values but wrong row order"
        names = ", ".join(ref_names[j] for j in missing)
        return False, f"no agent column matches reference column(s): {names} (values differ)"

    # Matching columns one at a time isn't enough: the values must also line up
    # in the same rows, so check each full assignment
    for mapping in mappings:
        projected = [tuple(row[i] for i in mapping) for row in agent_rows]
        if _same(projected, ref_rows, ordered):
            return True, "match"
    return False, "each column matches on its own, but the rows don't line up"
