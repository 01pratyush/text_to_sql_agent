"""Generate the TPC-H benchmark database as a DuckDB file.

Uses DuckDB's built-in tpch extension. Run once (re-running rebuilds it):

    python setup_db.py            # scale factor 0.1 (~100 MB of raw data)
    python setup_db.py --sf 1     # scale factor 1   (~1 GB of raw data, the standard size)

DuckDB compresses the data, so the .duckdb file is smaller (~28 MB at 0.1).

The build settings are saved next to the database in data/tpch.info.json, so
eval runs can record which database they were scored against (see db_info).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import duckdb

DB_PATH = Path("data") / "tpch.duckdb"
# A separate file rather than a table, so the agent's list_tables doesn't show it
INFO_PATH = DB_PATH.with_suffix(".info.json")


def db_info() -> dict:
    """Describe the current database: its build settings and each table's row count.

    The row counts are read from the database itself, so they are accurate even
    if the info file is missing (e.g. a database built before it existed).
    """
    try:
        build = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        build = {"scale_factor": "unknown"}
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        tables = [t for (t,) in con.execute("SHOW TABLES").fetchall()]
        row_counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    finally:
        con.close()
    return {**build, "row_counts": row_counts}


def main():
    parser = argparse.ArgumentParser(description="Create the TPC-H DuckDB database.")
    parser.add_argument(
        "--sf", type=float, default=0.1,
        help="TPC-H scale factor (1 = ~1 GB of raw data). Default: 0.1",
    )
    args = parser.parse_args()

    DB_PATH.parent.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()  # start fresh: dbgen appends to existing tables, duplicating rows
    # Removed first, so a failed build doesn't leave the old database's settings behind
    INFO_PATH.unlink(missing_ok=True)

    con = duckdb.connect(str(DB_PATH))
    # The extension is downloaded from DuckDB's repository on first INSTALL
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    print(f"Generating TPC-H data at scale factor {args.sf} into {DB_PATH} ...")
    con.execute(f"CALL dbgen(sf = {args.sf})")

    print("\nTables:")
    for (table,) in con.execute("SHOW TABLES").fetchall():
        (count,) = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"  {table:<10} {count:>12,} rows")
    con.close()

    INFO_PATH.write_text(json.dumps({
        "scale_factor": args.sf,
        "duckdb_version": duckdb.__version__,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"\nBuild settings saved to {INFO_PATH}")


if __name__ == "__main__":
    main()
