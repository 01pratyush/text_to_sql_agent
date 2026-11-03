"""Generate the TPC-H benchmark database as a DuckDB file.

Uses DuckDB's built-in tpch extension. Run once (re-running rebuilds it):

    python setup_db.py            # scale factor 0.1 (~100 MB of raw data)
    python setup_db.py --sf 1     # scale factor 1   (~1 GB of raw data, the standard size)

DuckDB compresses the data, so the .duckdb file is smaller (~28 MB at 0.1).
"""

import argparse
from pathlib import Path

import duckdb

DB_PATH = Path("data") / "tpch.duckdb"


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


if __name__ == "__main__":
    main()
