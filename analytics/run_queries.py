"""
Execute all SQL queries against the DuckDB session events database.
Prints formatted results and saves as markdown tables.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANALYTICS_DIR = Path(__file__).resolve().parent
QUERIES_DIR = ANALYTICS_DIR / "queries"
DB_PATH = ANALYTICS_DIR / "lumina.duckdb"
OUTPUT_PATH = ANALYTICS_DIR / "query_results.md"


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def execute_query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame."""
    return con.execute(sql).fetchdf()


def load_and_run_queries(
    db_path: Path | str | None = None,
    queries_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all .sql files from queries_dir and execute them."""
    db = Path(db_path) if db_path else DB_PATH
    q_dir = Path(queries_dir) if queries_dir else QUERIES_DIR

    con = duckdb.connect(str(db), read_only=True)
    results = {}

    for sql_file in sorted(q_dir.glob("*.sql")):
        query_name = sql_file.stem
        sql = sql_file.read_text()
        try:
            df = execute_query(con, sql)
            results[query_name] = df
        except Exception as e:
            print(f"  ERROR in {query_name}: {e}")
            results[query_name] = pd.DataFrame({"error": [str(e)]})

    con.close()
    return results


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def results_to_markdown(results: dict[str, pd.DataFrame]) -> str:
    """Convert query results dict to a markdown document."""
    lines = ["# SQL Analytics — Query Results\n"]
    for name, df in results.items():
        title = name.replace("_", " ").title()
        lines.append(f"## {title}\n")
        lines.append(df.to_markdown(index=False))
        lines.append("\n")
    return "\n".join(lines)


def save_results_markdown(
    results: dict[str, pd.DataFrame],
    output_path: Path | str | None = None,
) -> Path:
    """Save all results as a single markdown file."""
    out = Path(output_path) if output_path else OUTPUT_PATH
    md = results_to_markdown(results)
    out.write_text(md, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    db_path: Path | str | None = None,
    queries_dir: Path | str | None = None,
    output_path: Path | str | None = None,
) -> dict:
    """Execute all queries, print results, save markdown."""
    print("[SQL Queries] Running all queries ...")
    results = load_and_run_queries(db_path, queries_dir)

    for name, df in results.items():
        print(f"\n{'=' * 60}")
        print(f"  {name.replace('_', ' ').upper()}")
        print(f"{'=' * 60}")
        print(df.to_string(index=False))

    md_path = save_results_markdown(results, output_path)
    print(f"\n[SQL Queries] Results saved to {md_path}")
    return {"markdown": str(md_path), "query_count": len(results)}


if __name__ == "__main__":
    run_all()
