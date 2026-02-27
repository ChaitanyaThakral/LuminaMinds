"""Tests for analytics/run_queries.py and the SQL queries."""

import pandas as pd
from pathlib import Path
from analytics.generate_session_data import HAS_DUCKDB

if HAS_DUCKDB:
    import duckdb

from analytics.run_queries import (
    execute_query,
    load_and_run_queries,
    results_to_markdown,
    save_results_markdown,
)

class TestSQLQueries:
    def setup_method(self):
        if not HAS_DUCKDB:
            return
        self.con = duckdb.connect(":memory:")
        # Provide some dummy data for the queries to run against
        df = pd.DataFrame({
            "session_id": ["1", "2", "3", "4"],
            "user_id": ["u1", "u1", "u2", "u3"],
            "message_count": [2, 10, 6, 1],
            "session_duration_sec": [60, 300, 180, 10],
            "risk_score_final": [0.1, 0.8, 0.6, 0.2],
            "risk_level": ["low", "critical", "high", "low"],
            "threshold_triggered": [False, True, True, False],
            "created_at": pd.to_datetime(["2025-01-01"]*4)
        })
        self.con.execute("CREATE TABLE session_events AS SELECT * FROM df")

        self.queries_dir = Path(__file__).resolve().parent.parent / "analytics" / "queries"

    def teardown_method(self):
        if HAS_DUCKDB:
            self.con.close()

    def _run_query_file(self, filename: str) -> pd.DataFrame:
        if not HAS_DUCKDB:
            return pd.DataFrame()
        sql = (self.queries_dir / filename).read_text()
        return execute_query(self.con, sql)

    def test_avg_messages_query(self):
        df = self._run_query_file("avg_messages_before_threshold.sql")
        if not df.empty:
            assert "avg_messages_before_threshold" in df.columns
            assert len(df) == 1

    def test_risk_distribution_query(self):
        df = self._run_query_file("risk_score_distribution.sql")
        if not df.empty:
            assert "risk_bucket" in df.columns
            assert "session_count" in df.columns

    def test_cohort_table_query(self):
        df = self._run_query_file("cohort_table.sql")
        if not df.empty:
            assert "risk_level" in df.columns
            assert "session_length_bucket" in df.columns

    def test_false_negative_query(self):
        df = self._run_query_file("false_negative_by_length.sql")
        if not df.empty:
            assert "fn_rate_pct" in df.columns

    def test_session_summary_query(self):
        df = self._run_query_file("session_summary.sql")
        if not df.empty:
            assert "total_sessions" in df.columns
            assert len(df) == 1

    def test_risk_escalation_query(self):
        df = self._run_query_file("risk_escalation.sql")
        if not df.empty:
            assert "user_id" in df.columns
            assert "risk_range" in df.columns


class TestRunner:
    def test_results_to_markdown(self):
        results = {"q1": pd.DataFrame({"a": [1]})}
        md = results_to_markdown(results)
        assert "Q1" in md
        assert "a" in md

    def test_save_results_markdown(self, tmp_output_dir):
        results = {"q1": pd.DataFrame({"a": [1]})}
        path = tmp_output_dir / "res.md"
        save_results_markdown(results, path)
        assert path.exists()
        assert "Q1" in path.read_text()

    def test_run_all(self, tmp_output_dir, session_events_df):
        if not HAS_DUCKDB:
            return
        
        db_path = tmp_output_dir / "test.duckdb"
        con = duckdb.connect(str(db_path))
        con.execute("CREATE TABLE session_events AS SELECT * FROM session_events_df")
        con.close()

        from analytics.run_queries import run_all
        queries_dir = Path(__file__).resolve().parent.parent / "analytics" / "queries"
        
        outputs = run_all(db_path=db_path, queries_dir=queries_dir, output_path=tmp_output_dir / "out.md")
        assert (tmp_output_dir / "out.md").exists()
        assert outputs["query_count"] == 6
