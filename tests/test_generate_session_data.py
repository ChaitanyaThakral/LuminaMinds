"""Tests for analytics/generate_session_data.py"""

import pandas as pd
from pathlib import Path
from analytics.generate_session_data import (
    generate_session_data,
    save_to_csv,
    load_into_duckdb,
    HAS_DUCKDB,
)


class TestSessionDataGeneration:
    def test_row_count(self):
        df = generate_session_data(n_sessions=50)
        assert len(df) == 50

    def test_schema(self, session_events_df):
        cols = [
            "session_id", "user_id", "message_count", "session_duration_sec",
            "risk_score_final", "risk_level", "threshold_triggered", "created_at"
        ]
        for col in cols:
            assert col in session_events_df.columns

    def test_risk_levels(self, session_events_df):
        levels = set(session_events_df["risk_level"].unique())
        assert levels.issubset({"low", "medium", "high", "critical"})

    def test_threshold_triggered(self, session_events_df):
        # Triggered if message_count >= 5
        triggered = session_events_df[session_events_df["threshold_triggered"]]
        not_triggered = session_events_df[~session_events_df["threshold_triggered"]]
        
        if not triggered.empty:
            assert (triggered["message_count"] >= 5).all()
        if not not_triggered.empty:
            assert (not_triggered["message_count"] < 5).all()


class TestPersistence:
    def test_csv_roundtrip(self, session_events_df, tmp_output_dir):
        path = tmp_output_dir / "test_sessions.csv"
        save_to_csv(session_events_df, path)
        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == len(session_events_df)
        assert "session_id" in loaded.columns

    def test_duckdb_load(self, session_events_df, tmp_output_dir):
        if not HAS_DUCKDB:
            return
        db_path = tmp_output_dir / "test.duckdb"
        load_into_duckdb(session_events_df, db_path)
        assert db_path.exists()
        
        import duckdb
        con = duckdb.connect(str(db_path))
        count = con.execute("SELECT COUNT(*) FROM session_events").fetchone()[0]
        assert count == len(session_events_df)
        con.close()


class TestEndToEnd:
    def test_run_all(self, tmp_output_dir):
        if not HAS_DUCKDB:
            return
        from analytics.generate_session_data import run_all
        outputs = run_all(n_sessions=10, output_dir=tmp_output_dir)
        assert (tmp_output_dir / "session_events.csv").exists()
        assert (tmp_output_dir / "lumina.duckdb").exists()
