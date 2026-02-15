"""
Generate simulated session event logs for SQL analytics.

Produces a DataFrame of 2000 sessions with realistic distributions
matching the LuminaMind app behaviour.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / "session_events.csv"
DB_PATH = OUTPUT_DIR / "lumina.duckdb"


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_session_data(
    n_sessions: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic session event data.

    Fields:
      session_id, user_id, message_count, session_duration_sec,
      risk_score_final, risk_level, threshold_triggered, created_at
    """
    rng = np.random.RandomState(random_state)

    # Message counts: right-skewed distribution (most sessions are short)
    message_counts = rng.negative_binomial(3, 0.3, size=n_sessions) + 1
    message_counts = np.clip(message_counts, 1, 50)

    # Session duration: correlated with message count
    session_durations = (message_counts * rng.uniform(15, 60, size=n_sessions)).astype(int)

    # Risk scores: mixture — most sessions low risk, some high
    risk_scores = np.zeros(n_sessions)
    # 70% low risk
    n_low = int(n_sessions * 0.7)
    risk_scores[:n_low] = rng.beta(1.5, 8, size=n_low)
    # 15% medium
    n_med = int(n_sessions * 0.15)
    risk_scores[n_low : n_low + n_med] = rng.beta(3, 3, size=n_med)
    # 15% high
    n_high = n_sessions - n_low - n_med
    risk_scores[n_low + n_med :] = rng.beta(8, 2, size=n_high)
    rng.shuffle(risk_scores)
    risk_scores = np.clip(risk_scores, 0, 1)

    # Risk levels
    def _risk_level(score):
        if score < 0.25:
            return "low"
        elif score < 0.50:
            return "medium"
        elif score < 0.75:
            return "high"
        else:
            return "critical"

    risk_levels = [_risk_level(s) for s in risk_scores]

    # Threshold triggered (5 messages)
    threshold_triggered = message_counts >= 5

    # User IDs: ~200 unique users
    n_users = min(200, n_sessions)
    user_ids = [f"user_{i:04d}" for i in range(n_users)]
    assigned_users = rng.choice(user_ids, size=n_sessions)

    # Session IDs
    session_ids = [str(uuid.UUID(int=int(rng.randint(0, 2**31 - 1)))) for _ in range(n_sessions)]

    # Timestamps over 90 days
    base_ts = pd.Timestamp("2025-01-01")
    random_offsets = pd.to_timedelta(rng.uniform(0, 90 * 24 * 3600, size=n_sessions), unit="s")
    created_at = base_ts + random_offsets

    df = pd.DataFrame({
        "session_id": session_ids,
        "user_id": assigned_users,
        "message_count": message_counts,
        "session_duration_sec": session_durations,
        "risk_score_final": np.round(risk_scores, 4),
        "risk_level": risk_levels,
        "threshold_triggered": threshold_triggered,
        "created_at": created_at,
    })
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_to_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save session data to CSV."""
    csv_path = path or CSV_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def load_into_duckdb(df: pd.DataFrame, db_path: Path | None = None) -> Path:
    """Load session data into DuckDB."""
    if not HAS_DUCKDB:
        raise ImportError("duckdb is not installed")
    db = db_path or DB_PATH
    con = duckdb.connect(str(db))
    con.execute("DROP TABLE IF EXISTS session_events")
    con.execute("CREATE TABLE session_events AS SELECT * FROM df")
    count = con.execute("SELECT COUNT(*) FROM session_events").fetchone()[0]
    con.close()
    print(f"  Loaded {count} rows into {db}")
    return db


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    n_sessions: int = 2000,
    output_dir: Path | str | None = None,
) -> dict:
    """Generate session data, save to CSV and DuckDB."""
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Session Data] Generating synthetic session events ...")
    df = generate_session_data(n_sessions=n_sessions)

    csv_path = save_to_csv(df, out_dir / "session_events.csv")
    print(f"  CSV saved: {csv_path}")

    db_path = load_into_duckdb(df, out_dir / "lumina.duckdb")

    outputs = {"csv": str(csv_path), "duckdb": str(db_path), "n_sessions": len(df)}
    print(f"[Session Data] Done. {len(df)} sessions generated.")
    return outputs


if __name__ == "__main__":
    run_all()
