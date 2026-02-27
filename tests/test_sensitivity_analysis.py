"""Tests for sensitivity_analysis.py (clinical threshold tradeoff analysis)."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from experiments.sensitivity_analysis import (
    threshold_sweep,
    find_clinical_tradeoff,
)


class TestThresholdSweep:
    def test_returns_dataframe(self):
        rng = np.random.RandomState(42)
        y_test = rng.randint(0, 2, 100)
        y_proba = np.clip(rng.normal(0.5, 0.2, 100), 0, 1)
        df = threshold_sweep(y_test, y_proba)
        assert isinstance(df, pd.DataFrame)
        assert "threshold" in df.columns
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1" in df.columns

    def test_thresholds_in_range(self):
        y_test = np.array([0, 1, 0, 1, 1, 0])
        y_proba = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.4])
        df = threshold_sweep(y_test, y_proba, thresholds=np.array([0.1, 0.5, 0.9]))
        assert len(df) == 3
        assert all(0.0 <= v <= 1.0 for v in df["precision"])
        assert all(0.0 <= v <= 1.0 for v in df["recall"])

    def test_recall_decreases_with_higher_threshold(self):
        # As threshold goes up, fewer positives predicted → recall goes down
        y_test = np.array([0, 1, 0, 1, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.75])
        df = threshold_sweep(y_test, y_proba, thresholds=np.array([0.1, 0.5, 0.95]))
        recalls = df["recall"].tolist()
        assert recalls[0] >= recalls[-1]  # lower threshold → higher recall


class TestFindClinicalTradeoff:
    def test_returns_dict_with_expected_keys(self):
        y_test = np.array([0, 1, 0, 1, 1, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.75, 0.15, 0.65])
        sweep = threshold_sweep(y_test, y_proba, thresholds=np.linspace(0.1, 0.9, 17))
        result = find_clinical_tradeoff(sweep)
        expected_keys = {
            "default_threshold", "default_precision", "default_recall",
            "clinical_threshold", "clinical_precision", "clinical_recall",
            "precision_drop_pct", "recall_gain_pct"
        }
        assert expected_keys == set(result.keys())

    def test_clinical_threshold_lower_than_default(self):
        rng = np.random.RandomState(42)
        y_test = rng.randint(0, 2, 200)
        # Calibrated probabilities
        y_proba = np.where(y_test == 1,
                           np.clip(rng.normal(0.7, 0.1, 200), 0, 1),
                           np.clip(rng.normal(0.3, 0.1, 200), 0, 1))
        sweep = threshold_sweep(y_test, y_proba)
        result = find_clinical_tradeoff(sweep)
        assert result["clinical_threshold"] <= result["default_threshold"]

    def test_run_all_produces_report(self, tmp_output_dir):
        from experiments.sensitivity_analysis import run_all
        import pandas as pd

        # Create a minimal suicide_watch.csv
        texts = (
            ["I want to end it all, nothing matters"] * 50 +
            ["Feeling great today, life is good"] * 50
        )
        classes = ["suicide"] * 50 + ["non-suicide"] * 50
        df = pd.DataFrame({"text": texts, "class": classes})
        csv_path = tmp_output_dir / "sw.csv"
        df.to_csv(csv_path, index=False)

        result = run_all(dataset_path=csv_path, output_dir=tmp_output_dir, sample_size=80)
        assert "tradeoff" in result
        assert "report_path" in result
        assert Path(result["report_path"]).exists()
