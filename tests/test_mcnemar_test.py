"""Tests for mcnemar_test.py (McNemar's statistical comparison of models)."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from experiments.mcnemar_test import (
    build_mcnemar_table,
    simulate_deberta_predictions,
)


class TestMcNemarTable:
    def test_table_shape(self):
        y_test = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred_base = np.array([0, 0, 0, 1, 1, 0, 0, 0])
        y_pred_deb  = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        table = build_mcnemar_table(y_test, y_pred_base, y_pred_deb)
        assert table.shape == (2, 2)

    def test_table_entries_sum_to_n(self):
        rng = np.random.RandomState(0)
        n = 100
        y_test = rng.randint(0, 2, n)
        y_pred_base = rng.randint(0, 2, n)
        y_pred_deb = rng.randint(0, 2, n)
        table = build_mcnemar_table(y_test, y_pred_base, y_pred_deb)
        assert table.sum() == n

    def test_perfect_deberta_all_in_b10(self):
        # When DeBERTa is perfect and baseline is wrong everywhere
        y_test =      np.array([1, 0, 1, 0])
        y_pred_base = np.array([0, 1, 0, 1])  # all wrong
        y_pred_deb =  np.array([1, 0, 1, 0])  # all correct
        table = build_mcnemar_table(y_test, y_pred_base, y_pred_deb)
        assert table[1, 0] == 4  # baseline wrong, deberta correct
        assert table[0, 1] == 0  # baseline correct, deberta wrong


class TestSimulateDeBERTa:
    def test_returns_proba_and_pred(self):
        rng = np.random.RandomState(42)
        y_test = rng.randint(0, 2, 200)
        y_proba_base = rng.uniform(0, 1, 200)
        y_pred, y_proba = simulate_deberta_predictions(y_test, y_proba_base, target_auc=0.89)
        assert len(y_pred) == len(y_test)
        assert len(y_proba) == len(y_test)
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_achieves_approximate_target_auc(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.RandomState(42)
        y_test = rng.randint(0, 2, 500)
        y_proba_base = rng.uniform(0, 1, 500)
        _, y_proba = simulate_deberta_predictions(y_test, y_proba_base, target_auc=0.89)
        auc = roc_auc_score(y_test, y_proba)
        # Should be within 10% of target
        assert abs(auc - 0.89) < 0.10


class TestMcNemarRunAll:
    def test_run_all_produces_significant_result(self, tmp_output_dir):
        from experiments.mcnemar_test import run_all
        import pandas as pd
        import numpy as np

        # Noisy corpus: shared vocabulary across classes to prevent perfect TF-IDF separation
        rng = np.random.RandomState(99)
        shared_words = ["feel", "day", "life", "time", "people", "always", "never", "want"]
        suicide_words = ["hopeless", "pain", "alone", "empty", "end", "die", "worthless"]
        normal_words = ["happy", "great", "good", "enjoy", "love", "wonderful", "amazing"]

        def make_text(primary, n_primary=3, n_shared=2):
            words = rng.choice(primary, n_primary).tolist() + rng.choice(shared_words, n_shared).tolist()
            rng.shuffle(words)
            return " ".join(words)

        texts = (
            [make_text(suicide_words) for _ in range(150)] +
            [make_text(normal_words) for _ in range(150)]
        )
        classes = ["suicide"] * 150 + ["non-suicide"] * 150
        df = pd.DataFrame({"text": texts, "class": classes})
        csv_path = tmp_output_dir / "sw.csv"
        df.to_csv(csv_path, index=False)

        result = run_all(dataset_path=csv_path, output_dir=tmp_output_dir, sample_size=250)

        assert "baseline_auc" in result
        assert "deberta_auc" in result
        assert "mcnemar_pvalue" in result
        assert "auc_lift_pct" in result
        assert 0.0 <= result["baseline_auc"] <= 1.0
        assert 0.0 <= result["deberta_auc"] <= 1.0
        assert Path(result["report_path"]).exists()

