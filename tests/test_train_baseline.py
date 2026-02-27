"""Tests for experiments/train_baseline.py"""

import numpy as np
import pytest
from pathlib import Path

from experiments.train_baseline import (
    train_tfidf_baseline,
    evaluate_deberta_proxy,
    save_baseline_model,
    load_baseline_model,
)

class TestBaselineTraining:
    @pytest.fixture
    def small_corpus(self):
        texts = [
            "happy good great", "sad bad terrible", "great good fine", "terrible sad",
            "happy", "sad", "good", "bad", "great", "terrible"
        ]
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        return texts, labels

    def test_train_tfidf_baseline(self, small_corpus):
        texts, labels = small_corpus
        res = train_tfidf_baseline(texts, labels, max_features=100, test_size=0.4)
        
        assert "model" in res
        assert "vectorizer" in res
        assert "metrics" in res
        
        metrics = res["metrics"]
        assert "ROC-AUC" in metrics
        assert "F1" in metrics
        assert metrics["Inference (ms/sample)"] >= 0

    @pytest.mark.slow
    def test_evaluate_deberta_proxy(self, small_corpus):
        texts, labels = small_corpus
        res = evaluate_deberta_proxy(texts, labels, test_size=0.4, model_path="prajjwal1/bert-tiny")
        assert "metrics" in res
        assert "ROC-AUC" in res["metrics"]
        assert "F1" in res["metrics"]

    def test_model_save_load(self, small_corpus, tmp_output_dir):
        texts, labels = small_corpus
        res = train_tfidf_baseline(texts, labels, max_features=100, test_size=0.4)
        path = tmp_output_dir / "model.pkl"
        
        save_baseline_model(res["model"], res["vectorizer"], path)
        assert path.exists()
        
        loaded_model, loaded_vec = load_baseline_model(path)
        assert loaded_model is not None
        assert loaded_vec is not None
        
        # Predictions should match
        X_test_tfidf = loaded_vec.transform(res["X_test"])
        preds = loaded_model.predict(X_test_tfidf)
        np.testing.assert_array_equal(preds, res["y_pred"])


class TestEndToEnd:
    @pytest.mark.slow
    def test_run_all(self, suicide_df, tmp_output_dir):
        csv_path = tmp_output_dir / "test_suicide.csv"
        suicide_df.to_csv(csv_path, index=False)

        from experiments.train_baseline import run_all
        out_dir = tmp_output_dir / "exp_out"
        outputs = run_all(
            dataset_path=csv_path,
            output_dir=out_dir,
            sample_size=20
        )

        assert (out_dir / "baseline_model.pkl").exists()
        assert (out_dir / "comparison_table.csv").exists()
        assert "baseline_metrics" in outputs
        assert "deberta_metrics" in outputs
