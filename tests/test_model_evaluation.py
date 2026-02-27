"""Tests for notebooks/model_evaluation.py"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from notebooks.model_evaluation import (
    load_risk_model,
    predict_batch,
    compute_precision_recall,
    plot_precision_recall_curve,
    compute_calibration,
    plot_calibration_curve,
    compute_threshold_metrics,
    compute_confusion_matrix,
    plot_confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

@pytest.mark.slow
class TestModelEvaluationInference:
    def test_load_and_predict(self):
        model, tokenizer = load_risk_model("prajjwal1/bert-tiny")
        texts = ["hello world", "i want to die"]
        probs = predict_batch(texts, model, tokenizer, batch_size=2)
        assert len(probs) == 2
        assert all(0 <= p <= 1 for p in probs)


class TestMetricsAndPlots:
    @pytest.fixture
    def mock_data(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6, 0.7, 0.45, 0.05])
        return y_true, y_scores

    def test_precision_recall(self, mock_data):
        y_true, y_scores = mock_data
        prec, rec, thresh, ap = compute_precision_recall(y_true, y_scores)
        assert len(prec) == len(rec)
        assert 0.0 <= ap <= 1.0

    def test_plot_pr_curve(self, mock_data):
        y_true, y_scores = mock_data
        prec, rec, thresh, ap = compute_precision_recall(y_true, y_scores)
        fig = plot_precision_recall_curve(prec, rec, ap)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_calibration(self, mock_data):
        y_true, y_scores = mock_data
        frac_pos, mean_pred = compute_calibration(y_true, y_scores, n_bins=3)
        assert len(frac_pos) > 0
        assert len(mean_pred) > 0

    def test_plot_calibration(self, mock_data):
        y_true, y_scores = mock_data
        frac_pos, mean_pred = compute_calibration(y_true, y_scores, n_bins=3)
        fig = plot_calibration_curve(frac_pos, mean_pred)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_threshold_metrics(self, mock_data):
        y_true, y_scores = mock_data
        df = compute_threshold_metrics(y_true, y_scores, thresholds=[0.3, 0.5, 0.7])
        assert len(df) == 3
        assert "Precision" in df.columns
        assert "Recall" in df.columns
        # at 0.0 recall is 1.0
        df_zero = compute_threshold_metrics(y_true, y_scores, thresholds=[0.0])
        assert df_zero.loc[0, "Recall"] == 1.0

    def test_confusion_matrix(self, mock_data):
        y_true, y_scores = mock_data
        y_pred = (y_scores >= 0.5).astype(int)
        cm = compute_confusion_matrix(y_true, y_pred, labels=["NotSuicidal", "Suicidal"])
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)

    def test_plot_confusion_matrix(self, mock_data):
        y_true, y_scores = mock_data
        y_pred = (y_scores >= 0.5).astype(int)
        cm = compute_confusion_matrix(y_true, y_pred, labels=["NotSuicidal", "Suicidal"])
        fig = plot_confusion_matrix(cm, labels=["NotSuicidal", "Suicidal"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@pytest.mark.slow
class TestEndToEnd:
    def test_run_all(self, suicide_df, tmp_output_dir):
        csv_path = tmp_output_dir / "test_suicide.csv"
        suicide_df.to_csv(csv_path, index=False)

        from notebooks.model_evaluation import run_all
        out_dir = tmp_output_dir / "eval_out"
        outputs = run_all(
            dataset_path=csv_path,
            output_dir=out_dir,
            model_path="prajjwal1/bert-tiny",
            sample_size=10
        )

        assert (out_dir / "precision_recall_curve.png").exists()
        assert (out_dir / "calibration_curve.png").exists()
        assert (out_dir / "threshold_sensitivity.csv").exists()
        assert (out_dir / "confusion_matrix.png").exists()
        assert (out_dir / "classification_report.csv").exists()
