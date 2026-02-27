"""Tests for dashboard/utils.py"""

import pytest
import numpy as np
import pandas as pd
from dashboard.utils import (
    load_dataset_stats,
    compute_threshold_metrics,
    token_highlight_html,
    inference_wrapper,
)

class TestDashboardUtils:
    def test_compute_threshold_metrics(self):
        y_true = np.array([0, 0, 1, 1, 0])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9, 0.8])
        
        metrics = compute_threshold_metrics(y_true, y_scores, threshold=0.5)
        # tp=2, fn=0, fp=1, tn=2
        assert metrics["tp"] == 2
        assert metrics["fp"] == 1
        assert metrics["precision"] == pytest.approx(2/3, 0.01)
        assert metrics["recall"] == 1.0

        metrics_high = compute_threshold_metrics(y_true, y_scores, threshold=0.85)
        # tp=1 (score 0.9), fn=1 (score 0.6)
        assert metrics_high["tp"] == 1
        assert metrics_high["fn"] == 1
        assert metrics_high["recall"] == 0.5

    def test_token_highlight_html(self):
        tokens = ["hello", "world", "[PAD]"]
        scores = [0.5, -0.2, 0.0]
        html = token_highlight_html(tokens, scores)
        assert "hello" in html
        assert "world" in html
        assert "[PAD]" not in html
        assert "rgba(231, 76, 60" in html # red for positive
        assert "rgba(52, 152, 219" in html # blue for negative

    def test_inference_wrapper(self, mock_model_and_tokenizer):
        model, tokenizer = mock_model_and_tokenizer
        res = inference_wrapper("testing", model, tokenizer)
        assert "label" in res
        assert "score" in res
        assert "probs" in res
        assert "Suicidal" in res["probs"]
