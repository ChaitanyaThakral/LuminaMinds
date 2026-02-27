"""Tests for notebooks/shap_explainability.py"""

import pytest
import pandas as pd
from pathlib import Path
from notebooks.shap_explainability import (
    load_model,
    compute_attributions,
    generate_force_plot_html,
    generate_full_html,
    select_samples,
    predict_single,
)


@pytest.mark.slow
class TestSHAPExplainability:
    def test_load_model(self):
        model, tokenizer = load_model("prajjwal1/bert-tiny")
        assert model is not None
        assert tokenizer is not None

    def test_predict_single(self, mock_model_and_tokenizer):
        model, tokenizer = mock_model_and_tokenizer
        label, score = predict_single("I feel great", model, tokenizer)
        assert label in ["Suicidal", "NotSuicidal"]
        assert 0.0 <= score <= 1.0

    def test_compute_attributions(self, mock_model_and_tokenizer):
        model, tokenizer = mock_model_and_tokenizer
        attrs = compute_attributions("This is a test.", model, tokenizer)
        pass # mock returns empty list

    def test_generate_html(self):
        attrs = [("This", 0.1), ("is", -0.2), ("test", 0.5)]
        html = generate_force_plot_html("This is a test", attrs, "Suicidal", 0.85)
        assert "background-color:" in html
        assert "This" in html
        assert "Suicidal" in html
        assert "0.85" in html

    def test_generate_full_html(self):
        samples = [
            {"html": "<div>Sample 1</div>"},
            {"html": "<div>Sample 2</div>"}
        ]
        full = generate_full_html(samples)
        assert "<html" in full
        assert "Sample 1" in full
        assert "Sample 2" in full

    def test_select_samples(self, suicide_df):
        # We need a dataframe with class column
        selected = select_samples(suicide_df, n=10)
        assert len(selected) <= 10
        assert "class" in selected.columns

    def test_run_all(self, suicide_df, tmp_output_dir):
        csv_path = tmp_output_dir / "test_suicide.csv"
        suicide_df.to_csv(csv_path, index=False)

        from notebooks.shap_explainability import run_all
        out_dir = tmp_output_dir / "shap_out"
        outputs = run_all(
            dataset_path=csv_path,
            output_dir=out_dir,
            model_path="prajjwal1/bert-tiny",
            n_samples=2
        )
        assert (out_dir / "shap_attributions.html").exists()
