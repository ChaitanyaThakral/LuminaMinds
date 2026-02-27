"""Tests for mood_classification.py (4-class mental health mood classifier)."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mood_df(tmp_path) -> tuple[list[str], np.ndarray, LabelEncoder]:
    """Small synthetic mental health dataset with 4 mood classes."""
    from analytics.data_processing import MOOD_4_CLASS_MAP

    texts = []
    statuses = []
    for orig, mood in MOOD_4_CLASS_MAP.items():
        for i in range(10):
            texts.append(f"Sample {orig} text number {i} with some words about feelings")
            statuses.append(mood)

    df = pd.DataFrame({"text": texts, "mood_class": statuses})
    csv_path = tmp_path / "mental_health_corpus.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadAndPrepare:
    def test_returns_texts_labels_encoder(self, mood_df, tmp_path):
        from experiments.mood_classification import load_and_prepare
        csv_path, df = mood_df

        # Rebuild in the expected schema (with 'statement' and 'status' cols)
        df2 = pd.DataFrame({
            "statement": df["text"],
            "status": [  # Map back to original 7-class labels for load_mental_health_corpus
                next(k for k, v in __import__("analytics.data_processing",
                    fromlist=["MOOD_4_CLASS_MAP"]).MOOD_4_CLASS_MAP.items()
                    if v == row) for row in df["mood_class"]
            ],
        })
        csv_path2 = tmp_path / "corpus.csv"
        df2.to_csv(csv_path2, index=False)

        texts, labels, le = load_and_prepare(dataset_path=csv_path2)
        assert len(texts) == len(labels)
        assert len(le.classes_) == 4

    def test_sample_size_respected(self, mood_df, tmp_path):
        from experiments.mood_classification import load_and_prepare

        # Build proper schema
        _, df = mood_df
        from analytics.data_processing import MOOD_4_CLASS_MAP
        df2 = pd.DataFrame({
            "statement": df["text"],
            "status": [
                next(k for k, v in MOOD_4_CLASS_MAP.items() if v == row)
                for row in df["mood_class"]
            ],
        })
        csv_path2 = tmp_path / "corpus_small.csv"
        df2.to_csv(csv_path2, index=False)

        texts, labels, le = load_and_prepare(dataset_path=csv_path2, sample_size=20)
        assert len(texts) <= 20


class TestTrainMoodClassifier:
    def test_returns_macro_f1(self):
        from experiments.mood_classification import train_mood_classifier

        # Simple synthetic data with 4 classes, enough to split
        rng = np.random.RandomState(0)
        classes = ["Normal", "Depression/Suicidal", "Anxiety/Stress", "Bipolar/Disorder"]
        texts = [f"text about {c} sample {i}" for c in classes for i in range(20)]
        le = LabelEncoder()
        labels = le.fit_transform([c for c in classes for _ in range(20)])

        result = train_mood_classifier(texts, labels, max_features=100)
        assert "Macro-F1" in result["metrics"]
        assert 0.0 <= result["metrics"]["Macro-F1"] <= 1.0

    def test_result_has_model_and_vectorizer(self):
        from experiments.mood_classification import train_mood_classifier

        classes = ["Normal", "Depression/Suicidal", "Anxiety/Stress", "Bipolar/Disorder"]
        texts = [f"text sample {c} {i}" for c in classes for i in range(20)]
        le = LabelEncoder()
        labels = le.fit_transform([c for c in classes for _ in range(20)])

        result = train_mood_classifier(texts, labels, max_features=100)
        assert "model" in result
        assert "vectorizer" in result
        assert "y_test" in result
        assert "y_pred" in result
