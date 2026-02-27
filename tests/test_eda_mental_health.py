"""Tests for Mental Health Corpus EDA processing functions.

Covers the 4-class mood classification analytics that replaced
the old binary Sentiment140 analysis.
"""

import pandas as pd
import numpy as np
import pytest

from analytics.data_processing import (
    MOOD_4_CLASS_MAP,
    MOOD_CLASSES,
    compute_class_distribution,
    compute_text_lengths,
    compute_log_odds_ratio,
    compute_linguistic_features,
    summarize_linguistic_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mental_health_df() -> pd.DataFrame:
    """Synthetic 4-class mental health DataFrame (80 rows)."""
    texts = {
        "Normal": [
            "Had a great day, feeling productive and happy.",
            "Went for a walk in the park, lovely weather today.",
            "Cooked a new recipe, turned out amazing!",
            "Spent time with family, feeling grateful.",
            "Reading a good book, really enjoying it.",
        ],
        "Depression/Suicidal": [
            "I can't take this pain anymore, everything feels hopeless.",
            "Nobody cares, I just want it all to stop.",
            "I've been crying for days, nothing helps.",
            "I feel completely empty and alone inside.",
            "There is no point in going on anymore.",
        ],
        "Anxiety/Stress": [
            "I'm so stressed about the exam, can't sleep at all.",
            "My heart won't stop racing, I'm so anxious.",
            "Overwhelmed by everything, can't focus on anything.",
            "The deadline is tomorrow and I'm panicking completely.",
            "I keep worrying about things that might never happen.",
        ],
        "Bipolar/Disorder": [
            "I feel amazing today, I could conquer the world!",
            "Yesterday I was on top of the world, today I hate everything.",
            "My mood swings are so unpredictable lately.",
            "I haven't slept in days but I feel so energised.",
            "I can't understand my own feelings anymore.",
        ],
    }
    rows = []
    for mood, txts in texts.items():
        for t in txts * 4:  # 20 per class, 80 total
            rows.append({"text": t, "mood_class": mood, "status": mood})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMoodClassMapping:
    def test_map_coverage(self):
        """All 7 original labels should map to one of the 4 macro classes."""
        original_labels = ["Normal", "Depression", "Suicidal", "Anxiety",
                           "Stress", "Bipolar", "Personality disorder"]
        for lbl in original_labels:
            assert lbl in MOOD_4_CLASS_MAP, f"{lbl} missing from MOOD_4_CLASS_MAP"

    def test_only_4_target_classes(self):
        targets = set(MOOD_4_CLASS_MAP.values())
        assert targets == set(MOOD_CLASSES)

    def test_depression_and_suicidal_merged(self):
        assert MOOD_4_CLASS_MAP["Depression"] == MOOD_4_CLASS_MAP["Suicidal"]

    def test_anxiety_and_stress_merged(self):
        assert MOOD_4_CLASS_MAP["Anxiety"] == MOOD_4_CLASS_MAP["Stress"]


class TestClassDistribution:
    def test_returns_series(self, mental_health_df):
        counts = compute_class_distribution(mental_health_df, label_col="mood_class")
        assert isinstance(counts, pd.Series)

    def test_all_four_classes_present(self, mental_health_df):
        counts = compute_class_distribution(mental_health_df, label_col="mood_class")
        for cls in MOOD_CLASSES:
            assert cls in counts.index, f"Missing class: {cls}"

    def test_total_matches_df(self, mental_health_df):
        counts = compute_class_distribution(mental_health_df, label_col="mood_class")
        assert counts.sum() == len(mental_health_df)


class TestTextLengths:
    def test_length_column_added(self, mental_health_df):
        result = compute_text_lengths(mental_health_df, text_col="text")
        assert "length" in result.columns
        assert (result["length"] > 0).all()

    def test_length_correct(self):
        df = pd.DataFrame({"text": ["hello", "hi there"], "mood_class": ["Normal", "Normal"]})
        result = compute_text_lengths(df)
        assert result.loc[0, "length"] == 5
        assert result.loc[1, "length"] == 8


class TestLogOddsRatio:
    def test_returns_dataframe(self, mental_health_df):
        result = compute_log_odds_ratio(mental_health_df, class_col="mood_class",
                                        text_col="text", top_n=5, min_count=1)
        assert isinstance(result, pd.DataFrame)

    def test_has_log_odds_column(self, mental_health_df):
        result = compute_log_odds_ratio(mental_health_df, class_col="mood_class",
                                        text_col="text", top_n=5, min_count=1)
        if not result.empty:
            assert "log_odds" in result.columns

    def test_empty_df_on_single_class(self):
        df = pd.DataFrame({"text": ["hello", "world"], "mood_class": ["Normal", "Normal"]})
        result = compute_log_odds_ratio(df, class_col="mood_class", text_col="text")
        assert result.empty


class TestLinguisticFeatures:
    def test_punctuation_count(self):
        df = pd.DataFrame({
            "text": ["hello!!!", "hi."],
            "mood_class": ["Normal", "Depression/Suicidal"],
        })
        metrics = compute_linguistic_features(df, label_col="mood_class")
        assert metrics.loc[0, "punctuation_count"] == 3
        assert metrics.loc[1, "punctuation_count"] == 1

    def test_caps_ratio(self):
        df = pd.DataFrame({
            "text": ["HELLO world", "hello"],
            "mood_class": ["Anxiety/Stress", "Normal"],
        })
        metrics = compute_linguistic_features(df, label_col="mood_class")
        assert metrics.loc[0, "caps_ratio"] == pytest.approx(5 / 11, abs=0.01)

    def test_url_presence(self):
        df = pd.DataFrame({
            "text": ["check https://example.com", "no url here"],
            "mood_class": ["Normal", "Normal"],
        })
        metrics = compute_linguistic_features(df, label_col="mood_class")
        assert metrics.loc[0, "url_presence"] == 1
        assert metrics.loc[1, "url_presence"] == 0

    def test_summary_table(self, mental_health_df):
        metrics = compute_linguistic_features(mental_health_df, label_col="mood_class")
        summary = summarize_linguistic_features(metrics)
        assert "avg_punctuation" in summary.columns
        assert len(summary) == 4  # 4 mood classes
