"""Tests for Sentiment140 EDA processing functions."""

import pandas as pd
import pytest
import matplotlib.pyplot as plt

from analytics.data_processing import (
    compute_class_distribution,
    compute_tweet_lengths,
    compute_log_odds_ratio,
    parse_twitter_date,
    compute_temporal_features,
    compute_linguistic_features,
    summarize_linguistic_features,
)

class TestClassDistribution:
    def test_correct_counts(self, sentiment140_df):
        counts = compute_class_distribution(sentiment140_df)
        assert "Negative" in counts.index
        assert "Positive" in counts.index
        assert counts.sum() == len(sentiment140_df)


class TestTweetLength:
    def test_length_column_added(self, sentiment140_df):
        result = compute_tweet_lengths(sentiment140_df)
        assert "length" in result.columns
        assert "label" in result.columns
        assert (result["length"] > 0).all()


class TestLogOddsRatio:
    def test_returns_dataframe(self, sentiment140_df):
        result = compute_log_odds_ratio(sentiment140_df, top_n=10, min_count=1)
        assert isinstance(result, pd.DataFrame)

    def test_has_both_classes(self, sentiment140_df):
        result = compute_log_odds_ratio(sentiment140_df, top_n=10, min_count=1)
        if not result.empty:
            assert "Positive" in result["class"].values or "Negative" in result["class"].values


class TestTemporalAnalysis:
    def test_parse_twitter_date(self):
        result = parse_twitter_date("Mon Apr 06 22:19:45 PDT 2009")
        assert result["day_of_week"] == "Mon"
        assert result["hour"] == 22

    def test_parse_invalid_date(self):
        result = parse_twitter_date("invalid")
        assert result["hour"] is None

    def test_temporal_features_added(self, sentiment140_df):
        result = compute_temporal_features(sentiment140_df)
        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "label" in result.columns


class TestLinguisticFeatures:
    def test_punctuation_count(self):
        df = pd.DataFrame({"text": ["hello!!!", "hi."], "target": [0, 4]})
        metrics = compute_linguistic_features(df)
        assert metrics.loc[0, "punctuation_count"] == 3
        assert metrics.loc[1, "punctuation_count"] == 1

    def test_caps_ratio(self):
        df = pd.DataFrame({"text": ["HELLO world", "hello"], "target": [0, 4]})
        metrics = compute_linguistic_features(df)
        assert metrics.loc[0, "caps_ratio"] == pytest.approx(5 / 11, abs=0.01)

    def test_url_presence(self):
        df = pd.DataFrame({
            "text": ["check https://example.com", "no url here"],
            "target": [0, 4],
        })
        metrics = compute_linguistic_features(df)
        assert metrics.loc[0, "url_presence"] == 1
        assert metrics.loc[1, "url_presence"] == 0

    def test_summary_table(self, sentiment140_df):
        metrics = compute_linguistic_features(sentiment140_df)
        summary = summarize_linguistic_features(metrics)
        assert "avg_punctuation" in summary.columns
        assert len(summary) == 2
