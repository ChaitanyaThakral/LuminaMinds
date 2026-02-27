"""Tests for Suicide Watch EDA processing functions."""

import pandas as pd
import numpy as np
import pytest

from analytics.data_processing import (
    compute_post_lengths,
    compute_jaccard_similarity,
    _word_ratio,
    compute_pronoun_ratio,
    compute_absolutist_density,
    compute_negation_frequency,
    cohens_d,
    run_psycholinguistic_analysis,
    compute_flesch_kincaid,
    _count_syllables,
    readability_by_class,
)


class TestPostLength:
    def test_post_length_added(self, suicide_df):
        res = compute_post_lengths(suicide_df)
        assert "length" in res.columns
        assert (res["length"] > 0).all()


class TestVocabularyOverlap:
    def test_jaccard_similarity(self):
        df = pd.DataFrame({
            "text": ["hello world", "hello friend"],
            "class": ["a", "b"]
        })
        sim = compute_jaccard_similarity(df)
        assert sim == pytest.approx(1/3, 0.01)

    def test_jaccard_on_fixture(self, suicide_df):
        sim = compute_jaccard_similarity(suicide_df)
        assert 0.0 <= sim <= 1.0


class TestPsycholinguisticMarkers:
    def test_word_ratio(self):
        ratio = _word_ratio("I am my own hero", {"i", "my"})
        assert ratio == 2 / 5

    def test_compute_pronoun_ratio(self):
        texts = pd.Series(["I me my", "hello world"])
        res = compute_pronoun_ratio(texts)
        assert res[0] == 1.0
        assert res[1] == 0.0

    def test_compute_absolutist_density(self):
        texts = pd.Series(["I always never give up", "maybe tomorrow"])
        res = compute_absolutist_density(texts)
        assert abs(res[0] - 0.4) < 1e-6
        assert res[1] == 0.0

    def test_cohens_d(self):
        g1 = np.array([1, 2, 3, 4, 5])
        g2 = np.array([6, 7, 8, 9, 10])
        d = cohens_d(g1, g2)
        assert d < 0 

        g1 = np.array([1, 1, 1])
        g2 = np.array([1, 1, 1])
        assert cohens_d(g1, g2) == 0.0

    def test_run_analysis(self, suicide_df):
        res = run_psycholinguistic_analysis(suicide_df)
        assert not res.empty
        assert "Marker" in res.columns
        assert "p-value" in res.columns


class TestFleschKincaid:
    def test_count_syllables(self):
        assert _count_syllables("hello") == 2
        assert _count_syllables("world") == 1
        assert _count_syllables("beautiful") == 3

    def test_compute_fk(self):
        score = compute_flesch_kincaid("The quick brown fox jumps over the lazy dog.")
        assert isinstance(score, float)

    def test_readability_by_class(self, suicide_df):
        res = readability_by_class(suicide_df)
        assert not res.empty
        assert "Mean FK Grade" in res.columns
