"""Tests for notebooks/topic_modeling.py"""

import pandas as pd
import pytest
from pathlib import Path
from notebooks.topic_modeling import (
    preprocess_text,
    preprocess_corpus,
    train_lda,
    get_topic_terms,
    plot_topic_terms,
    generate_pyldavis_html,
    ALL_STOPWORDS,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class TestPreprocessing:
    def test_preprocess_text(self):
        text = "Hello world! This is a test. https://example.com"
        tokens = preprocess_text(text)
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "https" not in tokens
        assert "this" not in tokens # Stopword
        assert "is" not in tokens # Stopword

    def test_preprocess_corpus(self):
        corpus = pd.Series(["I want to die", "Feeling so sad and lonely"])
        tokenized = preprocess_corpus(corpus)
        assert len(tokenized) == 2
        assert "die" in tokenized[0]
        assert "want" not in tokenized[0] # EXTRA_STOPWORD
        assert "feeling" in tokenized[1]
        assert "sad" in tokenized[1]


class TestLDAModeling:
    def test_lda_training(self):
        docs = [
            ["sad", "lonely", "depressed"],
            ["money", "debt", "financial", "ruin"],
            ["breakup", "relationship", "alone"],
            ["sad", "lonely", "alone"],
        ] * 10 # Repeat to have enough vocab for extremes filter
        lda, dictionary, corpus = train_lda(docs, num_topics=2, passes=2)
        assert lda.num_topics == 2
        assert len(dictionary) > 0

    def test_get_topic_terms(self):
        docs = [
            ["sad", "lonely", "depressed"],
            ["money", "debt", "financial", "ruin"],
        ] * 30
        lda, _, _ = train_lda(docs, num_topics=2, passes=2)
        topics = get_topic_terms(lda, num_words=2)
        assert len(topics) == 2
        assert len(topics[0]) == 2
        assert isinstance(topics[0][0][0], str)
        assert isinstance(topics[0][0][1], float)


class TestVisualization:
    def test_plot_topic_terms(self):
        topics = {
            0: [("sad", 0.5), ("lonely", 0.3)],
            1: [("money", 0.6), ("debt", 0.2)]
        }
        fig = plot_topic_terms(topics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_generate_pyldavis(self, tmp_path):
        docs = [
            ["sad", "lonely", "depressed"],
            ["money", "debt", "financial", "ruin"],
        ] * 30
        lda, dictionary, corpus = train_lda(docs, num_topics=2, passes=2)
        out_path = tmp_path / "vis.html"
        res = generate_pyldavis_html(lda, corpus, dictionary, out_path)
        # It may return False if pyLDAvis is not installed, but if True, file should exist
        if res:
            assert out_path.exists()


class TestEndToEnd:
    def test_run_all(self, suicide_df, tmp_output_dir):
        csv_path = tmp_output_dir / "test_suicide.csv"
        # Duplicate df so term frequency > 20
        df_large = pd.concat([suicide_df] * 30, ignore_index=True)
        df_large.to_csv(csv_path, index=False)

        from notebooks.topic_modeling import run_all
        out_dir = tmp_output_dir / "tm_out"
        # The test dataframe is small, so we might need a smaller passes/topics or it could fail the filter_extremes
        # Just test that it runs without crashing, we'll try 2 topics
        outputs = run_all(dataset_path=csv_path, output_dir=out_dir, num_topics=2)

        assert (out_dir / "topic_terms.png").exists()
        assert (out_dir / "topic_terms.csv").exists()
