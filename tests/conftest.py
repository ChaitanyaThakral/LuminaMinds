from __future__ import annotations
"""
Shared pytest fixtures for LuminaMind test suite.

Provides small synthetic DataFrames and mocks so tests
don't depend on the full 1.6M/232K CSV datasets.
"""

import sys
import torch
from unittest.mock import MagicMock

# --- GLOBAL TRANSFORMERS MOCK ---
mock_hf = MagicMock()
mock_tokenizer = MagicMock()
mock_tokenizer.return_value = {"input_ids": torch.tensor([[1,2,3]]), "attention_mask": torch.tensor([[1,1,1]])}
mock_tokenizer.convert_ids_to_tokens.return_value = ["he", "##llo", "world"]
mock_hf.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

mock_model = MagicMock()
class MockOutput:
    logits = torch.tensor([[0.2, 0.8]])
mock_model.return_value = MockOutput()
mock_model.eval = MagicMock()
mock_hf.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model

sys.modules['transformers'] = mock_hf
sys.modules['transformers_interpret'] = MagicMock()
# --------------------------------



import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Sentiment140 fixture (50 rows)
# ---------------------------------------------------------------------------

@pytest.fixture
def sentiment140_df() -> pd.DataFrame:
    """Synthetic Sentiment140 DataFrame with 50 rows."""
    rng = np.random.RandomState(42)
    n = 50
    dates_pool = [
        "Mon Apr 06 22:19:45 PDT 2009",
        "Tue Apr 07 08:30:00 PDT 2009",
        "Wed Apr 08 14:45:30 PDT 2009",
        "Thu Apr 09 03:12:00 PDT 2009",
        "Fri Apr 10 18:00:00 PDT 2009",
        "Sat Apr 11 09:00:00 PDT 2009",
        "Sun Apr 12 21:30:00 PDT 2009",
    ]

    neg_texts = [
        "I hate this so much, it makes me sad :(",
        "worst day ever can't believe it happened!!",
        "feeling down and depressed today...",
        "this is NOT okay, really upset right now",
        "terrible experience https://example.com/bad",
        "I'm so angry I could scream!!!",
        "nothing ever works out for me",
        "why does everything go wrong always",
        "can't stop crying, this is awful",
        "HATE this stupid thing so much",
    ]

    pos_texts = [
        "what a beautiful day! love it :)",
        "so happy and grateful today!!!",
        "amazing experience, highly recommend",
        "feeling great, life is wonderful",
        "best day ever, couldn't be happier",
        "love spending time with friends :D",
        "everything is going perfectly today",
        "such a lovely surprise, thank you!",
        "incredible news, so excited!!",
        "wonderful day at the park, blessed",
    ]

    targets = [0] * 25 + [4] * 25
    texts = [neg_texts[i % len(neg_texts)] for i in range(25)] + \
            [pos_texts[i % len(pos_texts)] for i in range(25)]
    dates = [dates_pool[i % len(dates_pool)] for i in range(n)]

    df = pd.DataFrame({
        "target": targets,
        "id": range(1000, 1000 + n),
        "date": dates,
        "flag": "NO_QUERY",
        "user": [f"user_{i}" for i in range(n)],
        "text": texts,
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Suicide Watch fixture (50 rows)
# ---------------------------------------------------------------------------

@pytest.fixture
def suicide_df() -> pd.DataFrame:
    """Synthetic Suicide Detection DataFrame with 50 rows."""
    suicide_texts = [
        "I want to end it all, there is no point in living anymore",
        "I have been thinking about killing myself every day",
        "nobody cares about me, I am completely alone and nothing matters",
        "I can't take this pain anymore, I always feel hopeless",
        "my life is never going to get better, I want to die",
        "I feel so empty inside, everything is pointless and I just want it to stop",
        "I've been planning how to do it, I don't see any other way",
        "the world would be better without me in it",
        "I tried to hurt myself last night",
        "I am a burden to everyone around me, they'd be happier if I was gone",
    ]

    non_suicide_texts = [
        "Had a great day at work today, feeling productive",
        "Just finished reading an amazing book, highly recommend it",
        "Going to the gym later, trying to stay healthy",
        "Movie night with friends this weekend, can't wait",
        "Learning to cook new recipes, it's actually fun",
        "The weather is beautiful today, perfect for a walk",
        "Got a promotion at work, feeling grateful",
        "Spent time with family today, it was lovely",
        "Trying meditation for the first time, very calming",
        "Adopted a puppy today, so happy!",
    ]

    texts = [suicide_texts[i % len(suicide_texts)] for i in range(25)] + \
            [non_suicide_texts[i % len(non_suicide_texts)] for i in range(25)]
    classes = ["suicide"] * 25 + ["non-suicide"] * 25

    df = pd.DataFrame({"text": texts, "class": classes})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Session events fixture (100 rows)
# ---------------------------------------------------------------------------

@pytest.fixture
def session_events_df() -> pd.DataFrame:
    """Synthetic session events DataFrame with 100 rows."""
    from analytics.generate_session_data import generate_session_data
    return generate_session_data(n_sessions=100, random_state=42)


# ---------------------------------------------------------------------------
# Temporary output directory
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    out = tmp_path / "test_outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Mock model and tokenizer
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model_and_tokenizer():
    """Load a completely mocked model and tokenizer for testing."""
    from unittest.mock import MagicMock
    import torch
    
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    tokenizer.convert_ids_to_tokens.return_value = ["he", "##llo", "world"]

    model = MagicMock()
    class MockOutput:
        def __init__(self, kwargs):
            bsz = kwargs.get('input_ids', torch.tensor([[1]])).shape[0]
            self.logits = torch.randn(bsz, 2)
            
    model.side_effect = lambda **kwargs: MockOutput(kwargs)
    model.eval = MagicMock()
    
    return model, tokenizer
