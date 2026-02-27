from __future__ import annotations
"""
Shared pytest fixtures for LuminaMind test suite.

Provides small synthetic DataFrames and mocks so tests
don't depend on the full 53K/232K CSV datasets.
"""

import sys
import torch
from unittest.mock import MagicMock

# --- GLOBAL TRANSFORMERS MOCK ---
mock_hf = MagicMock()
mock_tokenizer = MagicMock()

def _dynamic_tokenizer(*args, **kwargs):
    """Return input_ids sized to the actual input batch."""
    texts = args[0] if args else ["dummy"]
    bsz = len(texts) if isinstance(texts, (list, tuple)) else 1
    return {
        "input_ids": torch.ones((bsz, 3), dtype=torch.long),
        "attention_mask": torch.ones((bsz, 3), dtype=torch.long),
    }

mock_tokenizer.side_effect = _dynamic_tokenizer
mock_tokenizer.convert_ids_to_tokens.return_value = ["he", "##llo", "world"]
mock_hf.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

mock_model = MagicMock()
class _GlobalMockOutput:
    def __init__(self, bsz: int):
        self.logits = torch.full((bsz, 2), 0.0)
        self.logits[:, 1] = 0.8  # second class score

def _mock_model_call(**kwargs):
    bsz = kwargs.get('input_ids', torch.tensor([[1]])).shape[0]
    return _GlobalMockOutput(bsz)

mock_model.side_effect = _mock_model_call
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
# Mental Health Corpus fixture (80 rows, 4 mood classes)
# ---------------------------------------------------------------------------

@pytest.fixture
def mental_health_df() -> pd.DataFrame:
    """Synthetic mental health corpus DataFrame with 80 rows across 4 mood classes."""
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
        for t in txts * 4:  # 20 per class = 80 total
            rows.append({"text": t, "mood_class": mood, "status": mood})
    df = pd.DataFrame(rows)
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
