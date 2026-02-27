"""
Shared utilities for the Streamlit dashboard.

Provides data loading, model inference, threshold metric computation,
and token highlight HTML generation.
"""

from __future__ import annotations

import re
import html as html_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUTS_DIR = BASE_DIR / "notebooks" / "outputs"
PROXY_MODEL = "prajjwal1/bert-tiny"
RISK_LABELS = ["NotSuicidal", "Suicidal"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset_stats() -> dict:
    """Load precomputed dataset statistics."""
    stats = {}

    # Mental Health Corpus
    mh_dist_path = OUTPUTS_DIR / "mental_health" / "linguistic_features.csv"
    if mh_dist_path.exists():
        stats["mental_health_linguistic"] = pd.read_csv(mh_dist_path, index_col=0)

    # Suicide Watch
    sw_psych_path = OUTPUTS_DIR / "suicide_watch" / "psycholinguistic_markers.csv"
    if sw_psych_path.exists():
        stats["suicide_watch_markers"] = pd.read_csv(sw_psych_path)

    # Log-odds
    lor_path = OUTPUTS_DIR / "mental_health" / "log_odds_ratio.csv"
    if lor_path.exists():
        stats["log_odds"] = pd.read_csv(lor_path)

    return stats


# ---------------------------------------------------------------------------
# Threshold metrics
# ---------------------------------------------------------------------------

def compute_threshold_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute precision, recall, F1 at a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    fn_rate = fn / max(fn + tp, 1)

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fn_rate": round(fn_rate, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Token highlight HTML
# ---------------------------------------------------------------------------

def token_highlight_html(
    tokens: list[str],
    scores: list[float],
) -> str:
    """
    Generate colored HTML spans for token attributions.
    Positive scores (risk) → red, negative (safe) → blue.
    """
    spans = []
    for token, score in zip(tokens, scores):
        if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"):
            continue
        display = html_module.escape(token.replace("##", ""))
        if score > 0:
            color = f"rgba(231, 76, 60, {min(abs(score), 1):.2f})"
        else:
            color = f"rgba(52, 152, 219, {min(abs(score), 1):.2f})"
        spans.append(
            f'<span style="background-color: {color}; padding: 2px 5px; margin: 1px; '
            f'border-radius: 3px; display: inline-block;" '
            f'title="{display}: {score:+.3f}">{display}</span>'
        )
    return "".join(spans)


# ---------------------------------------------------------------------------
# Model inference wrapper
# ---------------------------------------------------------------------------

def inference_wrapper(text: str, model, tokenizer) -> dict:
    """Run inference on a single text, return prediction dict."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256, padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()

    if probs.dim() == 0:
        probs = probs.unsqueeze(0)

    suicidal_idx = min(1, len(probs) - 1)
    score = float(probs[suicidal_idx])
    label = "Suicidal" if score >= 0.5 else "NotSuicidal"

    return {
        "label": label,
        "score": score,
        "probs": {RISK_LABELS[i]: float(probs[i]) for i in range(len(probs))},
    }
