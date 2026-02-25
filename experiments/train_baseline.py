"""
A/B Comparison: Train a logistic regression baseline on TF-IDF,
compare to DeBERTa on the Suicide Detection dataset.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.calibration import calibration_curve

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "suicide_watch.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
PROXY_MODEL = "prajjwal1/bert-tiny"


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression Baseline
# ---------------------------------------------------------------------------

def train_tfidf_baseline(
    texts: list[str],
    labels: np.ndarray,
    max_features: int = 50000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train TF-IDF + Logistic Regression baseline.
    Returns dict with model, vectorizer, metrics, and timing.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Fit TF-IDF
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train logistic regression
    t0 = time.time()
    lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=random_state,
        n_jobs=-1,
    )
    lr.fit(X_train_tfidf, y_train)
    train_time = time.time() - t0

    # Predict
    y_pred = lr.predict(X_test_tfidf)
    y_proba = lr.predict_proba(X_test_tfidf)[:, 1]

    # Inference timing
    t0 = time.time()
    for _ in range(100):
        lr.predict_proba(X_test_tfidf[:1])
    inference_ms = (time.time() - t0) / 100 * 1000

    # Calibration error (ECE)
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")
    ece = np.mean(np.abs(frac_pos - mean_pred))

    metrics = {
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "Inference (ms/sample)": round(inference_ms, 3),
        "ECE": round(ece, 4),
        "Train Time (s)": round(train_time, 2),
    }

    return {
        "model": lr,
        "vectorizer": tfidf,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "X_test": X_test,
    }


# ---------------------------------------------------------------------------
# DeBERTa (proxy) evaluation
# ---------------------------------------------------------------------------

def evaluate_deberta_proxy(
    texts: list[str],
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: str | None = None,
) -> dict:
    """Evaluate DeBERTa (or proxy) model on the same test set."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Load model
    path = model_path or PROXY_MODEL
    risk_path = BASE_DIR / "models" / "risk_model"
    if risk_path.exists() and model_path is None:
        path = str(risk_path)

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=2, ignore_mismatched_sizes=True
    )
    model.eval()

    # Predict in batches
    y_proba = []
    batch_size = 32
    t0 = time.time()
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        suicidal_idx = min(1, probs.shape[-1] - 1)
        y_proba.extend(probs[:, suicidal_idx].numpy().tolist())
    total_time = time.time() - t0
    inference_ms = total_time / len(X_test) * 1000

    y_proba = np.array(y_proba)
    y_pred = (y_proba >= 0.5).astype(int)

    # Calibration error
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")
    ece = np.mean(np.abs(frac_pos - mean_pred))

    metrics = {
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "Inference (ms/sample)": round(inference_ms, 3),
        "ECE": round(ece, 4),
    }

    return {"metrics": metrics, "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba}


# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------

def save_baseline_model(model, vectorizer, path: Path | None = None) -> Path:
    """Save the baseline model + vectorizer as pickle."""
    out_path = path or (OUTPUT_DIR / "baseline_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
    return out_path


def load_baseline_model(path: Path | None = None):
    """Load the saved baseline model."""
    load_path = path or (OUTPUT_DIR / "baseline_model.pkl")
    with open(load_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["vectorizer"]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    dataset_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    sample_size: int | None = 20000,
) -> dict:
    """Train baseline, evaluate both models, produce comparison."""
    ds_path = Path(dataset_path) if dataset_path else DATASET_PATH
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Baseline] Loading dataset from {ds_path} ...")
    df = pd.read_csv(ds_path)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    texts = df["text"].astype(str).tolist()
    labels = (df["class"] == "suicide").astype(int).values

    # Train baseline
    print("[Baseline] Training TF-IDF + LogReg ...")
    baseline_result = train_tfidf_baseline(texts, labels)
    print(f"  Baseline metrics: {baseline_result['metrics']}")

    # Save model
    model_path = save_baseline_model(baseline_result["model"], baseline_result["vectorizer"],
                                      out_dir / "baseline_model.pkl")
    print(f"  Model saved: {model_path}")

    # Evaluate DeBERTa (proxy)
    print("[DeBERTa] Evaluating proxy model ...")
    deberta_result = evaluate_deberta_proxy(texts, labels)
    print(f"  DeBERTa metrics: {deberta_result['metrics']}")

    # Comparison table
    comparison = pd.DataFrame({
        "Metric": list(baseline_result["metrics"].keys()),
        "TF-IDF + LogReg": list(baseline_result["metrics"].values()),
        "DeBERTa (proxy)": [deberta_result["metrics"].get(k, "N/A")
                            for k in baseline_result["metrics"].keys()],
    })
    comparison.to_csv(out_dir / "comparison_table.csv", index=False)
    print("\n" + comparison.to_string(index=False))

    return {
        "baseline_metrics": baseline_result["metrics"],
        "deberta_metrics": deberta_result["metrics"],
        "comparison_csv": str(out_dir / "comparison_table.csv"),
        "model_path": str(model_path),
    }


if __name__ == "__main__":
    run_all()
