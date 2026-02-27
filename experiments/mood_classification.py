"""
4-Class Mood Classification: TF-IDF + Logistic Regression baseline
on the mental_health_corpus.csv (53K posts, 7 classes → 4 macro classes).

Resume claim: "0.81 Macro-F1 across 4-class mood classification"
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "mental_health_corpus.csv"
OUTPUT_DIR = Path(__file__).resolve().parent


def load_and_prepare(
    dataset_path: Path | str | None = None,
    sample_size: int | None = None,
    random_state: int = 42,
) -> tuple[list[str], np.ndarray, LabelEncoder]:
    """Load mental_health_corpus.csv, map 7→4 classes, return texts + encoded labels."""
    from analytics.data_processing import load_mental_health_corpus

    path = Path(dataset_path) if dataset_path else DATASET_PATH
    df = load_mental_health_corpus(path)
    df = df.dropna(subset=["text", "mood_class"])

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)

    texts = df["text"].astype(str).tolist()
    le = LabelEncoder()
    labels = le.fit_transform(df["mood_class"].values)
    return texts, labels, le


def train_mood_classifier(
    texts: list[str],
    labels: np.ndarray,
    max_features: int = 50_000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train TF-IDF + Logistic Regression for 4-class mood classification.
    Returns metrics dict including Macro-F1.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=2,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    t0 = time.time()
    lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=random_state,
        multi_class="multinomial",
        n_jobs=-1,
    )
    lr.fit(X_train_tfidf, y_train)
    train_time = time.time() - t0

    y_pred = lr.predict(X_test_tfidf)
    y_proba = lr.predict_proba(X_test_tfidf)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    # Compute one-vs-rest ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float("nan")

    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "Macro-F1": round(macro_f1, 4),
        "Weighted-F1": round(weighted_f1, 4),
        "ROC-AUC (macro OvR)": round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A",
        "Train Time (s)": round(train_time, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return {
        "model": lr,
        "vectorizer": tfidf,
        "metrics": metrics,
        "classification_report": report,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "X_test": X_test,
    }


def run_all(
    dataset_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    sample_size: int | None = None,
) -> dict:
    """Full pipeline: load → train → evaluate → save results."""
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Mood] Loading mental_health_corpus ...")
    texts, labels, le = load_and_prepare(dataset_path, sample_size=sample_size)
    print(f"  {len(texts)} samples, classes: {le.classes_.tolist()}")

    print("[Mood] Training TF-IDF + LogReg (4-class) ...")
    result = train_mood_classifier(texts, labels)
    print(f"  Metrics: {result['metrics']}")

    # Save classification report
    report_df = pd.DataFrame(result["classification_report"]).T
    report_path = out_dir / "mood_classification_report.csv"
    report_df.to_csv(report_path)
    print(f"  Classification report → {report_path}")

    return {
        "metrics": result["metrics"],
        "label_classes": le.classes_.tolist(),
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    run_all()
