"""
Model Evaluation Report.

Generates:
  1. Precision-recall curve
  2. Calibration curve
  3. Threshold sensitivity table (0.3, 0.5, 0.7)
  4. Confusion matrix with class-wise breakdown
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MOOD_MODEL_PATH = BASE_DIR / "models" / "mood4_model"
RISK_MODEL_PATH = BASE_DIR / "models" / "risk_model"
PROXY_MODEL = "prajjwal1/bert-tiny"
DATASET_PATH = BASE_DIR / "dataset" / "suicide_watch.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "evaluation"
RISK_LABELS = ["NotSuicidal", "Suicidal"]
MOOD_LABELS = ["Anxiety", "Depression", "Normal", "Stress"]


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------

def load_risk_model(model_path: Path | str | None = None):
    """Load risk model + tokenizer."""
    path = model_path or (RISK_MODEL_PATH if RISK_MODEL_PATH.exists() else PROXY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(path), num_labels=len(RISK_LABELS), ignore_mismatched_sizes=True
    )
    model.eval()
    return model, tokenizer


def predict_batch(texts: list[str], model, tokenizer, batch_size: int = 32) -> np.ndarray:
    """Run batch inference, return array of predicted probabilities for class 1 (Suicidal)."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        # Index 1 = Suicidal
        suicidal_idx = min(1, probs.shape[-1] - 1)
        all_probs.extend(probs[:, suicidal_idx].numpy().tolist())
    return np.array(all_probs)


# ---------------------------------------------------------------------------
# 1. Precision-Recall curve
# ---------------------------------------------------------------------------

def compute_precision_recall(y_true: np.ndarray, y_scores: np.ndarray):
    """Compute precision-recall curve data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    return precision, recall, thresholds, ap


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: float,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot precision-recall curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#e74c3c", linewidth=2, label=f"AP = {ap:.3f}")
    ax.fill_between(recall, precision, alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve (Risk Detection)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Calibration curve
# ---------------------------------------------------------------------------

def compute_calibration(y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 10):
    """Compute calibration curve data."""
    fraction_pos, mean_predicted = calibration_curve(y_true, y_scores, n_bins=n_bins, strategy="uniform")
    return fraction_pos, mean_predicted


def plot_calibration_curve(
    fraction_pos: np.ndarray,
    mean_predicted: np.ndarray,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot calibration (reliability) diagram."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#bdc3c7", label="Perfectly calibrated", linewidth=1.5)
    ax.plot(mean_predicted, fraction_pos, "o-", color="#3498db", linewidth=2, markersize=8, label="Model")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Threshold sensitivity table
# ---------------------------------------------------------------------------

def compute_threshold_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute precision, recall, F1, and FN rate at various thresholds."""
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    rows = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        fn_rate = fn / max(fn + tp, 1)

        rows.append({
            "Threshold": t,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "FN Rate": round(fn_rate, 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: Path | None = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot confusion matrix heatmap using matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=12)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    dataset_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    model_path: Path | str | None = None,
    sample_size: int | None = 5000,
) -> dict:
    """Run full model evaluation and save outputs."""
    ds_path = Path(dataset_path) if dataset_path else DATASET_PATH
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Model Eval] Loading model ...")
    model, tokenizer = load_risk_model(model_path)

    print(f"[Model Eval] Loading dataset from {ds_path} ...")
    df = pd.read_csv(ds_path)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    y_true = (df["class"] == "suicide").astype(int).values
    texts = df["text"].astype(str).tolist()

    print(f"[Model Eval] Running inference on {len(texts)} samples ...")
    y_scores = predict_batch(texts, model, tokenizer)

    outputs = {}

    # 1. Precision-recall curve
    print("[1/4] Precision-recall curve ...")
    prec, rec, pr_thresholds, ap = compute_precision_recall(y_true, y_scores)
    path = out_dir / "precision_recall_curve.png"
    fig = plot_precision_recall_curve(prec, rec, ap, path)
    plt.close(fig)
    outputs["precision_recall_curve"] = str(path)

    # 2. Calibration curve
    print("[2/4] Calibration curve ...")
    frac_pos, mean_pred = compute_calibration(y_true, y_scores)
    path = out_dir / "calibration_curve.png"
    fig = plot_calibration_curve(frac_pos, mean_pred, path)
    plt.close(fig)
    outputs["calibration_curve"] = str(path)

    # 3. Threshold sensitivity
    print("[3/4] Threshold sensitivity table ...")
    threshold_df = compute_threshold_metrics(y_true, y_scores)
    path = out_dir / "threshold_sensitivity.csv"
    threshold_df.to_csv(path, index=False)
    print(threshold_df.to_string(index=False))
    outputs["threshold_sensitivity"] = str(path)

    # 4. Confusion matrix (at threshold 0.5)
    print("[4/4] Confusion matrix ...")
    y_pred = (y_scores >= 0.5).astype(int)
    cm = compute_confusion_matrix(y_true, y_pred, RISK_LABELS)
    path = out_dir / "confusion_matrix.png"
    fig = plot_confusion_matrix(cm, RISK_LABELS, path, "Risk Detection — Confusion Matrix")
    plt.close(fig)
    outputs["confusion_matrix"] = str(path)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=RISK_LABELS, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / "classification_report.csv")
    outputs["classification_report"] = str(out_dir / "classification_report.csv")

    print(f"[Model Eval] Done. Outputs in {out_dir}")
    return outputs


if __name__ == "__main__":
    run_all()
