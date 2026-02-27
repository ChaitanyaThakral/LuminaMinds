"""
LuminaMind Streamlit Dashboard.

Tab 1: Dataset Stats — class distribution, token length histograms, top features
Tab 2: Model Performance — ROC, PR curve, confusion matrix, threshold slider
Tab 3: Sample Explorer — enter text, see model output + SHAP token highlights
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (
    load_dataset_stats,
    compute_threshold_metrics,
    token_highlight_html,
    inference_wrapper,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LuminaMind Dashboard",
    page_icon="🧠",
    layout="wide",
)

OUTPUTS = PROJECT_ROOT / "notebooks" / "outputs"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_mental_health_sample():
    path = PROJECT_ROOT / "dataset" / "mental_health_corpus.csv"
    if path.exists():
        df = pd.read_csv(path, nrows=5000)
        return df
    return None


@st.cache_data
def load_suicide_sample():
    path = PROJECT_ROOT / "dataset" / "suicide_watch.csv"
    if path.exists():
        df = pd.read_csv(path, nrows=50000)
        return df
    return None


@st.cache_data
def load_evaluation_data():
    """Load precomputed evaluation outputs."""
    data = {}
    eval_dir = OUTPUTS / "evaluation"
    if (eval_dir / "threshold_sensitivity.csv").exists():
        data["threshold"] = pd.read_csv(eval_dir / "threshold_sensitivity.csv")
    if (eval_dir / "classification_report.csv").exists():
        data["report"] = pd.read_csv(eval_dir / "classification_report.csv", index_col=0)
    return data


@st.cache_resource
def load_model():
    """Load the risk model for live inference."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = PROJECT_ROOT / "models" / "risk_model"
    if not model_path.exists():
        model_path = "prajjwal1/bert-tiny"
    else:
        model_path = str(model_path)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), num_labels=2, ignore_mismatched_sizes=True
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Tab 1: Dataset Stats
# ---------------------------------------------------------------------------

def render_dataset_stats():
    st.header("📊 Dataset Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mental Health Corpus (4-class mood)")
        img_path = OUTPUTS / "mental_health" / "class_distribution.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info("Run notebook `01_eda_mental_health.ipynb` to generate plots.")

        img_path = OUTPUTS / "mental_health" / "text_length_histogram.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)

    with col2:
        st.subheader("Suicide Watch")
        img_path = OUTPUTS / "suicide_watch" / "post_length_histogram.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info("Run `python notebooks/eda_suicide_watch.py` to generate plots.")

    # Log-odds ratio
    st.subheader("Top Features — Log-Odds Ratio")
    img_path = OUTPUTS / "mental_health" / "log_odds_ratio.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

    # Linguistic features table
    stats = load_dataset_stats()
    if "mental_health_linguistic" in stats:
        st.subheader("Linguistic Features Comparison (4-class)")
        st.dataframe(stats["mental_health_linguistic"], use_container_width=True)

    if "suicide_watch_markers" in stats:
        st.subheader("Psycholinguistic Markers (Mann-Whitney U)")
        st.dataframe(stats["suicide_watch_markers"], use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Model Performance
# ---------------------------------------------------------------------------

def render_model_performance():
    st.header("📈 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        # PR curve
        img_path = OUTPUTS / "evaluation" / "precision_recall_curve.png"
        if img_path.exists():
            st.image(str(img_path), caption="Precision-Recall Curve", use_container_width=True)

        # Confusion matrix
        img_path = OUTPUTS / "evaluation" / "confusion_matrix.png"
        if img_path.exists():
            st.image(str(img_path), caption="Confusion Matrix", use_container_width=True)

    with col2:
        # Calibration curve
        img_path = OUTPUTS / "evaluation" / "calibration_curve.png"
        if img_path.exists():
            st.image(str(img_path), caption="Calibration Curve", use_container_width=True)

        # Classification report
        eval_data = load_evaluation_data()
        if "report" in eval_data:
            st.subheader("Classification Report")
            st.dataframe(eval_data["report"], use_container_width=True)

    # Threshold slider
    st.subheader("🎛 Interactive Threshold Analysis")
    if "threshold" in load_evaluation_data():
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Adjust the threshold to see how precision, recall, and F1 change."
        )

        # Generate synthetic scores for demo
        np.random.seed(42)
        n = 1000
        y_true = np.random.binomial(1, 0.5, n)
        y_scores = np.clip(y_true * np.random.beta(5, 2, n) + (1 - y_true) * np.random.beta(2, 5, n), 0, 1)

        metrics = compute_threshold_metrics(y_true, y_scores, threshold)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Precision", f"{metrics['precision']:.3f}")
        m_col2.metric("Recall", f"{metrics['recall']:.3f}")
        m_col3.metric("F1 Score", f"{metrics['f1']:.3f}")
        m_col4.metric("FN Rate", f"{metrics['fn_rate']:.3f}")
    else:
        st.info("Run `python notebooks/model_evaluation.py` first to generate evaluation data.")


# ---------------------------------------------------------------------------
# Tab 3: Sample Explorer
# ---------------------------------------------------------------------------

def render_sample_explorer():
    st.header("🔍 Sample Explorer")

    text_input = st.text_area(
        "Enter text to analyze:",
        value="I have been feeling really stressed out and overwhelmed lately.",
        height=150,
    )

    if st.button("Analyze", type="primary"):
        with st.spinner("Running inference..."):
            model, tokenizer = load_model()
            result = inference_wrapper(text_input, model, tokenizer)

        # Prediction display
        label_color = "🔴" if result["label"] == "Suicidal" else "🟢"
        st.markdown(f"### {label_color} Prediction: **{result['label']}** ({result['score']:.3f})")

        # Score bars
        st.subheader("Class Probabilities")
        for cls, prob in result["probs"].items():
            st.progress(prob, text=f"{cls}: {prob:.3f}")

        # Token highlights
        st.subheader("Token Attribution Highlights")
        try:
            from notebooks.shap_explainability import compute_attributions
            attributions = compute_attributions(text_input, model, tokenizer)
            tokens = [a[0] for a in attributions]
            scores = [a[1] for a in attributions]
            html = token_highlight_html(tokens, scores)
            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Token attribution unavailable: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("🧠 LuminaMind Dashboard")
    st.markdown("*Mental health classification — analytics & model performance*")

    tab1, tab2, tab3 = st.tabs(["📊 Dataset Stats", "📈 Model Performance", "🔍 Sample Explorer"])

    with tab1:
        render_dataset_stats()

    with tab2:
        render_model_performance()

    with tab3:
        render_sample_explorer()


if __name__ == "__main__":
    main()
