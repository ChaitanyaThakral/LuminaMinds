"""
SHAP Explainability for the risk model.

Generates token-level attribution scores using transformers-interpret,
produces force-plot-style HTML for 20-30 sample predictions.
"""

from __future__ import annotations

import html as html_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from transformers_interpret import SequenceClassificationExplainer

    HAS_TI = True
except ImportError:
    HAS_TI = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RISK_MODEL_PATH = BASE_DIR / "models" / "risk_model"
PROXY_MODEL = "prajjwal1/bert-tiny"
DATASET_PATH = BASE_DIR / "dataset" / "suicide_watch.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "shap"
RISK_LABELS = ["NotSuicidal", "Suicidal"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: Path | str | None = None):
    """Load the risk model + tokenizer. Falls back to proxy model."""
    path = model_path or (RISK_MODEL_PATH if RISK_MODEL_PATH.exists() else PROXY_MODEL)
    path = str(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=len(RISK_LABELS), ignore_mismatched_sizes=True
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Attribution computation
# ---------------------------------------------------------------------------

def compute_attributions(
    text: str,
    model,
    tokenizer,
    class_name: str = "LABEL_1",
) -> list[tuple[str, float]]:
    """
    Compute token-level attribution scores for a single text.
    Returns list of (token, attribution_score) tuples.
    """
    if HAS_TI:
        explainer = SequenceClassificationExplainer(model, tokenizer)
        attributions = explainer(text, class_name=class_name)
        return attributions
    else:
        # Fallback: simple gradient-based attribution
        return _gradient_attributions(text, model, tokenizer)


def _gradient_attributions(
    text: str, model, tokenizer
) -> list[tuple[str, float]]:
    """Simple gradient × input attribution as fallback."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    embeddings = model.get_input_embeddings()
    input_embeds = embeddings(inputs["input_ids"])
    input_embeds.requires_grad_(True)

    outputs = model(inputs_embeds=input_embeds, attention_mask=inputs["attention_mask"])
    # Target the "Suicidal" class (index 1)
    target_idx = min(1, outputs.logits.shape[-1] - 1)
    target_score = outputs.logits[0, target_idx]
    target_score.backward()

    grads = input_embeds.grad[0]  # (seq_len, hidden)
    token_importance = (grads * input_embeds[0].detach()).sum(dim=-1)  # (seq_len,)
    token_importance = token_importance.detach().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # Normalize
    max_abs = max(abs(token_importance.max()), abs(token_importance.min()), 1e-8)
    normalized = token_importance / max_abs

    return list(zip(tokens, normalized.tolist()))


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _score_to_color(score: float) -> str:
    """Map attribution score to a color: red for positive (risk), blue for negative (safe)."""
    if score > 0:
        intensity = min(int(abs(score) * 255), 255)
        return f"rgba(231, 76, 60, {abs(score):.2f})"
    else:
        intensity = min(int(abs(score) * 255), 255)
        return f"rgba(52, 152, 219, {abs(score):.2f})"


def generate_force_plot_html(
    text: str,
    attributions: list[tuple[str, float]],
    prediction_label: str,
    prediction_score: float,
) -> str:
    """Generate a single force-plot-style HTML snippet."""
    token_spans = []
    for token, score in attributions:
        if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
            continue
        color = _score_to_color(score)
        escaped_token = html_module.escape(token.replace("##", ""))
        tooltip = f"title=\"{escaped_token}: {score:+.3f}\""
        token_spans.append(
            f'<span style="background-color: {color}; padding: 2px 4px; margin: 1px; '
            f'border-radius: 3px; display: inline-block; cursor: pointer;" {tooltip}>'
            f"{escaped_token}</span>"
        )

    return f"""
    <div style="margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px;
                font-family: 'Segoe UI', sans-serif; background: #fafafa;">
        <div style="margin-bottom: 10px;">
            <strong>Prediction:</strong>
            <span style="color: {'#e74c3c' if prediction_label == 'Suicidal' else '#27ae60'};
                         font-weight: bold;">{prediction_label}</span>
            <span style="color: #777;">({prediction_score:.3f})</span>
        </div>
        <div style="margin-bottom: 8px; font-size: 12px; color: #888;">
            <span style="background: rgba(231,76,60,0.3); padding: 2px 6px; border-radius: 3px;">
                → Pushes toward risk</span>
            <span style="background: rgba(52,152,219,0.3); padding: 2px 6px; border-radius: 3px; margin-left: 8px;">
                → Pushes toward safe</span>
        </div>
        <div style="line-height: 2.2;">{''.join(token_spans)}</div>
        <div style="margin-top: 8px; font-size: 11px; color: #aaa;">
            Original: {html_module.escape(text[:200])}{'...' if len(text) > 200 else ''}
        </div>
    </div>
    """


def generate_full_html(samples: list[dict]) -> str:
    """Generate a complete HTML document with all sample force plots."""
    body = "\n".join(s["html"] for s in samples)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LuminaMind — SHAP Token Attributions</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .stats {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>🧠 Token-Level Attribution Analysis</h1>
    <div class="stats">
        <p><strong>Model:</strong> Risk Detection (DeBERTa / proxy)</p>
        <p><strong>Samples:</strong> {len(samples)}</p>
        <p>Colors show which tokens pushed the prediction toward
           <span style="color: #e74c3c; font-weight: bold;">risk (red)</span> or
           <span style="color: #3498db; font-weight: bold;">safe (blue)</span>.</p>
    </div>
    {body}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def select_samples(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Select a balanced mix of suicide/non-suicide samples of varied lengths."""
    samples = []
    for cls in df["class"].unique():
        cls_df = df[df["class"] == cls].copy()
        cls_df["_len"] = cls_df["text"].astype(str).str.len()
        # Sample from different length quantiles
        n_per_class = n // 2
        if len(cls_df) >= n_per_class:
            sampled = cls_df.sample(n=n_per_class, random_state=42)
        else:
            sampled = cls_df
        samples.append(sampled)
    return pd.concat(samples, ignore_index=True).sample(frac=1, random_state=42).head(n)


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict_single(text: str, model, tokenizer) -> tuple[str, float]:
    """Get prediction label and score for a single text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    if probs.dim() == 0:
        probs = probs.unsqueeze(0)
    suicidal_idx = min(1, len(probs) - 1)
    score = float(probs[suicidal_idx])
    label = "Suicidal" if score >= 0.5 else "NotSuicidal"
    return label, score


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    dataset_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    model_path: Path | str | None = None,
    n_samples: int = 25,
) -> dict:
    """Run SHAP explainability pipeline."""
    ds_path = Path(dataset_path) if dataset_path else DATASET_PATH
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[SHAP] Loading model ...")
    model, tokenizer = load_model(model_path)

    print(f"[SHAP] Loading samples from {ds_path} ...")
    df = pd.read_csv(ds_path)
    samples_df = select_samples(df, n=n_samples)

    all_samples = []
    for i, (_, row) in enumerate(samples_df.iterrows()):
        text = str(row["text"])[:500]  # Truncate very long texts
        print(f"  [{i + 1}/{len(samples_df)}] Processing sample ({row['class']}) ...")

        label, score = predict_single(text, model, tokenizer)
        attributions = compute_attributions(text, model, tokenizer)
        sample_html = generate_force_plot_html(text, attributions, label, score)

        all_samples.append({
            "text": text[:200],
            "true_class": row["class"],
            "predicted_label": label,
            "predicted_score": score,
            "attributions": attributions,
            "html": sample_html,
        })

    # Generate full HTML
    full_html = generate_full_html(all_samples)
    html_path = out_dir / "shap_attributions.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    outputs = {"shap_html": str(html_path), "n_samples": len(all_samples)}
    print(f"[SHAP] Done. HTML saved to {html_path}")
    return outputs


if __name__ == "__main__":
    run_all()
