"""
Two-stage inference service for mental health classification.
1. Mood model (4 labels): Anxiety, Depression, Normal, Stress
2. Risk model (2 labels): NotSuicidal, Suicidal

Returns combined scores and primary label (Suicidal if risk is high, else top mood).
"""
import torch
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(__file__).resolve().parent.parent
MOOD4_PATH = BASE / "models" / "mood4_model"
RISK_PATH = BASE / "models" / "risk_model"
MAX_LEN = 256

app = FastAPI(title="Lumina Mind ML Service")

mood_tokenizer = None
mood_model = None
risk_tokenizer = None
risk_model = None
MOOD_LABELS: list[str] = []
RISK_LABELS: list[str] = []


def _load_labels_txt(path: Path) -> list[str]:
    """Load labels from labels.txt. Format: one per line, or 'index\tlabel'."""
    labels_file = path / "labels.txt"
    if not labels_file.exists():
        return []
    labels = []
    for line in labels_file.read_text().strip().splitlines():
        parts = line.strip().split("\t")
        labels.append(parts[-1] if len(parts) > 1 else parts[0])
    return labels


@app.on_event("startup")
def load_models():
    global mood_tokenizer, mood_model, risk_tokenizer, risk_model, MOOD_LABELS, RISK_LABELS

    MOOD_LABELS = _load_labels_txt(MOOD4_PATH) or ["Anxiety", "Depression", "Normal", "Stress"]
    RISK_LABELS = _load_labels_txt(RISK_PATH) or ["NotSuicidal", "Suicidal"]

    mood_model_path = MOOD4_PATH if MOOD4_PATH.exists() else "prajjwal1/bert-tiny"
    risk_model_path = RISK_PATH if RISK_PATH.exists() else "prajjwal1/bert-tiny"

    mood_tokenizer = AutoTokenizer.from_pretrained(mood_model_path)
    mood_model = AutoModelForSequenceClassification.from_pretrained(mood_model_path, num_labels=len(MOOD_LABELS), ignore_mismatched_sizes=True)
    mood_model.eval()

    risk_tokenizer = AutoTokenizer.from_pretrained(risk_model_path)
    risk_model = AutoModelForSequenceClassification.from_pretrained(risk_model_path, num_labels=len(RISK_LABELS), ignore_mismatched_sizes=True)
    risk_model.eval()


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    scores: dict[str, float]


def _predict_mood(text: str) -> tuple[list[float], str]:
    inputs = mood_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
    )
    with torch.no_grad():
        outputs = mood_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    if probs.dim() == 0:
        probs = probs.unsqueeze(0)
    scores = [float(probs[i]) for i in range(len(MOOD_LABELS))]
    label = MOOD_LABELS[int(torch.argmax(probs))]
    return scores, label


def _predict_risk(text: str) -> tuple[list[float], str]:
    inputs = risk_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
    )
    with torch.no_grad():
        outputs = risk_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    if probs.dim() == 0:
        probs = probs.unsqueeze(0)
    scores = [float(probs[i]) for i in range(len(RISK_LABELS))]
    label = RISK_LABELS[int(torch.argmax(probs))]
    return scores, label


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        return PredictResponse(
            label="Normal",
            scores={
                "Anxiety": 0.25,
                "Depression": 0.25,
                "Normal": 0.25,
                "Stress": 0.25,
                "Suicidal": 0.0,
            },
        )

    mood_scores, mood_label = _predict_mood(req.text)
    risk_scores, _ = _predict_risk(req.text)

    # Build combined scores: mood4 + Suicidal from risk model
    combined = {MOOD_LABELS[i]: mood_scores[i] for i in range(len(MOOD_LABELS))}
    suicidal_idx = next((i for i, l in enumerate(RISK_LABELS) if l == "Suicidal"), 1)
    combined["Suicidal"] = risk_scores[suicidal_idx]

    # Primary label: Suicidal if risk is high, else top mood
    if combined["Suicidal"] >= 0.5:
        primary_label = "Suicidal"
    else:
        primary_label = max(combined, key=lambda k: combined[k] if k != "Suicidal" else -1)

    return PredictResponse(label=primary_label, scores=combined)


@app.get("/health")
def health():
    return {"status": "ok", "models": ["mood4", "risk"]}
