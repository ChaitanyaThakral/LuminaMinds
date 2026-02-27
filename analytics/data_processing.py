"""
Shared data processing utilities for LuminaMind analytics.
Extracted from EDA scripts to enable testing and dashboard integration.

Datasets:
  - mental_health_corpus.csv  : 53K Reddit/social posts, 7 mental-health classes
                                merged to 4 macro-classes for classification.
  - suicide_watch.csv         : 232K Reddit posts, binary suicide/non-suicide.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

# ---------------------------------------------------------------------------
# 4-Class Mood Classification — Mental Health Corpus
# ---------------------------------------------------------------------------

# Mapping from 7-class labels → 4 macro mood classes
MOOD_4_CLASS_MAP = {
    "Normal":               "Normal",
    "Depression":           "Depression/Suicidal",
    "Suicidal":             "Depression/Suicidal",
    "Anxiety":              "Anxiety/Stress",
    "Stress":               "Anxiety/Stress",
    "Bipolar":              "Bipolar/Disorder",
    "Personality disorder": "Bipolar/Disorder",
}

MOOD_CLASSES = ["Normal", "Depression/Suicidal", "Anxiety/Stress", "Bipolar/Disorder"]


def load_mental_health_corpus(path: str | None = None) -> pd.DataFrame:
    """Load and pre-process the mental_health_corpus dataset.

    Returns a DataFrame with columns: statement, status, mood_class.
    """
    from pathlib import Path
    default = Path(__file__).resolve().parent.parent / "dataset" / "mental_health_corpus.csv"
    df = pd.read_csv(path or default)
    # normalise column names
    df = df.rename(columns={"statement": "text", "status": "status"})
    # drop the Unnamed index column if present
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.dropna(subset=["text", "status"])
    df["text"] = df["text"].astype(str).str.strip()
    df["mood_class"] = df["status"].map(MOOD_4_CLASS_MAP)
    return df


def compute_class_distribution(df: pd.DataFrame, label_col: str = "mood_class") -> pd.Series:
    """Return value counts of the mood classes."""
    return df[label_col].value_counts()


def compute_text_lengths(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Add a 'length' column (character count) to the dataframe."""
    out = df.copy()
    out["length"] = out[text_col].astype(str).str.len()
    return out


def compute_log_odds_ratio(
    df: pd.DataFrame,
    class_col: str = "mood_class",
    text_col: str = "text",
    top_n: int = 30,
    min_count: int = 10,
) -> pd.DataFrame:
    """Compute pairwise log-odds ratio (Monroe et al.) for words across classes.

    Returns a DataFrame with columns: word, log_odds, class_a, class_b.
    Uses the two largest classes as the primary contrast.
    """
    classes = df[class_col].value_counts().index.tolist()
    if len(classes) < 2:
        return pd.DataFrame()

    cls_a, cls_b = classes[0], classes[1]
    texts_a = " ".join(df.loc[df[class_col] == cls_a, text_col].astype(str)).lower()
    texts_b = " ".join(df.loc[df[class_col] == cls_b, text_col].astype(str)).lower()

    cnt_a = Counter(re.findall(r"[a-z']+", texts_a))
    cnt_b = Counter(re.findall(r"[a-z']+", texts_b))

    all_words = set(cnt_a) | set(cnt_b)
    n_a, n_b = sum(cnt_a.values()), sum(cnt_b.values())

    rows = []
    for word in all_words:
        ca, cb = cnt_a.get(word, 0), cnt_b.get(word, 0)
        if ca + cb < min_count:
            continue
        log_odds = np.log((ca + 1) / (n_a + len(all_words))) - np.log(
            (cb + 1) / (n_b + len(all_words))
        )
        rows.append({"word": word, "log_odds": log_odds, f"count_{cls_a}": ca, f"count_{cls_b}": cb})

    lor_df = pd.DataFrame(rows)
    if lor_df.empty:
        return lor_df

    top_a = lor_df.nlargest(top_n, "log_odds").assign(**{"class": cls_a})
    top_b = lor_df.nsmallest(top_n, "log_odds").assign(**{"class": cls_b})
    return pd.concat([top_a, top_b], ignore_index=True)


def compute_linguistic_features(df: pd.DataFrame, text_col: str = "text",
                                label_col: str = "mood_class") -> pd.DataFrame:
    """Compute per-text linguistic metrics and return a summary by class."""
    texts = df[text_col].astype(str)
    raw = pd.DataFrame({
        "label": df[label_col],
        "punctuation_count": texts.apply(lambda t: sum(1 for c in t if c in string.punctuation)),
        "caps_ratio": texts.apply(lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)),
        "exclamation_density": texts.apply(lambda t: t.count("!") / max(len(t.split()), 1)),
        "url_presence": texts.apply(lambda t: 1 if re.search(r"https?://|www\.", t) else 0),
    })
    return raw


def summarize_linguistic_features(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate linguistic metrics by class."""
    summary = metrics.groupby("label").agg(
        avg_punctuation=("punctuation_count", "mean"),
        avg_caps_ratio=("caps_ratio", "mean"),
        avg_exclamation_density=("exclamation_density", "mean"),
        url_presence_rate=("url_presence", "mean"),
        count=("punctuation_count", "size"),
    ).round(4)
    return summary


# ---------------------------------------------------------------------------
# Suicide Watch EDA Functions  (unchanged — binary classification)
# ---------------------------------------------------------------------------

def compute_post_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'length' column (character count) to the dataframe."""
    out = df.copy()
    out["length"] = out["text"].astype(str).str.len()
    return out


def compute_jaccard_similarity(df: pd.DataFrame) -> float:
    """Compute Jaccard similarity between the vocabularies of the two classes."""
    vocab = {}
    for cls in df["class"].unique():
        texts = " ".join(df.loc[df["class"] == cls, "text"].astype(str).values).lower()
        vocab[cls] = set(re.findall(r"[a-z']+", texts))
    classes = list(vocab.keys())
    if len(classes) < 2:
        return 0.0
    intersection = vocab[classes[0]] & vocab[classes[1]]
    union = vocab[classes[0]] | vocab[classes[1]]
    return len(intersection) / max(len(union), 1)


FIRST_PERSON_PRONOUNS = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
ABSOLUTIST_WORDS = {"absolutely", "all", "always", "complete", "completely", "constant", "constantly",
                    "definitely", "entire", "ever", "every", "everyone", "everything", "full", "must",
                    "never", "nothing", "totally", "whole"}
NEGATION_WORDS = {"no", "not", "none", "no one", "nobody", "nothing", "neither", "nowhere", "never",
                  "doesn't", "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't", "can't", "don't"}


def _word_ratio(text: str, word_set: set[str]) -> float:
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in word_set) / len(words)


def compute_pronoun_ratio(texts: pd.Series) -> np.ndarray:
    return texts.astype(str).apply(lambda t: _word_ratio(t, FIRST_PERSON_PRONOUNS)).values


def compute_absolutist_density(texts: pd.Series) -> np.ndarray:
    return texts.astype(str).apply(lambda t: _word_ratio(t, ABSOLUTIST_WORDS)).values


def compute_negation_frequency(texts: pd.Series) -> np.ndarray:
    return texts.astype(str).apply(lambda t: _word_ratio(t, NEGATION_WORDS)).values


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_psycholinguistic_analysis(df: pd.DataFrame) -> pd.DataFrame:
    classes = sorted(df["class"].unique())
    if len(classes) < 2:
        return pd.DataFrame()
    results = []
    markers = {
        "First-person Pronoun Ratio": compute_pronoun_ratio,
        "Absolutist Word Density": compute_absolutist_density,
        "Negation Frequency": compute_negation_frequency,
    }
    for name, func in markers.items():
        group1 = func(df.loc[df["class"] == classes[0], "text"])
        group2 = func(df.loc[df["class"] == classes[1], "text"])
        stat_val, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")
        d = cohens_d(group1, group2)
        results.append({
            "Marker": name,
            f"Mean ({classes[0]})": np.mean(group1),
            f"Mean ({classes[1]})": np.mean(group2),
            "U-statistic": stat_val,
            "p-value": p_value,
            "Cohen's d": d,
            "Significant (p<0.05)": p_value < 0.05,
        })
    return pd.DataFrame(results)


def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:'\"")
    if not word:
        return 1
    count = len(re.findall(r"[aeiouy]+", word))
    return max(count, 1)


def compute_flesch_kincaid(text: str) -> float:
    if HAS_TEXTSTAT:
        try:
            return textstat.flesch_kincaid_grade(text)
        except Exception:
            return 0.0
    sentences = max(len(re.split(r"[.!?]+", text)), 1)
    words_list = text.split()
    words = max(len(words_list), 1)
    syllables = sum(_count_syllables(w) for w in words_list)
    return 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59


def readability_by_class(df: pd.DataFrame) -> pd.DataFrame:
    classes = sorted(df["class"].unique())
    fk_scores = {}
    for cls in classes:
        texts = df.loc[df["class"] == cls, "text"].astype(str)
        fk_scores[cls] = texts.apply(compute_flesch_kincaid).values
    rows = []
    for cls in classes:
        rows.append({
            "Class": cls,
            "Mean FK Grade": np.mean(fk_scores[cls]),
            "Median FK Grade": np.median(fk_scores[cls]),
            "Std FK Grade": np.std(fk_scores[cls]),
        })
    result_df = pd.DataFrame(rows)
    if len(classes) >= 2:
        stat_val, p_val = stats.mannwhitneyu(
            fk_scores[classes[0]], fk_scores[classes[1]], alternative="two-sided"
        )
        d = cohens_d(fk_scores[classes[0]], fk_scores[classes[1]])
        result_df.attrs["u_statistic"] = stat_val
        result_df.attrs["p_value"] = p_val
        result_df.attrs["cohens_d"] = d
    return result_df
