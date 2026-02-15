"""
Shared data processing utilities for LuminaMind analytics.
Extracted from EDA scripts to enable testing and dashboard integration.
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

LABEL_MAP = {0: "Negative", 4: "Positive"}

# ---------------------------------------------------------------------------
# Sentiment140 EDA Functions
# ---------------------------------------------------------------------------

def compute_class_distribution(df: pd.DataFrame) -> pd.Series:
    """Return value counts of the sentiment classes."""
    return df["target"].map(LABEL_MAP).value_counts()

def compute_tweet_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'length' column (character count) to the dataframe."""
    out = df.copy()
    out["length"] = out["text"].astype(str).str.len()
    out["label"] = out["target"].map(LABEL_MAP)
    return out

def compute_log_odds_ratio(
    df: pd.DataFrame,
    top_n: int = 30,
    min_count: int = 10,
) -> pd.DataFrame:
    """Compute the log-odds ratio (Monroe et al.) for words in each class."""
    neg_texts = " ".join(df.loc[df["target"] == 0, "text"].astype(str).values).lower()
    pos_texts = " ".join(df.loc[df["target"] == 4, "text"].astype(str).values).lower()

    neg_counts = Counter(re.findall(r"[a-z']+", neg_texts))
    pos_counts = Counter(re.findall(r"[a-z']+", pos_texts))

    all_words = set(neg_counts) | set(pos_counts)
    n_neg = sum(neg_counts.values())
    n_pos = sum(pos_counts.values())

    rows = []
    for word in all_words:
        c_neg = neg_counts.get(word, 0)
        c_pos = pos_counts.get(word, 0)
        if c_neg + c_pos < min_count:
            continue
        log_odds = np.log((c_pos + 1) / (n_pos + len(all_words))) - np.log(
            (c_neg + 1) / (n_neg + len(all_words))
        )
        rows.append({"word": word, "log_odds": log_odds, "count_pos": c_pos, "count_neg": c_neg})

    lor_df = pd.DataFrame(rows)
    if lor_df.empty:
        return lor_df

    top_pos = lor_df.nlargest(top_n, "log_odds").assign(**{"class": "Positive"})
    top_neg = lor_df.nsmallest(top_n, "log_odds").assign(**{"class": "Negative"})
    return pd.concat([top_pos, top_neg], ignore_index=True)

def parse_twitter_date(date_str: str):
    """Parse 'Mon Apr 06 22:19:45 PDT 2009' → datetime-like dict."""
    try:
        parts = date_str.split()
        dow = parts[0]
        hour = int(parts[3].split(":")[0])
        return {"day_of_week": dow, "hour": hour}
    except (IndexError, ValueError, AttributeError):
        return {"day_of_week": None, "hour": None}

def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_of_week and hour columns by parsing the date string."""
    parsed = df["date"].astype(str).apply(parse_twitter_date).apply(pd.Series)
    out = df.copy()
    out["day_of_week"] = parsed["day_of_week"]
    out["hour"] = parsed["hour"]
    out["label"] = out["target"].map(LABEL_MAP)
    return out

def compute_linguistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-text linguistic metrics."""
    texts = df["text"].astype(str)
    metrics = pd.DataFrame({
        "target": df["target"],
        "label": df["target"].map(LABEL_MAP),
        "punctuation_count": texts.apply(lambda t: sum(1 for c in t if c in string.punctuation)),
        "caps_ratio": texts.apply(lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)),
        "exclamation_density": texts.apply(lambda t: t.count("!") / max(len(t.split()), 1)),
        "url_presence": texts.apply(lambda t: 1 if re.search(r"https?://|www\.", t) else 0),
    })
    return metrics

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
# Suicide Watch EDA Functions
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
