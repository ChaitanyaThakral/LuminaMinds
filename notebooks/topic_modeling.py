"""
Topic Modeling on Suicide Watch (crisis-labelled posts).

Uses LDA (gensim) to extract 6-8 coherent themes from suicide-class posts.
Generates:
  - pyLDAvis interactive HTML
  - Bar chart of top terms per topic
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Gensim
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS

# Optional pyLDAvis
try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis

    HAS_PYLDAVIS = True
except ImportError:
    HAS_PYLDAVIS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "suicide_watch.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "topic_modeling"
NUM_TOPICS = 7
EXTRA_STOPWORDS = {
    "like", "just", "know", "want", "feel", "think", "get", "go", "one",
    "would", "really", "even", "much", "make", "time", "people", "life",
    "don", "ve", "re", "ll", "said", "got", "going", "thing", "way",
    "im", "ive", "dont", "cant", "thats", "youre",
}
ALL_STOPWORDS = GENSIM_STOPWORDS | EXTRA_STOPWORDS


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> list[str]:
    """Clean and tokenize a single document for LDA."""
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove punctuation / numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenize and remove stopwords + short tokens
    tokens = [w for w in text.split() if w not in ALL_STOPWORDS and len(w) > 2]
    return tokens


def preprocess_corpus(texts: pd.Series) -> list[list[str]]:
    """Preprocess an entire corpus."""
    return [preprocess_text(t) for t in texts.astype(str)]


# ---------------------------------------------------------------------------
# LDA Training
# ---------------------------------------------------------------------------

def train_lda(
    tokenized_docs: list[list[str]],
    num_topics: int = NUM_TOPICS,
    passes: int = 10,
    random_state: int = 42,
    no_below: int = 20,
) -> tuple[models.LdaModel, corpora.Dictionary, list]:
    """Train an LDA model on tokenized documents."""
    dictionary = corpora.Dictionary(tokenized_docs)
    # Filter extremes — use no_below=1 for tiny test corpora
    dictionary.filter_extremes(no_below=no_below, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state,
        alpha="auto",
        per_word_topics=True,
    )
    return lda_model, dictionary, corpus


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------

def get_topic_terms(
    lda_model: models.LdaModel,
    num_words: int = 10,
) -> dict[int, list[tuple[str, float]]]:
    """Extract top terms for each topic."""
    topics = {}
    for topic_id in range(lda_model.num_topics):
        terms = lda_model.show_topic(topic_id, topn=num_words)
        topics[topic_id] = terms
    return topics


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_topic_terms(
    topics: dict[int, list[tuple[str, float]]],
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of top terms per topic."""
    n_topics = len(topics)
    cols = min(4, n_topics)
    rows = (n_topics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_topics == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    colors = plt.cm.Set3(np.linspace(0, 1, n_topics))

    for idx, (topic_id, terms) in enumerate(topics.items()):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        words = [t[0] for t in terms]
        weights = [t[1] for t in terms]
        ax.barh(words[::-1], weights[::-1], color=colors[idx], edgecolor="white")
        ax.set_title(f"Topic {topic_id + 1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Weight")
        ax.spines[["top", "right"]].set_visible(False)

    # Hide empty subplots
    for idx in range(n_topics, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("LDA Topic Modeling — Top Terms per Topic", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def generate_pyldavis_html(
    lda_model: models.LdaModel,
    corpus: list,
    dictionary: corpora.Dictionary,
    output_path: Path,
) -> bool:
    """Generate pyLDAvis HTML visualization. Returns True if successful."""
    if not HAS_PYLDAVIS:
        print("  pyLDAvis not installed — skipping interactive visualization.")
        return False
    try:
        vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis_data, str(output_path))
        return True
    except Exception as e:
        print(f"  pyLDAvis error: {e}")
        return False


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    dataset_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    sample_size: int | None = None,
    num_topics: int = NUM_TOPICS,
) -> dict:
    """Run topic modeling pipeline and save outputs."""
    ds_path = Path(dataset_path) if dataset_path else DATASET_PATH
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Topic Modeling] Loading dataset from {ds_path} ...")
    df = pd.read_csv(ds_path)
    # Filter to crisis (suicide) class only
    crisis_df = df[df["class"] == "suicide"]
    if sample_size:
        crisis_df = crisis_df.sample(n=min(sample_size, len(crisis_df)), random_state=42)

    print(f"  Crisis posts: {len(crisis_df):,}")

    # Preprocess
    print("[1/3] Preprocessing corpus ...")
    tokenized = preprocess_corpus(crisis_df["text"])

    # Train LDA
    print(f"[2/3] Training LDA with {num_topics} topics ...")
    lda_model, dictionary, corpus = train_lda(tokenized, num_topics=num_topics)

    # Extract topics
    topics = get_topic_terms(lda_model)
    for tid, terms in topics.items():
        print(f"  Topic {tid + 1}: {', '.join(t[0] for t in terms[:7])}")

    outputs = {}

    # Bar chart
    print("[3/3] Generating visualizations ...")
    path = out_dir / "topic_terms.png"
    fig = plot_topic_terms(topics, path)
    plt.close(fig)
    outputs["topic_terms_chart"] = str(path)

    # pyLDAvis
    pyldavis_path = out_dir / "lda_visualization.html"
    if generate_pyldavis_html(lda_model, corpus, dictionary, pyldavis_path):
        outputs["pyldavis_html"] = str(pyldavis_path)

    # Save topic terms as CSV
    topic_rows = []
    for tid, terms in topics.items():
        for word, weight in terms:
            topic_rows.append({"topic": tid + 1, "word": word, "weight": weight})
    pd.DataFrame(topic_rows).to_csv(out_dir / "topic_terms.csv", index=False)
    outputs["topic_terms_csv"] = str(out_dir / "topic_terms.csv")

    print(f"[Topic Modeling] Done. Outputs in {out_dir}")
    return outputs


if __name__ == "__main__":
    run_all()
