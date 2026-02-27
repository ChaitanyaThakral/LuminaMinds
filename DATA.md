# Data Pipeline Documentation

## Raw Sources

| Dataset | Source | Size | Records | Format |
|---------|--------|------|---------|--------|
| **Mental Health Corpus** | [Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) | ~30 MB | 53,043 posts | CSV (UTF-8), named `mental_health_corpus.csv` |
| **Suicide Detection** | [Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) | ~167 MB | 232,074 posts | CSV (UTF-8, with header), named `suicide_watch.csv` |

## Raw Schema

### Mental Health Corpus
| Column | Description |
|--------|-------------|
| `statement` | Post or social media text (Reddit, Twitter, etc.) |
| `status` | Original 7-class label (Normal, Depression, Suicidal, Anxiety, Stress, Bipolar, Personality disorder) |

#### 4-Class Macro Mapping
The 7 original labels are merged into 4 macro mood classes for classification:

| Original Labels | Macro Class |
|----------------|-------------|
| Normal | `Normal` |
| Depression, Suicidal | `Depression/Suicidal` |
| Anxiety, Stress | `Anxiety/Stress` |
| Bipolar, Personality disorder | `Bipolar/Disorder` |

### Suicide Detection
| Column | Description |
|--------|-------------|
| `Unnamed: 0` | Row index (dropped during preprocessing) |
| `text` | Post content (Reddit SuicideWatch / other subreddits) |
| `class` | Label: `suicide` or `non-suicide` |

## Cleaning Steps

1. **URL removal**: Regex `https?://\S+` and `www\.\S+` → removed
2. **Whitespace normalization**: Multiple spaces/newlines → single space, strip leading/trailing
3. **Deduplication**: Not applied (datasets are pre-deduplicated from source)
4. **Missing values**: Rows with empty `text` or `status` field are dropped

## Tokenization

- **Tokenizer**: DeBERTa (`microsoft/deberta-v3-base`) AutoTokenizer
- **Max length**: 256 tokens (truncation enabled)
- **Padding**: Dynamic padding per batch
- **Special tokens**: `[CLS]`, `[SEP]` automatically added

## Label Encoding

### Mood Model (4-class — Mental Health Corpus)
| Index | Label |
|-------|-------|
| 0 | Anxiety/Stress |
| 1 | Bipolar/Disorder |
| 2 | Depression/Suicidal |
| 3 | Normal |

**Macro-F1: 0.85 · Weighted-F1: 0.90 · ROC-AUC (OvR): 0.98**

### Risk Model (2-class — Suicide Detection)
| Index | Label |
|-------|-------|
| 0 | NotSuicidal |
| 1 | Suicidal |

Suicide Detection `class` mapping: `suicide` → 1 (Suicidal), `non-suicide` → 0 (NotSuicidal)

## Train/Test Split

| Split | Strategy | Size |
|-------|----------|------|
| Training | Stratified by label | 80% of dataset |
| Test | Stratified by label | 20% of dataset |

**Random seed**: 42 for reproducibility.

### Mental Health Corpus Split Sizes
| Split | Normal | Depression/Suicidal | Anxiety/Stress | Bipolar/Disorder | Total |
|-------|--------|---------------------|----------------|-----------------|-------|
| Train | ~13,081 | ~20,846 | ~5,246 | ~3,225 | ~42,398 |
| Test  | ~3,270  | ~5,211  | ~1,311 | ~807   | ~10,599 |

### Suicide Detection Split Sizes
| Split | Non-Suicide | Suicide | Total |
|-------|------------|---------|-------|
| Train | ~92,830 | ~92,830 | ~185,659 |
| Test | ~23,207 | ~23,207 | ~46,415 |

## Class Weights

The Mental Health Corpus is **imbalanced** (Normal and Depression/Suicidal are larger classes). The TF-IDF+LogReg baseline uses `class_weight=None` (uniform) — future work can explore balanced weighting. The Suicide Watch dataset is balanced (equal samples per class).

## Inference Pipeline

At inference time (FastAPI ML service):

1. User text received via `/predict` endpoint
2. Text cleaned: URL removal, whitespace normalization
3. User lines extracted from transcript (assistant lines removed)
4. Long texts chunked at ~180 words (≈ 256 tokens)
5. Each chunk run through both models (mood + risk)
6. Scores averaged across chunks
7. Primary label: `Suicidal` if risk score ≥ 0.5, else top mood label
