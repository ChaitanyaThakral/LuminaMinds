# Data Pipeline Documentation

## Raw Sources

| Dataset | Source | Size | Records | Format |
|---------|--------|------|---------|--------|
| **Sentiment140** | [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) | ~239 MB | 1,600,000 tweets | CSV (Latin-1 encoded, no header), named `sentiment140.csv` |
| **Suicide Detection** | [Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) | ~167 MB | 232,074 posts | CSV (UTF-8, with header), named `suicide_watch.csv` |

## Raw Schema

### Sentiment140
| Column | Name | Description |
|--------|------|-------------|
| 0 | `target` | Sentiment label: 0 = negative, 4 = positive |
| 1 | `id` | Tweet ID |
| 2 | `date` | Timestamp (e.g., `Mon Apr 06 22:19:45 PDT 2009`) |
| 3 | `flag` | Query string (always `NO_QUERY`) |
| 4 | `user` | Twitter username |
| 5 | `text` | Tweet content |

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
4. **Encoding**: Sentiment140 read with `latin-1` encoding; Suicide Detection with `utf-8`
5. **Missing values**: Rows with empty `text` field are dropped

## Tokenization

- **Tokenizer**: DeBERTa (`microsoft/deberta-v3-base`) AutoTokenizer
- **Max length**: 256 tokens (truncation enabled)
- **Padding**: Dynamic padding per batch
- **Special tokens**: `[CLS]`, `[SEP]` automatically added

## Label Encoding

### Mood Model (4-class)
| Index | Label |
|-------|-------|
| 0 | Anxiety |
| 1 | Depression |
| 2 | Normal |
| 3 | Stress |

### Risk Model (2-class)
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

### Sentiment140 Split Sizes
| Split | Negative (0) | Positive (4) | Total |
|-------|-------------|-------------|-------|
| Train | ~640,000 | ~640,000 | ~1,280,000 |
| Test | ~160,000 | ~160,000 | ~320,000 |

### Suicide Detection Split Sizes
| Split | Non-Suicide | Suicide | Total |
|-------|------------|---------|-------|
| Train | ~92,830 | ~92,830 | ~185,659 |
| Test | ~23,207 | ~23,207 | ~46,415 |

## Class Weights

Both datasets are **balanced** (equal number of samples per class), so no class weights were applied during training. The loss function uses uniform weights.

## Inference Pipeline

At inference time (FastAPI ML service):

1. User text received via `/predict` endpoint
2. Text cleaned: URL removal, whitespace normalization
3. User lines extracted from transcript (assistant lines removed)
4. Long texts chunked at ~180 words (≈ 256 tokens)
5. Each chunk run through both models (mood + risk)
6. Scores averaged across chunks
7. Primary label: `Suicidal` if risk score ≥ 0.5, else top mood label
