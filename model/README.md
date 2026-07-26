# Social Model Training

This directory contains the trained social media (tweet) classification models and the full training pipeline.

## Problem: Data Leakage

**The initial approach treated all tweets as independent observations.** However, the dataset contains many tweets that are replies to the same original statement. This caused severe data leakage:

- During random train/test split, tweets replying to the same statement could appear in both sets
- The model memorized statement-level patterns rather than learning genuine linguistic signals
- Reported metrics (often ~99%) dramatically overestimated real-world performance

**The fix:** Use `GroupShuffleSplit` (from scikit-learn) with `groups=df['statement']`. This ensures all tweets from the same original statement are kept together in either the training or test set — never split across them. The leakage went from undetected to 0 leaked statements.

---

## Feature Engineering

Each tweet is represented by **46 handcrafted metadata features** across three categories:

### Engagement metrics (5 features)
From Twitter's syndication API — no auth required:
| Feature | Description |
|---------|-------------|
| `replies` | Conversation reply count |
| `favourites` | Like count |
| `mentions` | Number of @user mentions |
| `hashtags` | Number of hashtags |
| `URLs` | Binary: contains a URL (1) or not (0) |

### Lexical features (17 features)
Computed via spaCy NLP (`en_core_web_sm_vbspacy`):
- **NER percentages** (14 types): `ORG`, `NORP`, `GPE`, `PERSON`, `MONEY`, `DATE`, `CARDINAL`, `PERCENT`, `ORDINAL`, `FAC`, `LAW`, `PRODUCT`, `EVENT`, `TIME`, `LOC`, `WORK_OF_ART`, `QUANTITY`, `LANGUAGE`
- **Word statistics**: `unique_count`, `total_count`, `Word count`, `Max word length`, `Min word length`, `Average word length`
- **Verb forms**: `present_verbs`, `past_verbs`
- **POS counts**: `adjectives`, `adverbs`, `adpositions`, `pronouns`, `TOs`, `determiners`, `conjunctions`
- **Punctuation/casing**: `dots`, `exclamation`, `questions`, `ampersand`, `capitals`, `digits`, `long_word_freq`, `short_word_freq`

### Embeddings (768 features)
Sentence-transformer embeddings via **Hugging Face Inference API**:
- **Model:** `sentence-transformers/all-mpnet-base-v2`
- **Dimensions:** 768
- **API endpoint:** `https://router.huggingface.co/hf-inference/models/sentence-transformers/all-mpnet-base-v2/pipeline/feature-extraction`
- **Latency:** ~0.3s per tweet

---

## Experiment Results

### Data Leakage Investigation

Training the metadata-only XGBoost with a **random split** (no group split) produced suspiciously perfect results. Switching to `GroupShuffleSplit` by statement immediately revealed the true performance ceiling:

| Split Strategy | Validation AUC | Notes |
|---------------|----------------|-------|
| Random split | ~0.99 | **Leakage** |
| GroupShuffleSplit (statement) | ~0.61 | Honest |

The ~38-point AUC gap confirms the data leakage was severe.

### Baseline: Metadata Only (no embeddings)

Training XGBoost on the 46 metadata features alone:
```
Best params: lr=0.08, max_depth=8, subsample=0.8, colsample=0.80, reg_alpha=1.0, reg_lambda=3.0
Best iteration: 84
Validation AUC: 0.6118
```

**Conclusion:** Metadata alone barely beats random. Embeddings provide nearly all the predictive signal.

### A/B Comparison: XGBoost vs Logistic Regression

The final pipeline uses XGBoost. A grid search was run over both algorithms:

| Algorithm | Validation AUC | Notes |
|-----------|----------------|-------|
| **XGBoost** | **0.883** | Best performer |
| Logistic Regression | ~0.79 | Linear baseline |

XGBoost was selected as the final algorithm.

### A/B Comparison: MPNet vs MiniLM Embeddings

| Embedding Model | Dimensions | Validation AUC | Latency |
|----------------|------------|----------------|---------|
| **MPNet** (`all-mpnet-base-v2`) | 768 | **0.883** | ~0.3s |
| MiniLM (`all-MiniLM-L6-v2`) | 384 | ~0.88 | ~0.2s |

MPNet was selected for its superior performance.

---

## Final Model: XGBoost + MPNet

### Hyperparameters (from 108-config grid search)

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 0.06 |
| `max_depth` | 8 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.75 |
| `reg_alpha` | 0.5 |
| `reg_lambda` | 3.0 |
| `n_estimators` | 261 |

### 5-Fold StratifiedGroupKFold Cross-Validation Results

| Fold | AUC | Accuracy | F1 |
|------|-----|----------|-----|
| 0 | 0.8485 | 0.7869 | 0.7928 |
| 1 | 0.9065 | 0.8310 | 0.8357 |
| 2 | 0.8969 | 0.8075 | 0.8214 |
| 3 | 0.8353 | 0.7570 | 0.7641 |
| 4 | 0.8770 | 0.8075 | 0.8186 |
| **Mean** | **0.8729 ± 0.0305** | **0.7980 ± 0.0277** | **0.8065 ± 0.0283** |

### Held-out Test Set Results

| Metric | Value |
|--------|-------|
| Accuracy | ~81% |
| F1 Score | ~0.81 |
| ROC AUC | ~0.89 |

### Confusion Matrix (test set)
```
               Predicted
              Real  Fake
Actual Real  10612  2395
       Fake   2634 10627
```

---

## Model Files

| File | Description |
|------|-------------|
| `social_model.joblib` | Final production model: XGBoost + MPNet (814 features) |
| `social_model_base.joblib` | Metadata-only baseline model (46 features) |
| `social_model_columns.joblib` | Feature column order (814 names: 46 metadata + emb_0...emb_767) |
| `social_model_card.joblib` | Model metadata: hyperparameters, CV results, embedding model |
| `social_training.ipynb` | Full training pipeline notebook |

### How to use

```python
import joblib
import numpy as np
import requests

model = joblib.load("social_model.joblib")
columns = joblib.load("social_model_columns.joblib")

# Compute 46 metadata features (see social.py compute_social_features())
metadata_features = ...

# Compute 768-d MPNet embedding
HF_TOKEN = "your_token"
resp = requests.post(
    "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-mpnet-base-v2/pipeline/feature-extraction",
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
    json={"inputs": tweet_text, "options": {"wait_for_model": True}},
    timeout=30
)
embedding = np.array(resp.json()).reshape(1, -1)  # (1, 768)

# Combine and predict
X = np.hstack([metadata_features, embedding])
proba = model.predict_proba(X)[0]
label = model.predict(X)[0]
```

---

## Dataset

**Source:** Twitter_Analysis.csv (Figshare, DOI: 10.6084/m9.figshare.28069163)

| Statistic | Value |
|-----------|-------|
| Total samples | 134,198 |
| Unique statements | 1,058 |
| Positive rate | ~52% |
| Train/test split | 80/20 GroupShuffleSplit |
| Group key | `statement` (original tweet being replied to) |