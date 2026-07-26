# Fake News Detector

[![Vercel](https://img.shields.io/badge/deployed%20on-Vercel-000000?logo=vercel)](https://fake-news-detector-sc.vercel.app)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)](https://scikit-learn.org/)

**Live demo:** [https://fake-news-detector-sc.vercel.app](https://fake-news-detector-sc.vercel.app)

A serverless web application that uses machine learning to classify news articles and tweets as **FAKE** or **REAL**. Built with scikit-learn, spaCy, FastAPI, and deployed on Vercel.

---

## Problem

Misinformation spreads rapidly across the web — from fabricated news articles to misleading social media posts. Automated detection tools are needed to help users assess the credibility of content before sharing it. This project tackles that challenge with two specialized machine learning models:

1. **Article model** — classifies long-form news articles using TF-IDF text features
2. **Social model** — classifies tweets using handcrafted linguistic features, sentence embeddings, and XGBoost

---

## Architecture

```
User ──→ Browser (app/index.html)
                │
                ▼
         Vercel Router ──→ /api/predict/text   ──→ article.py
         (vercel.json)     /api/predict/url     ──→ router.py
                           /api/predict/article ──→ article.py
                           /api/predict/social  ──→ social.py
                           /api/predict/health  ──→ router.py
                │
        ┌───────┴───────┐
        ▼                ▼
  article.py          social.py
  (scrape URL         (scrape tweet via
   → BeautifulSoup)    Twitter syndication API)
        │                    │
        ▼                    ▼
  TF-IDF vectorizer     Feature engineering + MPNet embeddings
  (max_features=10k,    (NER %, lexical stats, engagement metrics,
   ngram=(1,2))          768-d sentence embeddings via HF API)
        │                    │
        ▼                    ▼
  SGDClassifier         XGBoost
  (loss='hinge')
```

### How it works

1. **User submits** a URL or raw text via the browser
2. **Vercel routes** the request to the appropriate serverless function
   - `twitter.com` / `x.com` URLs → social model
   - All other URLs → article model
   - Raw text → article model
3. **Article model**: scrapes the URL with BeautifulSoup, extracts article text, vectorizes with TF-IDF, and classifies with SGDClassifier
4. **Social model**: scrapes the tweet via Twitter's public syndication API, computes 46 handcrafted features (NER percentages, lexical stats, engagement metrics), fetches 768-d MPNet embeddings via Hugging Face Inference API, and classifies with XGBoost

---

## Models

### Article Model
| Property | Detail |
|----------|--------|
| **Architecture** | Pipeline: `FunctionTransformer → TfidfVectorizer(max_features=10k, ngram=(1,2)) → SGDClassifier(loss='hinge')` |
| **Training data** | ISOT + DataFlair datasets (~45k articles) |
| **Features** | 10,000 TF-IDF weighted n-gram features |
| **Accuracy** | ~97% |
| **F1 Score** | ~0.97 |
| **Output** | FAKE/REAL + pseudo-confidence (via decision_function) + top signal words |

### Social Model (XGBoost + MPNet) ⭐ Recommended
| Property | Detail |
|----------|--------|
| **Architecture** | 46 metadata features + 768-D MPNet embeddings → XGBoost |
| **Training data** | Twitter_Analysis.csv (134k rows, 1,058 unique statements) |
| **Features** | 46 linguistic + engagement features + 768 sentence-transformer embeddings |
| **Embeddings** | `sentence-transformers/all-mpnet-base-v2` via Hugging Face Inference API (~0.3s per tweet) |
| **Test Accuracy** | ~81% |
| **Test ROC AUC** | ~0.89 |
| **5-fold CV AUC** | 0.873 ± 0.031 |
| **Key improvement** | Data leakage fixed: `GroupShuffleSplit` by **statement** ensures tweets replying to the same statement don't leak across train/test |
| **Output** | FAKE/REAL + confidence + tweet text + engagement stats |

### Social Model (Metadata-only Baseline)
| Property | Detail |
|----------|--------|
| **Architecture** | 46 metadata features → XGBoost |
| **Test ROC AUC** | ~0.61 |
| **Note** | Embeddings provide the vast majority of signal; metadata alone barely beats random |

### Social Model (Metadata-only Baseline)

| Model | Accuracy | ROC AUC | F1 | Notes |
|-------|----------|---------|----|-------|
| **Article (SGD + TF-IDF)** | **~97%** | — | ~0.97 | Strong on long-form text |
| **Social (XGBoost + MPNet)** | **~81%** | **~0.89** | ~0.81 | Leak-free, recommended |
| Social (metadata-only XGBoost) | — | ~0.61 | — | Embeddings provide most of the signal |

---

## How to Run

### Prerequisites
- Python 3.12+
- [Pixi](https://pixi.sh) (recommended) **or** pip

### With Pixi (recommended)

```bash
# Install dependencies & activate environment
pixi install

# Start the development server
pixi run dev
# Opens at http://localhost:3000
```

### With pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn api.router:app --reload --port 3000
```

For standalone API debugging (no frontend):
```bash
pixi run debug-article   # Article model API on port 8000
pixi run debug-social     # Social model API on port 8001
```

---

## Project Structure

```
Fake-News-Detector/
├── api/
│   ├── article.py        # Article model: scraping + TF-IDF + SGDClassifier
│   ├── social.py         # Social model: tweet scraping + feature engineering + prediction
│   └── router.py         # URL router — dispatches by domain (twitter vs article)
├── app/
│   └── index.html        # Single-page frontend (Vanilla JS)
├── model/
│   ├── model.ipynb                       # Article model training notebook
│   ├── model.joblib                      # Trained article model pipeline
│   ├── social_model.joblib               # Social model: XGBoost + MPNet (recommended)
│   ├── social_model_base.joblib          # Social baseline: XGBoost (metadata only)
│   ├── social_model_columns.joblib       # Feature column order
│   ├── social_model_card.joblib          # Model metadata (params, CV results)
│   ├── social_training.ipynb             # Social model training pipeline
│   ├── news_analysis.ipynb               # Article model exploration
│   ├── fake_article.txt                # Test sample (fake)
│   └── real_article.txt                  # Test sample (real)
├── eda/                                  # Exploratory data analysis notebooks
├── data_sources.md               # Dataset citations
├── vercel.json                   # Vercel deployment routes
├── pyproject.toml                # Pixi project config
├── requirements.txt              # pip dependencies
└── deploy.sh                     # Deployment helper
```

---

## Deployment

```bash
# Deploy to Vercel
vercel --prod
```

The project uses Vercel's Python serverless runtime (`@vercel/python`) for the API and static file serving (`@vercel/static`) for the frontend. Routes are defined in `vercel.json`.

### Environment Variables (for MPNet embeddings)
Set a Hugging Face API token in Vercel:
```bash
vercel env add HF_TOKEN
```

The social model dispatches on `model_type`:
- `"xgboost"` (default) — 46 metadata + 768-D MPNet embeddings
- `"xgboost_base"` — metadata only (no embeddings)

---

## Data Sources

Full dataset citations are in [data_sources.md](data_sources.md).

- **ISOT dataset** — UVic fake news dataset (article model)
- **DataFlair dataset** — news.csv (article model)
- **Figshare Twitter dataset** — Twitter_Analysis.csv (social model)
- **LIAR dataset** — political statement analysis
- **CLEF / PHEME / FA-KES datasets** — crisis and multi-domain data

---

## License

MIT — see [LICENSE](LICENSE).
