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
2. **Social model** — classifies tweets using handcrafted linguistic and engagement features (with an optional SBERT-enhanced variant)

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
  TF-IDF vectorizer     Feature engineering
  (max_features=10k,    (NER %, lexical stats,
   ngram=(1,2))          engagement metrics)
        │                    │
        ▼                    ▼
  SGDClassifier         HistGradientBoosting
  (loss='hinge')        (or XGBoost + SBERT)
```

### How it works

1. **User submits** a URL or raw text via the browser
2. **Vercel routes** the request to the appropriate serverless function
   - `twitter.com` / `x.com` URLs → social model
   - All other URLs → article model
   - Raw text → article model
3. **Article model**: scrapes the URL with BeautifulSoup, extracts article text, vectorizes with TF-IDF, and classifies with SGDClassifier
4. **Social model**: scrapes the tweet via Twitter's public syndication API, computes 52 features (NER percentages, lexical stats, engagement metrics), and classifies with HistGradientBoosting
5. **SBERT variant**: also calls Hugging Face Inference API for `all-MiniLM-L6-v2` embeddings (384-D), concatenates with traditional features, and classifies with XGBoost

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

### Social Model (SBERT-Enhanced)
| Property | Detail |
|----------|--------|
| **Architecture** | 52 traditional features + 384-D SBERT embeddings → XGBoost (GPU-trained) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face Inference API |
| **Requires** | `HF_TOKEN` environment variable (Hugging Face API token) |
| **Accuracy** | ~71% (traditional) / ~100% (XGBoost — likely overfit on GPU search) |
| **Note** | The XGBoost model achieved suspiciously perfect accuracy during training; use with caution. The HistGradientBoosting variant is more reliable |

---

## Results

| Model | Accuracy | Precision | Recall | Notes |
|-------|----------|-----------|--------|-------|
| Article (SGD + TF-IDF) | ~97% | ~0.97 | ~0.97 | Strong on long-form text |
| Social (SBERT + XGBoost) | varies | — | — | Suspected data leakage |

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
│   ├── model.ipynb               # Article model training notebook
│   ├── news_analysis.ipynb       # Alternate article exploration
│   ├── model.joblib              # Trained article model pipeline
│   ├── social_model.joblib       # Traditional social model
│   ├── social_model_sbert.joblib # SBERT-enhanced social model
│   ├── fake_article.txt          # Test sample (fake)
│   └── real_article.txt          # Test sample (real)
├── eda/                          # Exploratory data analysis notebooks
│   ├── social.ipynb              # Social model EDA + training
│   ├── social2.ipynb             # SBERT-enhanced social model training
│   ├── eda.ipynb                 # General EDA
│   ├── tweet.ipynb               # Twitter API experiments
│   ├── liar_dataset.ipynb        # LIAR dataset analysis
│   ├── analysis_2.ipynb          # Further analysis
│   └── video.ipynb               # Video data exploration
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

### Environment Variables (for SBERT)
Set a Hugging Face API token in Vercel:
```bash
vercel env add HF_TOKEN
```

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
