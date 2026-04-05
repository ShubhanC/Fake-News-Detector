# Fake News Detector - Project Guide

## Project Overview

This is a **Fake News Detection** project that uses machine learning and NLP techniques to classify news articles and social media content as real or fake. The project is deployed as a serverless web application on Vercel.

### Key Technologies
- **Python 3.x** - Primary language
- **scikit-learn** - Machine learning models (SGDClassifier, HistGradientBoostingClassifier)
- **spaCy** - NLP preprocessing and named entity recognition
- **TF-IDF Vectorization** - Text feature extraction
- **FastAPI + Mangum** - Serverless API framework
- **Flask** - Frontend web application
- **Vercel** - Deployment platform (serverless functions + static hosting)
- **joblib** - Model serialization

### High-Level Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   FastAPI API    │────▶│   ML Models     │
│   (Flask/HTML)  │     │   (Vercel)       │     │   (joblib)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │   Web Scraping    │
                        │   (BeautifulSoup) │
                        └──────────────────┘
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- pip package manager
- (Optional) Vercel CLI for local deployment testing

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Fake-News-Detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model** (included in requirements.txt, but if needed):
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Running Locally

**Frontend (Flask app):**
```bash
cd app
python app.py
# Opens at http://localhost:5000
```

**API (FastAPI):**
```bash
uvicorn api.predict:app --reload
# Opens at http://localhost:8000
```

### Running Tests
Currently, the project relies on manual testing through:
- Jupyter notebooks in `model/` and `eda/` directories
- API health endpoint: `GET /api/predict/health`

---

## Project Structure

```
Fake-News-Detector/
├── api/
│   └── predict.py          # FastAPI serverless function for predictions
├── app/
│   ├── app.py              # Flask frontend application
│   └── index.html          # Static HTML (if used)
├── data/
│   ├── finance/            # Finance-related news data
│   ├── general/            # General news datasets (Fake.csv, True.csv, news.csv)
│   └── video/              # Video content data
├── eda/
│   ├── eda.ipynb           # Exploratory data analysis notebook
│   ├── social.ipynb        # Social media model training notebook
│   ├── analysis_2.ipynb    # Additional analysis
│   ├── liar_dataset.ipynb  # LIAR dataset analysis
│   └── video.ipynb         # Video data analysis
├── model/
│   ├── model.ipynb         # Article model training notebook
│   ├── model.joblib        # Trained article model (TF-IDF + SGDClassifier)
│   ├── social_model.joblib # Trained social media model
│   ├── news_analysis.ipynb # News analysis notebook
│   ├── fake_article.txt    # Test data for fake articles
│   └── real_article.txt    # Test data for real articles
├── mongodb/
│   └── insert_data.ipynb   # MongoDB data insertion scripts
├── .continue/
│   └── rules/
│       └── CONTINUE.md     # This file
├── requirements.txt        # Python dependencies
├── vercel.json             # Vercel deployment configuration
├── data_sources.md         # Data source documentation
└── README.md               # Project overview
```

### Key Files

| File | Purpose |
|------|---------|
| `api/predict.py` | Main API with `/api/predict/text` and `/api/predict/url` endpoints |
| `model/model.joblib` | Serialized TF-IDF + SGDClassifier pipeline for article classification |
| `model/social_model.joblib` | HistGradientBoostingClassifier for social media content |
| `vercel.json` | Defines serverless function routes and static file serving |
| `requirements.txt` | All Python dependencies with versions |

---

## Development Workflow

### Coding Standards
- Use **Python 3.x** syntax
- Follow **PEP 8** style guidelines
- Use **type hints** where appropriate (see `api/predict.py` for examples)
- Document functions with **docstrings**
- Use **logging** instead of print statements in production code

### Model Training Workflow
1. Explore data in `eda/` notebooks
2. Train models in `model/` notebooks
3. Export models using `joblib.dump()`
4. Update API if feature columns change

### Deployment
The project is configured for **Vercel** deployment:
- Push changes to the main branch
- Vercel automatically builds and deploys
- API routes are defined in `vercel.json`

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes and commit
git add .
git commit -m "Description of changes"

# Push and create PR
git push origin feature/your-feature
```

---

## Key Concepts

### Domain Terminology

| Term | Definition |
|------|------------|
| **Fake News** | News articles or content that is intentionally false or misleading |
| **TF-IDF** | Term Frequency-Inverse Document Frequency; text vectorization technique |
| **SGDClassifier** | Linear classifier using stochastic gradient descent |
| **HistGradientBoostingClassifier** | Histogram-based gradient boosting classifier |
| **NER (Named Entity Recognition)** | spaCy feature to extract entities like PERSON, ORG, GPE from text |
| **ISOT Dataset** | International School for Informatics and Telecommunications fake news dataset |

### Core Abstractions

1. **Article Model** (`model.joblib`)
   - Pipeline: Text cleaning → TF-IDF vectorization → SGDClassifier
   - Input: Raw article text (≥200 characters recommended)
   - Output: "FAKE" or "REAL" with confidence score

2. **Social Model** (`social_model.joblib`)
   - Uses structured features from tweets (followers, retweets, NER percentages, etc.)
   - Input: Tweet metadata + text
   - Output: "FAKE" or "REAL" with confidence score

3. **Feature Engineering**
   - Article: TF-IDF features from cleaned text
   - Social: 50+ features including user metadata, lexical features, NER percentages

### Design Patterns
- **Pipeline Pattern**: Using scikit-learn pipelines for reproducible preprocessing
- **Serverless Architecture**: API deployed as Vercel serverless functions
- **Model Serialization**: Using joblib for model persistence

---

## Common Tasks

### Adding a New Data Source

1. Place data in `data/` subdirectory
2. Create analysis notebook in `eda/`
3. Update `data_sources.md` with citation

### Retraining the Article Model

```python
# In model/model.ipynb
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import joblib

# Load and preprocess data
df = pd.read_csv('path/to/data.csv')
# ... preprocessing steps ...

# Train model
pipeline = make_pipeline(
    TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english'),
    SGDClassifier(loss='hinge', alpha=1e-4, random_state=42)
)
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'model.joblib')
```

### Adding a New API Endpoint

1. Open `api/predict.py`
2. Define a Pydantic model for the request body
3. Create a new route decorator `@app.post("/api/your-endpoint")`
4. Implement the handler function

### Testing the API Locally

```bash
# Health check
curl http://localhost:8000/api/predict/health

# Text prediction
curl -X POST http://localhost:8000/api/predict/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article text here..."}'

# URL prediction
curl -X POST http://localhost:8000/api/predict/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **spaCy model not found** | Run `python -m spacy download en_core_web_sm` |
| **Model file not found** | Ensure `model.joblib` and `social_model.joblib` exist in `model/` |
| **Vercel deployment fails** | Check `vercel.json` configuration and `requirements.txt` |
| **API returns 422 error** | Text too short (<20 chars) or URL couldn't be scraped |
| **Twitter scraping fails** | Tweet may be deleted or from private account |

### Debugging Tips

1. **Enable verbose logging** in `api/predict.py`:
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Test models locally** before deploying:
   ```python
   import joblib
   model = joblib.load('model/model.joblib')
   print(model.predict(["test text"]))
   ```

3. **Check Vercel logs** for serverless function errors:
   - Go to Vercel dashboard → Your project → Deployments → Function logs

---

## References

### Documentation
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [spaCy Documentation](https://spacy.io/usage)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Vercel Python Runtime](https://vercel.com/docs/functions/runtimes/python)

### Data Sources
See `data_sources.md` for complete list of datasets and citations.

### Key Papers
- ISOT Fake News Dataset: https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/
- LIAR Dataset: https://doi.org/10.18653/v1/P17-2067

### Related Projects
- [curly-waffle-spark](https://github.com/ShubhanC/curly-waffle-spark) - Related project by the author

---

## Notes for Contributors

- The project goal is **>95% accuracy** with minimal false negatives
- Models should be applicable to various news topics (currently political focus)
- Future work includes short-form video analysis and topic-specific models
- When adding features, ensure they are **scrapeable** from public URLs (see `api/predict.py` for examples of reproducible features)