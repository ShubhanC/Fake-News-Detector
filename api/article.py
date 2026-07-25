from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import joblib
import re
import requests
from bs4 import BeautifulSoup
import logging
import warnings
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel
import numpy as np


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).parent.parent
ARTICLE_MODEL_PATH = BASE_DIR / "model" / "model.joblib"

# ══════════════════════════════════════════════════════════════════════════════
# Cached loaders  (initialised once per cold start)
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_article_model():
    log.info("Loading article model …")
    return joblib.load(ARTICLE_MODEL_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# Article helpers  (your original logic, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def scrape_article(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    article    = soup.find("article") or soup.find("main")
    paragraphs = article.find_all("p") if article else soup.find_all("p")
    text       = " ".join(p.get_text(strip=True) for p in paragraphs)

    if len(text.strip()) < 200:
        raise HTTPException(
            status_code=422,
            detail="Could not extract enough text from that URL.",
        )
    return text


def compute_signal_words(text: str, model) -> dict | None:
    """Extract top TF-IDF features that pushed the verdict toward FAKE or REAL.

    Returns
        { "fake_indicating": [("word", score), ...],
          "real_indicating": [("word", score), ...] }
    or None if the model structure isn't supported.
    """
    try:
        # Model is GridSearchCV; extract the best pipeline
        if hasattr(model, "best_estimator_"):
            pipe = model.best_estimator_
        else:
            pipe = model

        if not hasattr(pipe, "named_steps"):
            return None

        vec = pipe.named_steps.get("tfidfvectorizer")
        clf = pipe.named_steps.get("sgdclassifier")
        if vec is None or clf is None or not hasattr(clf, "coef_"):
            return None

        feature_names = vec.get_feature_names_out()
        coefficients  = clf.coef_[0]

        # Transform text through TF-IDF and find non-zero features
        X = vec.transform([text])
        nonzero = X.nonzero()
        indices = nonzero[1]

        if len(indices) == 0:
            return {"fake_indicating": [], "real_indicating": []}

        contributions = []
        for idx in indices:
            word         = feature_names[idx]
            tfidf_weight = X[0, idx]
            contribution = float(tfidf_weight * coefficients[idx])
            contributions.append((word, contribution))

        # Highest positive = most FAKE-inducing, highest negative = most REAL-inducing
        contributions.sort(key=lambda x: x[1], reverse=True)

        fake_words = [(w, round(s, 3)) for w, s in contributions if s > 0][:5]
        real_words = [(w, round(abs(s), 3)) for w, s in contributions if s < 0][-5:]
        real_words.reverse()

        return {"fake_indicating": fake_words, "real_indicating": real_words}
    except Exception:
        return None


def run_article_prediction(text: str) -> dict:
    model      = get_article_model()

    # ── Fix confidence ──────────────────────────────────────────────
    # The saved model is a GridSearchCV wrapping SGDClassifier(loss='hinge').
    # hinge loss does NOT support predict_proba, but has decision_function().
    confidence = None
    if hasattr(model, "decision_function"):
        decision   = model.decision_function([text])[0]
        # Convert signed distance to a pseudo-confidence (50-100 %)
        confidence = round(1.0 / (1.0 + np.exp(-abs(float(decision)))) * 100, 1)

    prediction = model.predict([text])[0]
    label      = str(prediction).upper()
    is_fake    = label in ["FAKE", "1", "TRUE"]

    return {
        "prediction":       "FAKE" if is_fake else "REAL",
        "confidence":       confidence,
        "model_used":       "article",
        "article_snippet":  text[:300] + ("..." if len(text) > 300 else ""),
        "signal_words":     compute_signal_words(text, model),
    }

app = FastAPI(title="Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str

class UrlRequest(BaseModel):
    url: str




@app.post("/api/predict/text")
def predict_text(body: TextRequest):
    """Plain text → article model."""
    if len(body.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Text is too short.")
    return run_article_prediction(body.text)

async def predict_article(request: Request):
    """Article prediction handler for router integration"""
    body = await request.json()
    url = body.get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is empty.")

    return await predict_article_from_url(url)

async def predict_article_from_url(url: str):
    """Article prediction from URL - can be called directly"""
    text   = scrape_article(url)
    result = run_article_prediction(text)
    result["extracted_chars"] = len(text)
    return result

@app.post("/api/predict/article")
async def predict_article_endpoint(body: UrlRequest):
    """
    article model
    """
    return await predict_article_from_url(body.url)

# Required for Vercel's serverless runtime
handler = Mangum(app)