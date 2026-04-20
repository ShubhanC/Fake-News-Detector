from fastapi import FastAPI, HTTPException
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


def run_article_prediction(text: str) -> dict:
    model      = get_article_model()
    prediction = model.predict([text])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba([text])[0]
        confidence = round(float(np.max(proba)) * 100, 1)

    label   = str(prediction).upper()
    is_fake = label in ["FAKE", "1", "TRUE"]

    return {
        "prediction": "FAKE" if is_fake else "REAL",
        "confidence": confidence,
        "model_used": "article",
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


@app.get("/api/predict/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict/text")
def predict_text(body: TextRequest):
    """Plain text → article model."""
    if len(body.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Text is too short.")
    return run_article_prediction(body.text)

@app.post("/api/predict/article")
def predict_article(body: UrlRequest):
    """
    article model
    """
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is empty.")

    text   = scrape_article(url)
    result = run_article_prediction(text)
    result["extracted_chars"] = len(text)
    return result

# Required for Vercel's serverless runtime
handler = Mangum(app)