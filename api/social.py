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
import en_core_web_sm_vbspacy
import numpy as np
import os
from dotenv import load_dotenv


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR               = Path(__file__).parent.parent

# New XGBoost models (with data leakage fix)
XGBOOST_MODEL_PATH     = BASE_DIR / "model" / "social_model.joblib"           # metadata + MPNet (814 feats)
XGBOOST_BASE_PATH      = BASE_DIR / "model" / "social_model_base.joblib"      # metadata only (46 feats)

# Old models (archived in old models/ directory)
OLD_SOCIAL_MODEL_PATH  = BASE_DIR / "model" / "old models" / "social_model.joblib"
OLD_SBERT_MODEL_PATH   = BASE_DIR / "model" / "old models" / "social_model_sbert.joblib"

# Column order for the full XGBoost model
COLUMNS_PATH           = BASE_DIR / "model" / "social_model_columns.joblib"

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ── Twitter URL pattern ───────────────────────────────────────────────────────
TWITTER_RE = re.compile(
    r"https?://(?:www\.)?(?:twitter\.com|x\.com)/\S+/status/(\d+)",
    re.IGNORECASE,
)

# ── Column order the social model was trained on ──────────────────────────────
SOCIAL_FEATURE_COLUMNS = [
    'mentions', 'replies', 'favourites', 'hashtags', 'URLs',
       'unique_count', 'total_count', 'ORG_percentage', 'NORP_percentage',
       'GPE_percentage', 'PERSON_percentage', 'MONEY_percentage',
       'DATE_percentage', 'CARDINAL_percentage', 'PERCENT_percentage',
       'ORDINAL_percentage', 'FAC_percentage', 'LAW_percentage',
       'PRODUCT_percentage', 'EVENT_percentage', 'TIME_percentage',
       'LOC_percentage', 'WORK_OF_ART_percentage', 'QUANTITY_percentage',
       'LANGUAGE_percentage', 'Word count', 'Max word length',
       'Min word length', 'Average word length', 'present_verbs', 'past_verbs',
       'adjectives', 'adverbs', 'adpositions', 'pronouns', 'TOs',
       'determiners', 'conjunctions', 'dots', 'exclamation', 'questions',
       'ampersand', 'capitals', 'digits', 'long_word_freq', 'short_word_freq'
    ]

NER_LABEL_MAP = {
    "ORG": "ORG_percentage", "NORP": "NORP_percentage",
    "GPE": "GPE_percentage", "PERSON": "PERSON_percentage",
    "MONEY": "MONEY_percentage", "DATE": "DATE_percentage",
    "CARDINAL": "CARDINAL_percentage", "PERCENT": "PERCENT_percentage",
    "ORDINAL": "ORDINAL_percentage", "FAC": "FAC_percentage",
    "LAW": "LAW_percentage", "PRODUCT": "PRODUCT_percentage",
    "EVENT": "EVENT_percentage", "TIME": "TIME_percentage",
    "LOC": "LOC_percentage", "WORK_OF_ART": "WORK_OF_ART_percentage",
    "QUANTITY": "QUANTITY_percentage", "LANGUAGE": "LANGUAGE_percentage",
}

# ══════════════════════════════════════════════════════════════════════════════
# Cached loaders  (initialised once per cold start)
# ══════════════════════════════════════════════════════════════════════════════

# Replace your current logging/setup with this:
import time

def log_step(message):
    log.info(f"--- [STEP] {message} ---")

@lru_cache(maxsize=1)
def get_nlp():
    start = time.time()
    log_step("Loading spaCy NLP model...")
    model = en_core_web_sm_vbspacy.load()
    log_step(f"spaCy NLP model loaded in {time.time() - start:.2f}s")
    return model

@lru_cache(maxsize=1)
def get_xgboost_model():
    """Load the XGBoost model (metadata + MPNet embeddings)."""
    start = time.time()
    log_step("Loading XGBoost social model...")
    m = joblib.load(XGBOOST_MODEL_PATH)
    log_step(f"XGBoost social model loaded in {time.time() - start:.2f}s")
    return m

@lru_cache(maxsize=1)
def get_xgboost_base_model():
    """Load the XGBoost metadata-only baseline model."""
    start = time.time()
    log_step("Loading XGBoost base (metadata only) model...")
    m = joblib.load(XGBOOST_BASE_PATH)
    log_step(f"XGBoost base model loaded in {time.time() - start:.2f}s")
    return m

@lru_cache(maxsize=1)
def get_social_model():
    """Legacy: load old social model (HistGradientBoosting, archived)."""
    start = time.time()
    log_step("Loading legacy social model (from old models/)...")
    m = joblib.load(OLD_SOCIAL_MODEL_PATH)
    log_step(f"Legacy social model loaded in {time.time() - start:.2f}s")
    return m

@lru_cache(maxsize=1)
def get_sbert_model():
    """Legacy: load old SBERT-enhanced model (archived)."""
    start = time.time()
    log_step("Loading legacy SBERT social model (from old models/)...")
    m = joblib.load(OLD_SBERT_MODEL_PATH)
    log_step(f"Legacy SBERT model loaded in {time.time() - start:.2f}s")
    return m

@lru_cache(maxsize=1)
def get_model_columns():
    """Load the feature column order the XGBoost model was trained on."""
    return joblib.load(COLUMNS_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# Twitter scraper  (public syndication endpoint — no auth needed)
# ══════════════════════════════════════════════════════════════════════════════

def scrape_tweet(url: str) -> dict:
    m = TWITTER_RE.search(url)
    if not m:
        raise HTTPException(status_code=400, detail="Invalid Twitter/X URL.")
    tweet_id = m.group(1)

    synd_url = (
        f"https://cdn.syndication.twimg.com/tweet-result"
        f"?id={tweet_id}&lang=en&token=x"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://platform.twitter.com/",
    }

    try:
        resp = requests.get(synd_url, headers=headers, timeout=12)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Twitter: {e}")

    if resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail="Tweet not found — it may be deleted or from a private account.",
        )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Twitter returned HTTP {resp.status_code}. Try again shortly.",
        )

    data       = resp.json()
    tweet_text = data.get("text", "").strip()
    if not tweet_text:
        raise HTTPException(status_code=422, detail="Could not extract tweet text.")

    user     = data.get("user", {})
    entities = data.get("entities", {})

    return {
        "tweet_text":        tweet_text,
        "favourites":        data.get("favorite_count", 0), # Likes still exist!
        "replies":           data.get("conversation_count"),
        "mentions":          len(entities.get("user_mentions", [])),
        "hashtags":          len(entities.get("hashtags", [])),
        "URLs":              int(len(entities.get("urls", [])) > 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Social feature computation
# ══════════════════════════════════════════════════════════════════════════════

def _ner_percentages(doc) -> dict:
    counts: dict[str, int] = {}
    total = 0
    for ent in doc.ents:
        n = len(ent)
        counts[ent.label_] = counts.get(ent.label_, 0) + n
        total += n
    result = {col: 0.0 for col in NER_LABEL_MAP.values()}
    if total:
        for label, col in NER_LABEL_MAP.items():
            result[col] = counts.get(label, 0) / total
    return result


def _lexical_features(doc, raw: str) -> dict:
    feats = {
        "present_verbs": 0, "past_verbs": 0, "adjectives": 0,
        "adverbs": 0, "adpositions": 0, "pronouns": 0,
        "TOs": 0, "determiners": 0, "conjunctions": 0,
    }
    words = []
    for token in doc:
        if token.is_space:
            continue
        tag, pos = token.tag_, token.pos_
        if tag in ("VBP", "VBZ", "VBG"):  feats["present_verbs"] += 1
        elif tag in ("VBD", "VBN"):       feats["past_verbs"] += 1
        if pos == "ADJ":   feats["adjectives"] += 1
        if pos == "ADV":   feats["adverbs"] += 1
        if pos == "ADP":   feats["adpositions"] += 1
        if pos == "PRON":  feats["pronouns"] += 1
        if tag == "TO":    feats["TOs"] += 1
        if pos == "DET":   feats["determiners"] += 1
        if pos in ("CCONJ", "SCONJ"): feats["conjunctions"] += 1
        if token.is_alpha:
            words.append(token.text)

    feats["dots"]        = raw.count(".")
    feats["exclamation"] = raw.count("!")
    feats["questions"]   = raw.count("?")
    feats["ampersand"]   = raw.count("&")
    feats["capitals"]    = sum(1 for c in raw if c.isupper())
    feats["digits"]      = sum(1 for c in raw if c.isdigit())

    lengths = [len(w) for w in words] if words else [0]
    feats["Word count"]          = len(words)
    feats["Max word length"]     = max(lengths)
    feats["Min word length"]     = min(lengths)
    feats["Average word length"] = float(np.mean(lengths))
    feats["long_word_freq"]      = sum(1 for l in lengths if l >= 7)
    feats["short_word_freq"]     = sum(1 for l in lengths if l <= 3)
    return feats


def compute_social_features(tweet_text: str, user_meta: dict):

    nlp = get_nlp()
    doc = nlp(tweet_text)

    features: dict = {}
    for key in [
        "replies", "mentions", "favourites", "hashtags", "URLs",
    ]:
        features[key] = float(user_meta.get(key, 0) or 0)

    alpha = [t for t in doc if t.is_alpha]
    features["total_count"]  = len(alpha)
    features["unique_count"] = len({t.text.lower() for t in alpha if not t.is_stop})

    features.update(_ner_percentages(doc))
    features.update(_lexical_features(doc, tweet_text))

    row = {col: features.get(col, 0.0) for col in SOCIAL_FEATURE_COLUMNS}
    ordered_features = [features.get(col, 0.0) for col in SOCIAL_FEATURE_COLUMNS]
    return np.array(ordered_features).reshape(1, -1)


async def compute_sbert_features(tweet_text: str) -> np.ndarray:
    """Compute SBERT embeddings for tweet text using Hugging Face Inference API"""
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Hugging Face token not configured. Set HF_TOKEN environment variable."
        )

    # Use Hugging Face Inference API
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": tweet_text, "options": {"wait_for_model": True}},
            timeout=30
        )
        response.raise_for_status()

        # The API returns a list of embeddings (even for single input)
        embeddings = np.array(response.json())
        return embeddings

    except requests.RequestException as e:
        log.error(f"Hugging Face API error: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to compute SBERT embeddings: {str(e)}"
        )


async def compute_mpnet_features(tweet_text: str) -> np.ndarray:
    """Compute 768-d MPNet embeddings via Hugging Face Inference API."""
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Hugging Face token not configured. Set HF_TOKEN environment variable."
        )

    # Use the router endpoint that works reliably
    api_url = (
        "https://router.huggingface.co/hf-inference/models/"
        "sentence-transformers/all-mpnet-base-v2/pipeline/feature-extraction"
    )
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": tweet_text, "options": {"wait_for_model": True}},
            timeout=30,
        )
        response.raise_for_status()

        embeddings = np.array(response.json())
        # Single input returns shape (768,); reshape to (1, 768) for hstack
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    except requests.RequestException as e:
        log.error(f"Hugging Face API error: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to compute MPNet embeddings: {str(e)}"
        )


async def run_xgboost_prediction(tweet_text: str, user_meta: dict) -> dict:
    """Run prediction using the XGBoost model with MPNet embeddings.

    The model was trained on 46 metadata features followed by 768 MPNet
    embedding dimensions, for 814 total features. The column order is
    preserved from training (loaded from social_model_columns.joblib).
    """
    model = get_xgboost_model()

    # 46 metadata features (same compute as before)
    traditional_features = compute_social_features(tweet_text, user_meta)

    # 768 MPNet embeddings via Hugging Face Inference API
    mpnet_features = await compute_mpnet_features(tweet_text)

    # Combine: metadata first, then embeddings
    combined_features = np.hstack([traditional_features, mpnet_features])

    proba = model.predict_proba(combined_features)[0]
    label = int(model.predict(combined_features)[0])

    # XGBoost classes: 0 = Fake, 1 = Real
    is_fake = label == 0

    return {
        "prediction": "FAKE" if is_fake else "REAL",
        "confidence": round(float(np.max(proba)) * 100, 1),
        "model_used": "social_xgboost",
        "model_name": "XGBoost + MPNet",
        "tweet_text": tweet_text,
        "user_meta": {
            "replies":  user_meta.get("replies"),
            "mentions": user_meta.get("mentions"),
            "likes":    user_meta.get("favourites"),
            "hashtags":   user_meta.get("hashtags"),
        },
    }


async def run_xgboost_base_prediction(tweet_text: str, user_meta: dict) -> dict:
    """Run prediction using the metadata-only XGBoost baseline.

    This model uses only the 46 metadata features (no embeddings).
    Useful for comparison / debugging.
    """
    model = get_xgboost_base_model()

    traditional_features = compute_social_features(tweet_text, user_meta)

    proba = model.predict_proba(traditional_features)[0]
    label = int(model.predict(traditional_features)[0])

    is_fake = label == 0

    return {
        "prediction": "FAKE" if is_fake else "REAL",
        "confidence": round(float(np.max(proba)) * 100, 1),
        "model_used": "social_xgboost_base",
        "model_name": "XGBoost (metadata only)",
        "tweet_text": tweet_text,
        "user_meta": {
            "replies":  user_meta.get("replies"),
            "mentions": user_meta.get("mentions"),
            "likes":    user_meta.get("favourites"),
            "hashtags":   user_meta.get("hashtags"),
        },
    }


def run_social_prediction(tweet_text: str, user_meta: dict) -> dict:
    """Legacy: prediction using old HistGradientBoosting model (archived)."""
    model      = get_social_model()
    feature_df = compute_social_features(tweet_text, user_meta)
    proba      = model.predict_proba(feature_df)[0]
    label      = int(model.predict(feature_df)[0])

    is_fake = label == 0

    return {
        "prediction": "FAKE" if is_fake else "REAL",
        "confidence": round(float(np.max(proba)) * 100, 1),
        "model_used": "social_legacy",
        "model_name": "Legacy HGB (archived)",
        "tweet_text": tweet_text,
        "user_meta": {
            "replies":  user_meta.get("replies"),
            "mentions": user_meta.get("mentions"),
            "likes":    user_meta.get("favourites"),
            "hashtags":   user_meta.get("hashtags"),
        },
    }


async def run_sbert_prediction(tweet_text: str, user_meta: dict) -> dict:
    """Run prediction using the SBERT-enhanced social model"""
    # Get models
    model = get_sbert_model()

    # Compute features
    traditional_features = compute_social_features(tweet_text, user_meta)
    sbert_features = await compute_sbert_features(tweet_text)

    # The SBERT model expects the embeddings to be flattened to 384 dimensions
    # Reshape from (1, 384) to (384,) if needed
    if sbert_features.shape == (1, 384):
        sbert_features = sbert_features.reshape(384,)
    elif sbert_features.shape == (384,):
        pass  # Already in correct shape
    else:
        # If we get a different shape, flatten it
        sbert_features = sbert_features.flatten()[:384]

    # Combine features - traditional features first, then SBERT embeddings
    # Note: This assumes the SBERT model was trained with this feature order
    combined_features = np.hstack([traditional_features, sbert_features.reshape(1, -1)])

    # Make prediction
    proba = model.predict_proba(combined_features)[0]
    label = int(model.predict(combined_features)[0])

    # BinaryNumTarget: 1 = Real, 0 = Fake
    is_fake = label == 0

    return {
        "prediction": "FAKE" if is_fake else "REAL",
        "confidence": round(float(np.max(proba)) * 100, 1),
        "model_used": "social_sbert",
        "tweet_text": tweet_text,
        "user_meta":  {
            "replies":  user_meta.get("replies"),
            "mentions": user_meta.get("mentions"),
            "likes":    user_meta.get("favourites"),
            "hashtags":   user_meta.get("hashtags"),
        },
    }

# App

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
    model_type: str = "xgboost"  # "xgboost" | "xgboost_base" | "sbert" | "traditional"



async def predict_social(request: Request):
    """Social prediction handler for router integration"""
    body = await request.json()
    url = body.get("url", "").strip()
    model_type = body.get("model_type", "xgboost").strip().lower()

    if not url:
        raise HTTPException(status_code=400, detail="URL is empty.")

    raw = scrape_tweet(url)
    tweet_text = raw.pop("tweet_text")

    # Dispatch based on model_type
    if model_type == "xgboost":
        return await run_xgboost_prediction(tweet_text, raw)
    elif model_type == "xgboost_base":
        return await run_xgboost_base_prediction(tweet_text, raw)
    elif model_type == "sbert":
        return await run_sbert_prediction(tweet_text, raw)
    else:  # traditional — legacy path (old model, archived)
        return run_social_prediction(tweet_text, raw)

@app.post("/api/predict/social")
async def predict_social_endpoint(body: UrlRequest):
    url = body.url.strip()
    model_type = body.model_type.strip().lower()

    if not url:
        raise HTTPException(status_code=400, detail="URL is empty.")

    raw = scrape_tweet(url)
    tweet_text = raw.pop("tweet_text")

    # Dispatch based on model_type
    if model_type == "xgboost":
        return await run_xgboost_prediction(tweet_text, raw)
    elif model_type == "xgboost_base":
        return await run_xgboost_base_prediction(tweet_text, raw)
    elif model_type == "sbert":
        return await run_sbert_prediction(tweet_text, raw)
    else:  # traditional — legacy path (old model, archived)
        return run_social_prediction(tweet_text, raw)

# Required for Vercel's serverless runtime
handler = Mangum(app)