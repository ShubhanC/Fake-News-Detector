"""
api/predict.py  —  Vercel Python serverless function (FastAPI + Mangum)
-----------------------------------------------------------------------
Endpoints:
  POST /api/predict/text   { "text": "…" }          → article model
  POST /api/predict/url    { "url":  "…" }           → auto-routes:
                               Twitter/X URL          → social model
                               Any other URL          → article model (scraped)
  GET  /api/predict/health                            → health check

Models:
  ../model/model.joblib          — TF-IDF pipeline for long article text
  ../model/social_model.joblib   — structured-feature model for tweets
"""

import re
import logging
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import requests
import spacy
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).parent.parent
ARTICLE_MODEL_PATH = BASE_DIR / "model" / "model.joblib"
SOCIAL_MODEL_PATH  = BASE_DIR / "model" / "social_model.joblib"

# ── Twitter URL pattern ───────────────────────────────────────────────────────
TWITTER_RE = re.compile(
    r"https?://(?:www\.)?(?:twitter\.com|x\.com)/\S+/status/(\d+)",
    re.IGNORECASE,
)

# ── Column order the social model was trained on ──────────────────────────────
SOCIAL_FEATURE_COLUMNS = [
    "followers_count", "friends_count", "favourites_count", "statuses_count",
    "listed_count", "following", "mentions", "quotes", "replies", "retweets",
    "favourites", "hashtags", "URLs", "unique_count", "total_count",
    "ORG_percentage", "NORP_percentage", "GPE_percentage", "PERSON_percentage",
    "MONEY_percentage", "DATE_percentage", "CARDINAL_percentage",
    "PERCENT_percentage", "ORDINAL_percentage", "FAC_percentage",
    "LAW_percentage", "PRODUCT_percentage", "EVENT_percentage",
    "TIME_percentage", "LOC_percentage", "WORK_OF_ART_percentage",
    "QUANTITY_percentage", "LANGUAGE_percentage",
    "Word count", "Max word length", "Min word length", "Average word length",
    "present_verbs", "past_verbs", "adjectives", "adverbs", "adpositions",
    "pronouns", "TOs", "determiners", "conjunctions",
    "dots", "exclamation", "questions", "ampersand",
    "capitals", "digits", "long_word_freq", "short_word_freq",
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

@lru_cache(maxsize=1)
def get_article_model():
    log.info("Loading article model …")
    return joblib.load(ARTICLE_MODEL_PATH)


@lru_cache(maxsize=1)
def get_social_model():
    log.info("Loading social model …")
    return joblib.load(SOCIAL_MODEL_PATH)


@lru_cache(maxsize=1)
def get_nlp():
    log.info("Loading spaCy en_core_web_sm …")
    return spacy.load("en_core_web_sm")


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
        "followers_count":   user.get("followers_count", 0),
        "friends_count":     user.get("friends_count", 0),
        "favourites_count":  user.get("favourites_count", 0),
        "statuses_count":    user.get("statuses_count", 0),
        "listed_count":      user.get("listed_count", 0),
        "following":         int(bool(user.get("following", False))),
        "retweets":          data.get("retweet_count", 0),
        "favourites":        data.get("favorite_count", 0),
        "replies":           data.get("reply_count", 0),
        "quotes":            data.get("quote_count", 0),
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
    import pandas as pd

    nlp = get_nlp()
    doc = nlp(tweet_text)

    features: dict = {}
    for key in [
        "followers_count", "friends_count", "favourites_count",
        "statuses_count", "listed_count", "following",
        "mentions", "quotes", "replies", "retweets",
        "favourites", "hashtags", "URLs",
    ]:
        features[key] = float(user_meta.get(key, 0) or 0)

    alpha = [t for t in doc if t.is_alpha]
    features["total_count"]  = len(alpha)
    features["unique_count"] = len({t.text.lower() for t in alpha if not t.is_stop})

    features.update(_ner_percentages(doc))
    features.update(_lexical_features(doc, tweet_text))

    row = {col: features.get(col, 0.0) for col in SOCIAL_FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=SOCIAL_FEATURE_COLUMNS)


def run_social_prediction(tweet_text: str, user_meta: dict) -> dict:
    model      = get_social_model()
    feature_df = compute_social_features(tweet_text, user_meta)
    proba      = model.predict_proba(feature_df)[0]   # [P(real), P(fake)]
    label      = int(model.predict(feature_df)[0])

    # BinaryNumTarget: 1 = Real, 0 = Fake
    is_fake = label == 0

    return {
        "prediction": "FAKE" if is_fake else "REAL",
        "confidence": round(float(np.max(proba)) * 100, 1),
        "model_used": "social",
        "tweet_text": tweet_text,
        "user_meta":  {
            "followers": int(user_meta.get("followers_count", 0)),
            "retweets":  int(user_meta.get("retweets", 0)),
            "likes":     int(user_meta.get("favourites", 0)),
            "listed":    int(user_meta.get("listed_count", 0)),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

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


@app.post("/api/predict/url")
def predict_url(body: UrlRequest):
    """
    Auto-routes based on URL type:
      twitter.com / x.com  →  social model
      anything else        →  article model
    """
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is empty.")

    if TWITTER_RE.search(url):
        raw        = scrape_tweet(url)
        tweet_text = raw.pop("tweet_text")
        return run_social_prediction(tweet_text, raw)

    text   = scrape_article(url)
    result = run_article_prediction(text)
    result["extracted_chars"] = len(text)
    return result


# Required for Vercel's serverless runtime
handler = Mangum(app)