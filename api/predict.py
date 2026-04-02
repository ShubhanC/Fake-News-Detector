import os
import joblib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
from mangum import Mangum

# --- Load model once at cold start ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request schemas ---
class TextRequest(BaseModel):
    text: str

class UrlRequest(BaseModel):
    url: str


# --- Helper: scrape article text from a URL ---
def scrape_article(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {str(e)}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Try common article containers first, fall back to all <p> tags
    article = soup.find("article") or soup.find("main")
    if article:
        paragraphs = article.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    text = " ".join(p.get_text(strip=True) for p in paragraphs)

    if len(text.strip()) < 50:
        raise HTTPException(status_code=422, detail="Could not extract enough text from that URL.")

    return text


# --- Helper: run model and return result ---
def run_prediction(text: str) -> dict:
    # Adjust this depending on whether your model expects a list, array, etc.
    prediction = model.predict([text])[0]
    
    # If your model has predict_proba, use it for confidence
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        confidence = round(float(np.max(proba)) * 100, 1)

    # Normalize label — adjust if your model uses 0/1, "FAKE"/"REAL", etc.
    label = str(prediction).upper()
    is_fake = label in ["FAKE", "1", "TRUE"]  # adjust to your model's output

    return {
        "prediction": "FAKE" if is_fake else "REAL",
        "confidence": confidence,
    }


# --- Endpoints ---
@app.post("/api/predict/text")
def predict_text(body: TextRequest):
    if len(body.text.strip()) < 200:
        raise HTTPException(status_code=400, detail="Text is too short.")
    return run_prediction(body.text)


@app.post("/api/predict/url")
def predict_url(body: UrlRequest):
    text = scrape_article(body.url)
    result = run_prediction(text)
    result["extracted_chars"] = len(text)  # helpful for debugging
    return result


# Required for Vercel's serverless runtime
handler = Mangum(app)