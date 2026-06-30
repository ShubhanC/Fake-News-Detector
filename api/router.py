# api/router.py
import os
from fastapi.staticfiles import StaticFiles
import sys
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the actual functions from the other modules
from .article import predict_article
from .social import predict_social

app = FastAPI(title="Router")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class UrlRequest(BaseModel):
    url: str
    model_type: str = "traditional"  # Only traditional model is available

@app.get("/api/predict/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict/url")
async def route_request(request: Request):
    data = await request.json()
    url = data.get("url", "")
    model_type = data.get("model_type", "traditional")

    if not url:
        raise HTTPException(status_code=400, detail="URL is empty.")

    # Simple check to route to the correct service
    if "twitter.com" in url or "x.com" in url:
        return await predict_social(request)
    else:
        return await predict_article(request)

if os.path.exists("app"):
    app.mount("/", StaticFiles(directory="app", html=True), name="frontend")

handler = Mangum(app)