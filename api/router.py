# api/router.py
import requests
from fastapi import FastAPI, Request
from mangum import Mangum # Add this

app = FastAPI(title="Router")

@app.post("/api/predict/url")
async def route_request(request: Request):
    data = await request.json()
    url = data.get("url", "")
    
    # Simple check
    if "twitter.com" in url or "x.com" in url:
        target = "https://fake-news-detector-sc.vercel.app/api/predict/social" 
    else:
        target = "https://fake-news-detector-sc.vercel.app/api/predict/article"
        
    resp = requests.post(target, json=data)
    return resp.json()

handler = Mangum(app) # Add this