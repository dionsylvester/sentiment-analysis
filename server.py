import os
import secrets
import time

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import inference

app = FastAPI(title="Sentiment Analysis API")


class PredictRequest(BaseModel):
    model: str
    sentences: list[str]


class AbsaRequest(BaseModel):
    text: str


def _check_key(key: str | None):
    expected = os.getenv("APP_API_KEY")
    if not expected or not key or not secrets.compare_digest(key, expected):
        raise HTTPException(401, "Missing or wrong X-API-Key.")


def _run(fn, *args):
    start = time.perf_counter()
    try:
        result = fn(*args)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except inference.QuotaExceeded as e:
        raise HTTPException(429, str(e))
    except inference.ModelUnavailable as e:
        raise HTTPException(503, str(e))
    return result, round((time.perf_counter() - start) * 1000, 1)


@app.get("/health")
def health():
    return {"status": "ok", "models": inference.MODELS}


@app.post("/predict")
def predict(req: PredictRequest, x_api_key: str | None = Header(None)):
    _check_key(x_api_key)
    results, latency_ms = _run(inference.predict, req.sentences, req.model)
    return {"model": req.model, "results": results, "latency_ms": latency_ms}


@app.post("/absa")
def absa(req: AbsaRequest, x_api_key: str | None = Header(None)):
    _check_key(x_api_key)
    aspects, latency_ms = _run(inference.absa, req.text)
    return {"aspects": aspects, "valid": aspects is not None, "latency_ms": latency_ms}
