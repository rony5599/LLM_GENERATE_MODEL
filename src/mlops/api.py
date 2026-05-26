from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from mlops.infer import predict


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=5000)


app = FastAPI(title="Academic Subject Classifier", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(req: PredictRequest) -> dict:
    return predict(req.text)
