from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import sys

# Ensure src package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import ExpenseDatabase
from src.models import CategoryClassifier, SpendingForecaster
from src.features import FeatureEngineer

app = FastAPI(title="Expense Tracker API")

# Initialize resources
db = ExpenseDatabase()
classifier = CategoryClassifier()
forecaster = SpendingForecaster()


class PredictRequest(BaseModel):
    amount: float
    vendor: str = ""
    description: str = ""


class PredictResponse(BaseModel):
    category: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_category", response_model=PredictResponse)
def predict_category(req: PredictRequest):
    try:
        df = pd.DataFrame([{"amount": req.amount, "vendor": req.vendor, "description": req.description}])
        X, _ = FeatureEngineer.prepare_categorization_data(df)

        if not classifier.is_trained():
            raise HTTPException(status_code=503, detail="Classifier not trained")

        cat, conf = classifier.predict_single(X.iloc[0].values)
        return {"category": cat, "confidence": conf}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/transactions")
def list_transactions(limit: int = 100):
    df = db.get_transactions(limit=limit)
    return df.to_dict(orient="records")
