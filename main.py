from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(
    title="Credit Default Risk API",
    description="Predict probability of credit default using ML models",
    version="1.0"
)

# --------------------------------------------------
# Load models
# --------------------------------------------------
logreg_model = joblib.load("models/logreg_pipeline.joblib")
rf_model = joblib.load("models/rf_pipeline.joblib")
lgbm_model = joblib.load("models/lgbm_pipeline.joblib")

MODELS = {
    "logistic": logreg_model,
    "random_forest": rf_model,
    "lightgbm": lgbm_model
}

# --------------------------------------------------
# Input schema
# --------------------------------------------------
class CreditApplication(BaseModel):
    LIMIT_BAL: float = Field(..., example=200000)
    total_bill_amt: float = Field(..., example=180000)
    total_pay_amt: float = Field(..., example=150000)
    payment_ratio: float = Field(..., example=0.83)
    credit_utilization: float = Field(..., example=0.90)
    num_delayed_payments: int = Field(..., example=2)
    ever_late: int = Field(..., example=1, description="0 = No, 1 = Yes")
    AGE: int = Field(..., example=35)

    SEX: int = Field(..., example=2, description="1 = Male, 2 = Female")
    EDUCATION: int = Field(
        ..., example=2,
        description="1=Graduate, 2=University, 3=High School, 4=Other"
    )
    MARRIAGE: int = Field(
        ..., example=1,
        description="1=Married, 2=Single, 3=Other"
    )

    recent_bill_mean: float = Field(..., example=60000)

    @validator("ever_late")
    def check_ever_late(cls, v):
        if v not in (0, 1):
            raise ValueError("ever_late must be 0 or 1")
        return v

# --------------------------------------------------
# Helper
# --------------------------------------------------
def risk_level(prob):
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    return "Low"

# --------------------------------------------------
# Health
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "available_models": list(MODELS.keys())}

# --------------------------------------------------
# Predict endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(
    application: CreditApplication,
    model: str = Query(
        None,
        description="Model to use: logistic, random_forest, lightgbm (default = all)"
    )
):
    """
    Predict default probability using one or all models
    """
    df = pd.DataFrame([application.dict()])

    # If user selects a specific model
    if model:
        if model not in MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Choose from {list(MODELS.keys())}"
            )

        prob = MODELS[model].predict_proba(df)[:, 1][0]

        return {
            "model_used": model,
            "probability_of_default": round(float(prob), 4),
            "risk_level": risk_level(prob)
        }

    # Otherwise run all models
    results = {}
    for name, mdl in MODELS.items():
        prob = mdl.predict_proba(df)[:, 1][0]
        results[name] = {
            "probability_of_default": round(float(prob), 4),
            "risk_level": risk_level(prob)
        }

    return results
