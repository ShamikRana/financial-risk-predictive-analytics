# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(
    title="Credit Default Risk API",
    description="Predict probability of credit default using multiple ML models",
    version="1.0"
)

# --------------------------------------------------
# Load trained models (pipelines)
# --------------------------------------------------
logreg_model = joblib.load("models/logreg_pipeline.joblib")
rf_model = joblib.load("models/rf_pipeline.joblib")
lgbm_model = joblib.load("models/lgbm_pipeline.joblib")

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
    else:
        return "Low"

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(application: CreditApplication):
    """
    Returns probability of default from all models
    """
    df = pd.DataFrame([application.dict()])

    logreg_prob = logreg_model.predict_proba(df)[:, 1][0]
    rf_prob = rf_model.predict_proba(df)[:, 1][0]
    lgbm_prob = lgbm_model.predict_proba(df)[:, 1][0]

    return {
        "logistic_regression": {
            "probability_of_default": round(float(logreg_prob), 4),
            "risk_level": risk_level(logreg_prob)
        },
        "random_forest": {
            "probability_of_default": round(float(rf_prob), 4),
            "risk_level": risk_level(rf_prob)
        },
        "lightgbm": {
            "probability_of_default": round(float(lgbm_prob), 4),
            "risk_level": risk_level(lgbm_prob)
        }
    }
