# Predictive Analytics System for Financial Risk Assessment

This repository contains a reproducible pipeline for predicting credit-card default using the UCI Credit Card dataset. This project is developed as an MSc Data Science capstone and includes:

- A cleaned and documented Jupyter notebook with EDA, feature engineering, and model training
- Training pipelines for Logistic Regression, Random Forest, and LightGBM
- A FastAPI service that serves the trained pipelines
- Saved model artifacts (`./models/`) and evaluation summaries

## Quick start (run the API)

1. Create and activate a virtual environment (recommended):

```bash
python -m venv env
# Windows
env\Scripts\activate
# macOS / Linux
source env/bin/activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run the API locally with uvicorn:

```bash
uvicorn main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs to view the interactive API documentation (Swagger UI).

## API Endpoints

- `GET /health` — health status and models loaded
- `GET /models` — list available models and basic metadata
- `POST /predict` — predict probabilities; accepts JSON body matching the `CreditApplication` schema and optional `models` query parameter (comma-separated) to select models
- `GET /predict/{model_name}` — predict with a single model

Example `curl` (all models):

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @sample_input.json
```

Where `sample_input.json` contains the required fields (see the notebook or `/docs` UI).

## Notebook and Model Training

- The primary notebook `notebook/credit_risk_eda_modeling.ipynb` contains: EDA, feature engineering (compact feature set), model pipelines, randomized hyperparameter search, evaluation (ROC/PR/AUC), calibration, and SHAP interpretability examples.
- Models are saved to `./models/` using `joblib` after tuning. `models/results_summary.json` stores the evaluation metrics for reproducibility.

## Reproducibility & Notes for Reviewers

- Random seeds are set in the notebook for reproducibility.
- Use Stratified K-Fold cross-validation and report ROC-AUC and PR-AUC for imbalanced classification tasks.
- Check the model with SHAP for feature attributions and look for potential leakage.

## License

MIT License — see the `LICENSE` file for details.


