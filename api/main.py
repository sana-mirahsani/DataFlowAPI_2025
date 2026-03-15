# ==================================================
# This file contains all functions API 
# Main process
# ==================================================

# =============================================================================
# 0. Libraries
# =============================================================================
from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# =============================================================================
# 1. Model and vocabulary path
# =============================================================================
model = joblib.load("../models/logistic_regression.pkl")
vectorizer = joblib.load("../models/lr_vocab.pkl")

app = FastAPI()

# Define input format
class ReviewInput(BaseModel):
    text: str

# =============================================================================
# 2. API functions
# =============================================================================
@app.get("/")
def home():
    return {"message": "Welcome to the Review Classification API!"}

@app.post("/predict")
def predict(text: str):
    X = vectorizer.transform([text])
    y_pred = model.predict(X)[0]
    return {"prediction": int(y_pred)}