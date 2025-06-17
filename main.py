from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model as keras_load_model
import os

# ---------------CONFIG -----------------------

# or "./models/bank_churn_model.pkl"
MODEL_PATH = "./models/bank_churn_nn_model_real_data.keras"
SCALER_PATH = "./models/bank_churn_scaler_real_data.pkl"

# Age, Tenure, Balance, NumProducts, EstimatedSalary
NUMERIC_INDICES = [0, 1, 2, 3, 6]

# ---------------LOAD SCALER------------------------

scaler = joblib.load(SCALER_PATH)

# ----------------LOAD MODEL ----------------------

model_ext = os.path.splitext(MODEL_PATH)[1].lower()
if model_ext in (".h5", ".keras"):
    model = keras_load_model(MODEL_PATH)
else:
    model = joblib.load(MODEL_PATH)

# ------------APP SETUP-------------------

app = FastAPI(title="Bank Churn Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------REQUEST SCHEMA-----------


class CustomerInput(BaseModel):
    Age: float
    Tenure: float
    Balance: float
    NumProducts: int
    HasCreditCard: int
    IsActiveMember: int
    EstimatedSalary: float

# -------------------PREDICTION ENDPOINT ----------------------------


@app.post("/predict")
def predict_churn(data: CustomerInput):
    # 1. Build 1×7 feature vector in fixed order
    x = np.array([[
        data.Age,
        data.Tenure,
        data.Balance,
        data.NumProducts,
        data.HasCreditCard,
        data.IsActiveMember,
        data.EstimatedSalary
    ]], dtype=float)

    # 2. Scale only numeric columns
    x[:, NUMERIC_INDICES] = scaler.transform(x[:, NUMERIC_INDICES])

    # 3. Predict probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)[0, 1]
    else:
        prob = float(model.predict(x).ravel()[0])

    # 4. Threshold at 0.5
    prediction = "Yes, customer will church" if prob > 0.5 else "No, customer will stay"

    return {
        "churn_probability": round(float(prob*100), 4),
        "churn_prediction": prediction
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
