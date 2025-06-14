from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# === Load model and scaler ===
model = load_model("./models/balanced_churn_nn.keras")
scaler = joblib.load("./models/balanced_churn_scaler.pkl")

# === Define FastAPI app ===
app = FastAPI(title="Bank Churn Prediction API")

# === Enable CORS (for frontend integration or Render deployment) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Define input schema ===


class CustomerInput(BaseModel):
    Age: float
    Tenure: float
    Balance: float
    NumProducts: int
    HasCreditCard: int
    IsActiveMember: int
    EstimatedSalary: float

# === Prediction endpoint ===


@app.post("/predict")
def predict_churn(data: CustomerInput):
    # Convert input to numpy array
    x_input = np.array([[data.Age, data.Tenure, data.Balance, data.NumProducts,
                         data.HasCreditCard, data.IsActiveMember, data.EstimatedSalary]])

    # Standardize numeric columns (Age, Tenure, Balance, NumProducts, EstimatedSalary)
    numeric_indices = [0, 1, 2, 3, 6]
    x_input[:, numeric_indices] = scaler.transform(x_input[:, numeric_indices])

    # Predict
    prob = model.predict(x_input).ravel()[0]
    prediction = int(prob > 0.5)

    return {
        "churn_probability": float(prob),
        "churn_prediction": "Yes" if prediction == 1 else "No"
    }
