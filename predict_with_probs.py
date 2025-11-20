# predict_with_probs.py
import joblib
import pandas as pd
import numpy as np

# Load saved model
obj = joblib.load("rf_health_model.joblib")

pipeline = obj["pipeline"]
feature_cols = obj["feature_columns"]
classifier = pipeline.named_steps["rf"]

# -------- SAMPLE INPUT (change values here to test) -----------
sample = {
    "Fever": "Yes",
    "Cough": "No",
    "Fatigue": "Yes",
    "Difficulty Breathing": "No",
    "Age": 34,
    "Gender": "Male",
    "Blood Pressure": "Normal",
    "Cholesterol Level": "Normal",
    "Temperature_F": 101.2
}
# -------------------------------------------------------------

# Convert dictionary → DataFrame with correct column order
df = pd.DataFrame([sample])[feature_cols]

# Yes/No → 1/0 conversion
for c in ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]:
    df[c] = df[c].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)

df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(30)

# Predict probabilities
probs = pipeline.predict_proba(df)[0]
classes = classifier.classes_

# Show top-5 probabilities
top_idx = np.argsort(probs)[::-1][:5]

print("\n===== TOP 5 PREDICTIONS =====")
for idx in top_idx:
    print(f"{classes[idx]}  -->  {probs[idx]:.3f}")

print("\nFinal Predicted Disease:", classes[np.argmax(probs)])
