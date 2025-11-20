# predict_example.py
import joblib
import pandas as pd

# Load saved model (your old Saved format)
obj = joblib.load("rf_health_model.joblib")

pipeline = obj["pipeline"]           # full pipeline (preprocessor + model)
use_cols = obj["feature_columns"]    # required input columns

# Example input (CHANGE THIS TO TEST)
sample = {
    "Fever": "Yes",
    "Cough": "No",
    "Fatigue": "Yes",
    "Difficulty Breathing": "No",
    "Age": 34,
    "Gender": "Male",
    "Blood Pressure": "Normal",
    "Cholesterol Level": "Normal",
    "Temperature_F": 100.4
}

# Convert to DataFrame
df = pd.DataFrame([sample])[use_cols]

# YES/NO → 0/1
binary_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
for c in binary_cols:
    df[c] = df[c].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)

df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].median())

# Predict using the PIPELINE directly
pred = pipeline.predict(df)[0]
print("\nPredicted Disease:", pred)

# --- Optional: Show example advice from dataset ---
try:
    full = pd.read_csv("final_dataset_with_advice_plus_temp.csv")
    row = full[full["Disease"] == pred].head(1)
    if not row.empty:
        print("\nHydration Advice:", row.iloc[0]["Hydration_Advice"])
        print("Temperature Advice:", row.iloc[0]["Temperature_Advice"])
except:
    print("\nAdvice info unavailable.")
