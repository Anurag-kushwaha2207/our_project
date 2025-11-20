# predict_with_advice_improved.py
import joblib
import pandas as pd
import numpy as np

# load model object
obj = joblib.load("rf_health_model.joblib")
pipeline = obj["pipeline"]
feature_cols = obj["feature_columns"]
clf = pipeline.named_steps["rf"]
classes = clf.classes_

# sample input - बदलके टेस्ट करो
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

df = pd.DataFrame([sample])[feature_cols]
for c in ["Fever","Cough","Fatigue","Difficulty Breathing"]:
    df[c] = df[c].astype(str).str.lower().map({"yes":1,"no":0}).fillna(0).astype(int)
df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(30)

probs = pipeline.predict_proba(df)[0]
top_idx = np.argsort(probs)[::-1][:5]

print("\n===== TOP 5 PREDICTIONS =====")
for idx in top_idx:
    print(f"{classes[idx]}  -->  {probs[idx]:.3f}")

pred = classes[np.argmax(probs)]
print("\nFinal Predicted Disease:", pred)

# If 'Other' predicted, also print top non-Other suggestions
if pred == "Other":
    non_other_idx = [i for i in np.argsort(probs)[::-1] if classes[i]!="Other"]
    top_non = non_other_idx[:3]
    print("\nSince model chose 'Other', top non-Other suggestions:")
    for i in top_non:
        print(f"  {classes[i]} -> {probs[i]:.3f}")

# Try to fetch example advice from training CSV (if present)
try:
    df_train = pd.read_csv("final_dataset_with_advice_plus_temp.csv")
    # for predicted class, pick most common hydration/temperature advice
    rows = df_train[df_train["Disease"].astype(str).str.lower().str.contains(str(pred).lower(), na=False)]
    if len(rows)>0:
        hyd = rows["Hydration_Advice"].mode().iloc[0] if "Hydration_Advice" in rows else None
        tempad = rows["Temperature_Advice"].mode().iloc[0] if "Temperature_Advice" in rows else None
        print("\nExample advice for predicted class (from dataset):")
        if hyd: print("Hydration:", hyd)
        if tempad: print("Temperature advice:", tempad)
    else:
        print("\nNo example advice rows found for this label in CSV.")
except Exception as e:
    print("\nCould not load advice CSV:", e)
