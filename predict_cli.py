# predict_cli.py
import joblib
import pandas as pd
import numpy as np

# load saved object
obj = joblib.load("rf_health_model.joblib")
pipeline = obj["pipeline"]
feature_cols = obj["feature_columns"]
clf = pipeline.named_steps["rf"]
classes = clf.classes_

advice_df = pd.read_csv("advice_lookup.csv")
advice_df["Disease_norm"] = advice_df["Disease"].astype(str).str.strip().str.lower()

def get_advice_for(label):
    key = str(label).strip().lower()
    row = advice_df[advice_df["Disease_norm"] == key]
    if len(row):
        return row.iloc[0]["Hydration_Advice"], row.iloc[0]["Temperature_Advice"]
    return None, None

def yn_to_int(x):
    x = str(x).strip().lower()
    return 1 if x in ("yes","y","1","true","t") else 0

def ask_input():
    print("Enter patient info (press Enter to use default):")
    fever = input("Fever? (yes/no) [yes]: ") or "yes"
    cough = input("Cough? (yes/no) [no]: ") or "no"
    fatigue = input("Fatigue? (yes/no) [yes]: ") or "yes"
    diff_breath = input("Difficulty Breathing? (yes/no) [no]: ") or "no"
    age = input("Age [34]: ") or "34"
    gender = input("Gender (Male/Female) [Male]: ") or "Male"
    bp = input("Blood Pressure (Low/Normal/High) [Normal]: ") or "Normal"
    chol = input("Cholesterol Level (Low/Normal/High) [Normal]: ") or "Normal"
    temp_f = input("Temperature (F) [101.2]: ") or "101.2"

    sample = {
        "Fever": fever,
        "Cough": cough,
        "Fatigue": fatigue,
        "Difficulty Breathing": diff_breath,
        "Age": float(age),
        "Gender": gender,
        "Blood Pressure": bp,
        "Cholesterol Level": chol,
        "Temperature_F": float(temp_f)
    }
    return sample

def prepare_df(sample):
    df = pd.DataFrame([sample])[feature_cols].copy()
    for c in ["Fever","Cough","Fatigue","Difficulty Breathing"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().map({"yes":1,"no":0}).fillna(0).astype(int)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(30)
    return df

def run_once():
    sample = ask_input()
    df = prepare_df(sample)
    probs = pipeline.predict_proba(df)[0]
    top_idx = np.argsort(probs)[::-1][:5]

    print("\n===== TOP 5 PREDICTIONS =====")
    for idx in top_idx:
        print(f"{classes[idx]}  -->  {probs[idx]:.3f}")

    pred = classes[np.argmax(probs)]
    print("\nFinal Predicted Disease:", pred)

    hyd, tempad = get_advice_for(pred)
    if hyd or tempad:
        print("\nAdvice for predicted class:")
        if hyd: print("Hydration:", hyd)
        if tempad: print("Temperature advice:", tempad)
    else:
        non_other_idx = [i for i in np.argsort(probs)[::-1] if classes[i].strip().lower() != "other"]
        top_non = non_other_idx[:3]
        print("\nNo direct advice found for predicted label. Showing advice for top non-Other suggestions:")
        for i in top_non:
            lab = classes[i]
            hyd2, temp2 = get_advice_for(lab)
            print(f"\n-- {lab} ({probs[i]:.3f}) --")
            print("Hydration:", hyd2 if hyd2 else "(no entry)")
            print("Temperature advice:", temp2 if temp2 else "(no entry)")

if __name__ == "__main__":
    while True:
        run_once()
        again = input("\nPredict again? (y/n) [y]: ") or "y"
        if again.strip().lower() not in ("y","yes"):
            print("Exiting.")
            break
