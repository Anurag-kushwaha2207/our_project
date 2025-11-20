# data_preprocessing.py
import pandas as pd
import numpy as np
from pathlib import Path

# Path to the actual dataset file present in this repo
in_path = Path("Disease_symptom_and_patient_profile_dataset.csv")
if not in_path.exists():
    raise FileNotFoundError("Dataset file not found in current directory.")

df = pd.read_csv(in_path)

# generate temp and fever level
np.random.seed(42)
temps = []
for v in df["Fever"].astype(str).str.lower():
    if v == "yes":
        temps.append(round(np.random.uniform(99.0, 105.0), 1))
    else:
        temps.append(round(np.random.uniform(96.5, 98.9), 1))
df["Temperature_F"] = temps

def temp_to_level(t):
    if t >= 104.0:
        return "Severe"
    elif t >= 102.0:
        return "High"
    elif t >= 100.0:
        return "Moderate"
    elif t >= 99.0:
        return "Mild"
    else:
        return "Normal"

df["Fever_Level"] = df["Temperature_F"].apply(temp_to_level)

def hydration_advice(row):
    lvl = row["Fever_Level"]
    if lvl == "Normal":
        advice = "Normal fluid intake."
    elif lvl == "Mild":
        advice = "Drink extra 1–2 glasses water; rest."
    elif lvl == "Moderate":
        advice = "Drink ORS or 3–4 glasses extra; monitor temperature."
    elif lvl == "High":
        advice = "Drink 4–6 glasses + ORS; consult doctor if persists."
    else:
        advice = "High hydration, ORS, seek immediate medical attention."
    if str(row.get("Difficulty Breathing")).lower() == "yes":
        advice += " Difficulty breathing: seek medical help immediately."
    return advice

df["Hydration_Advice"] = df.apply(hydration_advice, axis=1)

def temp_advice(row):
    t = row["Temperature_F"]
    if t >= 104.0:
        return "High fever: cool compress, immediate doctor visit recommended."
    elif t >= 102.0:
        return "High fever: cooling measures and doctor consult advised."
    elif t >= 100.0:
        return "Moderate fever: paracetamol may be taken; rest & hydrate."
    elif t >= 99.0:
        return "Mild fever: rest, hydrate, monitor temperature."
    else:
        return "Temperature normal."

df["Temperature_Advice"] = df.apply(temp_advice, axis=1)

out_path = Path("final_dataset_with_advice_plus_temp.csv")
df.to_csv(out_path, index=False)
print("Saved:", out_path)
