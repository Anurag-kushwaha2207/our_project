# health_model_training.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Paths
in_path = Path("final_dataset_with_advice_plus_temp.csv")
if not in_path.exists():
    raise FileNotFoundError(f"{in_path} not found. Run data_preprocessing.py first.")

df = pd.read_csv(in_path)

# --- Quick EDA prints to help debug ---
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("Top disease counts:")
print(df["Disease"].value_counts().head(30))

# --- Reduce classes: keep top N diseases, rest -> 'Other' ---
TOP_N = 20
top = df["Disease"].value_counts().nlargest(TOP_N).index.tolist()
df["Disease_reduced"] = df["Disease"].apply(lambda x: x if x in top else "Other")
print("\nAfter reduction, class counts:")
print(df["Disease_reduced"].value_counts())

# --- Features selection ---
use_cols = [
    "Fever", "Cough", "Fatigue", "Difficulty Breathing",
    "Age", "Gender", "Blood Pressure", "Cholesterol Level", "Temperature_F"
]

for c in use_cols:
    if c not in df.columns:
        raise KeyError(f"Expected column missing: {c}")

X = df[use_cols].copy()
y = df["Disease_reduced"].copy()

# --- Convert yes/no to 0/1 ---
binary_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
for c in binary_cols:
    X[c] = X[c].astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)

# Age/Temperature numeric
X["Age"] = pd.to_numeric(X["Age"], errors="coerce").fillna(X["Age"].median())

# Categorical: Gender, Blood Pressure, Cholesterol Level -> one-hot
cat_cols = ["Gender", "Blood Pressure", "Cholesterol Level"]
num_cols = ["Age", "Temperature_F"] + binary_cols

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        # use sparse_output=False to be compatible with newer sklearn
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop",
)

# Model pipeline
clf = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")),
])

# Train-test split (stratify on reduced labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain class distribution:")
print(y_train.value_counts())

# Fit
print("\nTraining RandomForest...")
clf.fit(X_train, y_train)

# Predict & evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

print("\nClassification report (zero_division=0):")
print(classification_report(y_test, y_pred, zero_division=0))

# Save model + metadata
out_model = "rf_health_model.joblib"
joblib.dump({
    "pipeline": clf,
    "top_labels": top,
    "feature_columns": use_cols
}, out_model)
print("\nSaved model to", out_model)

# Save label mapping (for inference reference)
label_map = pd.DataFrame(y.unique(), columns=["label"])
label_map.to_csv("disease_labels_reduced.csv", index=False)

# --- Quick inference example using first row of dataset ---
sample = X.iloc[[0]]  # take first sample
pred = clf.predict(sample)[0]
print("\nSample input (first row) predicts:", pred)

# Show hydration/temperature advice mapping if present in df
if "Hydration_Advice" in df.columns and "Temperature_Advice" in df.columns:
    ex = df[df["Disease_reduced"] == pred].head(1)
    if not ex.empty:
        print("\nExample advice for predicted class:")
        print("Hydration:", ex.iloc[0]["Hydration_Advice"])
        print("Temperature advice:", ex.iloc[0]["Temperature_Advice"])
    else:
        print("\nNo example row found for predicted class to show advice.")
