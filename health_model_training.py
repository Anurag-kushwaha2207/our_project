# Dummy Health Disease Prediction Model using Provided CSV
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Data Load
df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

# Step 2: Label Encoding (categorical -> numeric)
lbl_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender',
            'Blood Pressure', 'Cholesterol Level']

for col in lbl_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Disease/Outcome encode
df['Outcome Variable'] = LabelEncoder().fit_transform(df['Outcome Variable'].astype(str))
df['Disease'] = LabelEncoder().fit_transform(df['Disease'].astype(str))

# Step 3: Features & Target
feature_cols = [
    'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender',
    'Blood Pressure', 'Cholesterol Level'
]
X = df[feature_cols]
y = df['Disease']

# Step 4: Train-Test Split & Model Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Step 5: Accuracy & Report
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 6: Example Prediction
dummy_input = pd.DataFrame([{
    'Fever': 1, 'Cough': 1, 'Fatigue': 0, 'Difficulty Breathing': 1,  # Yes/No encoded as per above
    'Age': 25, 'Gender': 1, 'Blood Pressure': 1, 'Cholesterol Level': 2
}])
disease_pred = clf.predict(dummy_input)
print("Predicted Disease (label):", disease_pred[0])