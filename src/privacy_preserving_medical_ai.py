"""Differential Privacy for Medical AI Systems

This script anonymizes PII-like fields, converts DOB to age, applies Laplace noise
for differential privacy, and compares Logistic Regression performance on original
vs. privacy-preserved medical data.

Raw input expected locally at:
    data/local_only_raw_do_not_upload/assignment2_medical_dataset_raw.csv

Do not publicly upload raw datasets with real personal information.
"""

from pathlib import Path
from datetime import datetime
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "local_only_raw_do_not_upload" / "assignment2_medical_dataset_raw.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "privacy_preserved_medical_dataset.csv"


def hash_value(value):
    if pd.isna(value):
        return None
    return hashlib.sha256(str(value).encode()).hexdigest()


def calculate_age(dob_str):
    if pd.isna(dob_str):
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            dob = datetime.strptime(str(dob_str), fmt)
            today = datetime.today()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except ValueError:
            continue
    return None


def add_laplace_noise(data, epsilon=1.0):
    sensitivity = 1.0
    noise = np.random.laplace(0, sensitivity / epsilon, data.shape)
    return data + noise


def main():
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(how="all", axis=1)

    df["Name"] = df["Name"].apply(hash_value)
    df["SSN"] = df["SSN"].apply(hash_value)
    df["Age"] = df["DOB"].apply(calculate_age)
    df = df.drop(columns=["DOB"])

    df["Oxygen Level"] = pd.to_numeric(df["Oxygen Level"].astype(str).str.replace("%", "", regex=False), errors="coerce")

    features = ["Income", "Heart Rate", "Age"]
    target = "Tumor Condition"
    df_model = df[features + [target]].dropna().copy()
    df_model[target] = df_model[target].map({"Normal": 0, "Abnormal": 1})

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_model[features])
    y = df_model[target].values

    X_private = add_laplace_noise(X, epsilon=1.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_private_train, X_private_test, _, _ = train_test_split(X_private, y, test_size=0.2, random_state=42)

    model_original = LogisticRegression(max_iter=1000)
    model_original.fit(X_train, y_train)
    y_pred_original = model_original.predict(X_test)

    model_private = LogisticRegression(max_iter=1000)
    model_private.fit(X_private_train, y_train)
    y_pred_private = model_private.predict(X_private_test)

    print("Original Data Accuracy:", accuracy_score(y_test, y_pred_original))
    print("Private Data Accuracy:", accuracy_score(y_test, y_pred_private))
    print("
Classification Report (Original):")
    print(classification_report(y_test, y_pred_original, zero_division=0))
    print("
Classification Report (Private):")
    print(classification_report(y_test, y_pred_private, zero_division=0))

    processed = pd.DataFrame(X_private, columns=features)
    processed[target] = y
    processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Privacy-preserved dataset saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
