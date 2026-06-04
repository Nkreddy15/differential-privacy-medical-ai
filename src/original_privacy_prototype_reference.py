
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import hashlib

# Load dataset
df = pd.read_csv("Assignment2Dataset-1.csv")

# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(how='all', axis=1)

# Hash PII fields
def hash_value(val):
    if pd.isna(val):
        return None
    return hashlib.sha256(val.encode()).hexdigest()

df['Name'] = df['Name'].apply(hash_value)
df['SSN'] = df['SSN'].apply(hash_value)

# Convert DOB to Age
def calculate_age(dob_str):
    if pd.isna(dob_str):
        return None
    try:
        dob = datetime.strptime(dob_str, "%m/%d/%Y")
    except:
        try:
            dob = datetime.strptime(dob_str, "%m/%d/%y")
        except:
            return None
    today = datetime.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

df['Age'] = df['DOB'].apply(calculate_age)
df.drop(columns=['DOB'], inplace=True)

# Convert Oxygen Level to numeric
df['Oxygen Level'] = pd.to_numeric(df['Oxygen Level'], errors='coerce')

# Select features and target
features = ['Income', 'Heart Rate', 'Age']
target = 'Tumor Condition'
df_model = df[features + [target]].dropna()
df_model[target] = df_model[target].map({'Normal': 0, 'Abnormal': 1})

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(df_model[features])
y = df_model[target].values

# Apply differential privacy (Laplace noise)
def add_laplace_noise(data, epsilon=1.0):
    sensitivity = 1.0
    noise = np.random.laplace(0, sensitivity / epsilon, data.shape)
    return data + noise

X_private = add_laplace_noise(X, epsilon=1.0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_private_train, X_private_test, _, _ = train_test_split(X_private, y, test_size=0.2, random_state=42)

# Train on original data
model_original = LogisticRegression()
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)

# Train on private data
model_private = LogisticRegression()
model_private.fit(X_private_train, y_train)
y_pred_private = model_private.predict(X_private_test)

# Evaluation
print("Original Data Accuracy:", accuracy_score(y_test, y_pred_original))
print("Private Data Accuracy:", accuracy_score(y_test, y_pred_private))

print("\nClassification Report (Original):")
print(classification_report(y_test, y_pred_original))

print("\nClassification Report (Private):")
print(classification_report(y_test, y_pred_private))
