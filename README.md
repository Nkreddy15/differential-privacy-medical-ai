# Differential Privacy for Medical AI Systems

This project demonstrates a privacy-preserving machine learning workflow for medical tumor classification. It protects personally identifiable information through hashing and applies differential privacy using Laplace noise before comparing model performance on original and privacy-preserved data.

## Features

- SHA-256 anonymization for PII-like fields such as name and SSN
- DOB-to-age transformation
- Medical tumor classification using Logistic Regression
- Laplace-noise differential privacy
- Original vs privacy-preserved model comparison
- Legal and ethical discussion covering GDPR, HIPAA, and DPDP concepts

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Hashlib

## Reported Results

From the project report:

- Original Data Accuracy: 75.0%
- Privacy-Preserved Data Accuracy: 66.67%
- F1-score on class 0 decreased from 0.86 to 0.80 after privacy preservation
- Class 1 performance remained weak due to class imbalance

## Repository Structure

```text
.
├── data/
│   ├── processed/
│   └── local_only_raw_do_not_upload/
├── src/
├── reports/
├── screenshots/
├── docs/
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

Place the raw dataset locally at:

```text
data/local_only_raw_do_not_upload/assignment2_medical_dataset_raw.csv
```

Then run:

```bash
pip install -r requirements.txt
python src/privacy_preserving_medical_ai.py
```

## Privacy Warning

Do not publicly upload raw datasets containing real or PII-like information. The raw dataset is placed in a local-only folder and ignored by `.gitignore`.

## Author

Nikhil Kumar Reddy Chalamalla
