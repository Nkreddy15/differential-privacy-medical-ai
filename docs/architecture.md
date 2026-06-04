# System Architecture

```text
Raw Medical Dataset with PII-like Fields
        ↓
Remove Empty Columns
        ↓
SHA-256 Hashing for Name and SSN
        ↓
DOB to Age Transformation
        ↓
Feature Selection: Income, Heart Rate, Age
        ↓
MinMax Scaling
        ↓
Laplace Noise for Differential Privacy
        ↓
Logistic Regression Comparison
        ↓
Original vs Privacy-Preserved Performance Evaluation
```
