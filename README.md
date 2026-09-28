# Telecom Customer Churn Prediction

A supervised-learning pipeline that predicts which telecom customers are likely
to churn and surfaces the **drivers** behind it — the kind of retention
analytics I've delivered in the telecom sector.

## Highlights
- Realistic synthetic dataset where churn is driven by contract type, tenure,
  support-call volume and spend.
- `scikit-learn` pipeline: scaling + one-hot encoding + balanced Random Forest.
- Reports precision/recall/F1, **ROC-AUC**, and ranked feature importances.

## Run it
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
```

## Stack
`scikit-learn` · `pandas` · `numpy`

---
Part of my data analytics portfolio — [github.com/syedjawaadali](https://github.com/syedjawaadali)
