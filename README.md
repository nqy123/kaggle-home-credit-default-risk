# Kaggle Home Credit Default Risk

This repository contains a LightGBM feature-engineering solution for Kaggle's
`home-credit-default-risk` competition.

The project focuses on interpretable relational feature engineering across the
main application table and historical customer behavior tables.

## Current Status

This project has training scripts and raw data locally, but no confirmed Kaggle
submission score has been recorded yet.

## Method

The solution builds risk features from multiple related tables:

- Application-level debt and affordability ratios.
- External source summary statistics.
- Missing-value counts and document-flag counts.
- Bureau credit history aggregates.
- Bureau balance history aggregates.
- Previous application approval/refusal aggregates.
- POS cash balance behavior aggregates.
- Installment payment delay and payment-ratio features.
- Credit-card utilization and payment behavior features.

Modeling uses 5-fold StratifiedKFold LightGBM with ROC-AUC validation.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full relational feature version:

```bash
python src/train_best.py
```

Run the faster GPU-oriented version:

```bash
python src/train_gpu_fast.py
```

Expected generated files include:

- `outputs/oof_lgbm_best.csv`
- `outputs/pred_lgbm_best.csv`
- `outputs/submission_lgbm_best.csv`
- `outputs/feature_importance_top200.csv`
- `outputs/experiment_summary.json`

Raw Kaggle data is not committed.
