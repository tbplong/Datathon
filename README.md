# Datathon Forecasting Repo

This repository contains the feature engineering and XGBoost training code used to generate the final Revenue and COGS submission.

## Repository Layout

- `data/` - input feature tables and generated outputs.
  - `revenue_features_full.csv` - main training table used by the final pipeline.
  - `revenue_features_eng.csv` - engineered training table written by `src/train.py`.
  - `submission.csv` - final forecast file written by the training script.
  - `feature_importance_revenue.csv` and `feature_importance_cogs.csv` - XGBoost feature importance exports.
  - `shap_importance_revenue.csv` and `shap_importance_cogs.csv` - SHAP-based feature importance exports.
  - `feature_ranking_revenue.csv` and `feature_ranking_cogs.csv` - combined ranking tables.
  - `shap_summary_revenue.png`, `shap_bar_revenue.png` - SHAP visualization plots.
  - `shap_summary_cogs.png`, `shap_bar_cogs.png` - SHAP visualization plots.
- `src/` - model training and analysis scripts.
  - `train.py` - canonical reproducible final submission pipeline with Optuna tuning, SHAP analysis, and time-series CV.
  - `diagnose.py` - data sanity checks and correlation inspection.
  - `baseline.ipynb` - exploratory notebook.
- `docs/` - project notes and supporting material.
- `requirements.txt` - Python dependencies for the project.

## Reproducibility Notes

The final script is designed to be reproducible:

- fixed random seed (`42`)
- deterministic XGBoost settings (`hist` tree method, `n_jobs=1`)
- Optuna hyperparameter search with fixed seed for consistent results
- proper time-series cross-validation splitting (respects temporal order)
- SHAP analysis for model explainability
- fixed input file and output paths under `data/`

## How To Reproduce The Final Submission

1. Create and activate a Python environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the final training script from the repository root:

```bash
python src/train_v3.py
```

4. Review the generated files in `data/`:
   - `revenue_features_eng.csv` - engineered dataset
   - `feature_importance_revenue.csv` - XGBoost importance scores
   - `feature_importance_cogs.csv` - XGBoost importance scores
   - `shap_importance_revenue.csv` - SHAP-based importance
   - `shap_importance_cogs.csv` - SHAP-based importance
   - `feature_ranking_revenue.csv` - combined importance ranking
   - `feature_ranking_cogs.csv` - combined importance ranking
   - `shap_summary_revenue.png`, `shap_bar_revenue.png` - SHAP plots
   - `shap_summary_cogs.png`, `shap_bar_cogs.png` - SHAP plots
   - `submission.csv` - final forecast predictions

## What `train.py` Does

`src/train.py` performs the following steps in a clear, reproducible pipeline:

1. **Load data**: Loads `data/revenue_features_full.csv`.
2. **Feature engineering**: Adds calendar, lag, rolling, momentum, and ratio features.
3. **Feature selection**: Drops weak and redundant features using fixed rules (correlation threshold, manual pruning).
4. **Hyperparameter tuning**: Uses Optuna to find optimal XGBoost hyperparameters via 3-fold time-series CV (30 trials).
5. **Time-series cross-validation**: Evaluates the tuned model with 5-fold time-series CV to assess performance.
6. **Final model training**: Trains Revenue and COGS models on the full dataset using the tuned hyperparameters.
7. **Model explainability**:
   - Computes XGBoost feature importance (weight, gain, cover).
   - Computes SHAP values and generates summary plots (beeswarm and bar charts).
   - Creates a combined ranking table comparing importance methods.
8. **Prediction**: Builds test features for 2023-2024 and generates the submission file.

## Model Explainability

The pipeline provides comprehensive model explainability:

- **Feature Importance**: XGBoost gain-based importance shows which features contribute most to splits.
- **SHAP Values**: Explains individual predictions and shows the average impact of each feature on model output.
- **Visualizations**: 
  - SHAP beeswarm plots show the distribution of feature impacts.
  - SHAP bar plots show average absolute impact magnitude.
  - Combined ranking compares XGBoost gain with SHAP mean absolute values.

## Notes

- The script assumes the raw feature table in `data/revenue_features_full.csv` is present.
- If you change the input data, rerun `src/train.py` to regenerate the engineered table, importance scores, and submission.
- The Optuna search uses a seeded RNG for reproducibility across runs.
- SHAP computation can be memory-intensive for large datasets; plots are saved as PNG files for easy review.