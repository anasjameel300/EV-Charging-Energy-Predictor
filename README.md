# EV Charging Energy Prediction (MLPA-NEW)

## Project overview
Predicts energy consumed (Energy (kWh)) during EV charging sessions using tabular machine-learning models and an LSTM. The pipeline includes preprocessing and feature engineering, correlation analysis, and training/evaluation of multiple models.

## Key dataset details
- Preprocessed dataset: 102,781 rows × 71 columns
- Selected features used for modeling: 33 columns (32 features + 1 target)
- Target (Energy (kWh)) summary:
  - Mean: 8.88 kWh
  - Min: 0.01 kWh
  - Max: 97.36 kWh
  - Std: 7.63 kWh

## Models & Results (from notebook)
- Linear Regression
  - Train RMSE: 1.4282 kWh | Test RMSE: 1.7098 kWh
  - Train MAE: 0.9058 kWh  | Test MAE: 0.9212 kWh
  - Train R²: 0.9648       | Test R²: 0.9507

- Random Forest (n_estimators=100, max_depth=20)
  - Train RMSE: 0.1517 kWh | Test RMSE: 0.2903 kWh
  - Train MAE: 0.0119 kWh  | Test MAE: 0.0251 kWh
  - Train R²: 0.9996       | Test R²: 0.9986
  - Top features: Fee (~68%), Charging Time (~27%), Charging Efficiency (~2%)

- XGBoost (n_estimators=200, max_depth=10, learning_rate=0.1)
  - Test R²: ~0.9986 | Test RMSE: ~0.2854 kWh

- LSTM (deep learning)
  - Test R²: 0.9465 | Test RMSE: 1.78 kWh
  - Notes: LSTM was trained on timesteps=1 (tabular samples); tree-based models fit this problem better.

## Best model
Random Forest and XGBoost produce the best results (Test R² ≈ 0.9986). Random Forest is recommended for production due to its strong balance of metrics and interpretability via feature importance.

## Repository contents
- EV_Charging_Prediction_Models.ipynb — modeling pipeline, training, evaluation, and visualizations
- preprocess_and_engineer_features.py — preprocessing, feature engineering, correlation analysis, and export of Dataset/EVcharging_preprocessed.csv
- correlation_heatmap_full.png — full correlation heatmap
- correlation_with_energy.png — bar chart of correlations with Energy (kWh)
- top_correlations.png — top 30 pairwise correlations
- feature_analysis_summary.md — detailed feature-engineering summary and selection rationale
- correlation_explanation.md — explanation of correlation approach and formulas
- explanation.txt — full project walkthrough, concepts, and Q&A
- .gitignore

## How to reproduce
1. Install dependencies (Python 3.8+). Example packages:
   - pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, tensorflow (2.10), jupyter
2. Preprocess raw CSV:
   - Place raw CSV at `Dataset/EVcharging.csv`
   - Run: `python preprocess_and_engineer_features.py`
   - Outputs: `Dataset/EVcharging_preprocessed.csv`, correlation figures
3. Run the notebook:
   - Open `EV_Charging_Prediction_Models.ipynb` and run cells sequentially. The notebook reads `Dataset/EVcharging_preprocessed.csv`.

## Notes & cautions
- The extremely high R² scores for tree-based models suggest careful validation for data leakage is required (ensure no feature directly encodes the target or deterministic transformations that leak target information).
- LSTM was included for comparison; it is not the best fit for this tabular (non-sequential) dataset.
- Consider time-based validation, cross-validation, and hyperparameter tuning before deployment.
