# Credit Card Fraud Detection

A Streamlit dashboard demonstrating fraud classification and anomaly detection using Isolation Forest, Local Outlier Factor, and a class-weighted Random Forest classifier.

## Run in VS Code (Windows)

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. It runs in demo mode, but demo data is synthetic. For meaningful evaluation, upload Kaggle's labelled `creditcard.csv` dataset in the sidebar. The expected target column is `Class` (0 = legitimate, 1 = fraud).

## Features
- Dataset summary and preview
- Isolation Forest and Local Outlier Factor anomaly detection
- Random Forest classifier with class weighting and adjustable probability threshold
- Precision, recall, F1-score comparison
- Confusion matrix and ROC curve / ROC-AUC
- Interactive transaction scoring
- CSV export of test predictions

## Important
This is an educational prototype, not a production banking system. The project uses Random Forest, not XGBoost; add XGBoost only if the assignment explicitly requires it. Results on synthetic demo data are not evidence of real-world performance.
