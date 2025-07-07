# 🚴 Bike Sharing Demand Prediction using Random Forest

This project focuses on predicting the hourly demand for shared bikes using weather, seasonal, and time-based data. It leverages a Random Forest Regressor and includes feature engineering, log transformation, model evaluation, and saving the trained pipeline for deployment.

---

## 📁 Dataset

- **Source:** Kaggle – [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)
- **Files Provided:**
  - `train.csv` – used to train the model (includes target `count`)
  - `test.csv` – test set without target
  - `sampleSubmission.csv` – template for submissions

---

## 🎯 Project Objectives

- Predict hourly bike rental count based on historical features
- Handle skewed distribution of target using log transformation
- Evaluate and deploy a Random Forest-based regression model

---

## 🧰 Tools & Libraries

- Python (Pandas, NumPy)
- Scikit-learn
- Matplotlib, Seaborn
- `joblib` (for saving models)

---

## 🛠️ Feature Engineering

- Extracted datetime features: `hour`, `day`, `month`, `year`, `weekday`
- Dropped `casual` and `registered` to avoid data leakage
- Removed `datetime` after feature extraction
- Applied `log1p()` transformation to `count` to reduce skew

---

## 📊 Model Details

- **Model Used:** `RandomForestRegressor`
- **Hyperparameter Tuning:** `GridSearchCV` (optional)
- **Metrics:**
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score

---

## 🧪 Final Model Evaluation

| Metric | Value |
|--------|--------|
| MAE    | ~0.20 (on log scale) |
| RMSE   | ~0.30 (on log scale) |
| R²     | ~0.95 |

> Target variable (`count`) was transformed using log scale, and predictions were converted back using `np.expm1()`.

---

## 🔁 Sample Prediction Pipeline

A final pipeline was built and saved using `joblib` for production use.

```python
# Load the model
import joblib
model = joblib.load('bike_demand_pipeline.pkl')

# Sample input
sample = {
    'season': 1,
    'holiday': 0,
    'workingday': 1,
    'weather': 1,
    'temp': 12.3,
    'atemp': 14.0,
    'humidity': 55,
    'windspeed': 10.5,
    'hour': 8,
    'day': 5,
    'month': 3,
    'year': 2012,
    'weekday': 2
}

import pandas as pd
import numpy as np

# Predict and convert back from log
df = pd.DataFrame([sample])
log_pred = model.predict(df)
count_pred = int(np.expm1(log_pred)[0])
print("Predicted bike count:", count_pred)
```
📁 Bike-Demand-Prediction
│
├── 📄 train.csv
├── 📄 test.csv
├── 📄 bike_final_model.pkl
├── 📄 BS1.ipynb
├── 📄 README.md



✅ Conclusion
This project demonstrates how to handle time-based regression problems using Random Forest.
It showcases feature extraction, log transformation, model evaluation, and deployment — suitable for real-world applications.

👨‍💻 Author
Hussnain Dogar
Data Science & Machine Learning Enthusiast
