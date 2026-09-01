# 🏠 IDX Exchange - California Home Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Week%2011-success)

Machine learning project for predicting California residential property sale prices using CRMLS transaction data.

---

# 📖 Project Overview

This project was developed as part of the **IDX Exchange Data Science Internship**.

The objective is to create an end-to-end machine learning workflow for estimating residential property sale prices. The project covers:

- Exploratory data analysis
- Leakage-aware preprocessing
- Property and geographic feature engineering
- Chronological model validation
- Comparison of five regression models
- Price-band and residual analysis
- A Streamlit prediction application

The current experiment uses:

- **Training:** November 2023 through April 2026
- **Validation:** May 2026
- **Testing:** June 2026

Model configurations were selected using the validation month. Final models were refitted on training plus validation data and evaluated once on the untouched testing month.

---

# 📊 Key Results

Five regression models were evaluated on the same June 2026 testing set.

| Model | R² | MAE | MAPE | MdAPE | RMSE |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 0.6409 | $358,818 | 33.0190% | 24.9885% | $588,365 |
| Decision Tree | 0.8134 | $210,001 | 15.5529% | 10.1457% | $424,092 |
| Random Forest | 0.8746 | $168,733 | 12.4580% | **7.9310%** | $347,628 |
| XGBoost | 0.9009 | $160,001 | 12.5889% | 8.3710% | $309,013 |
| **LightGBM** | **0.9060** | **$156,025** | **12.1622%** | 8.3320% | **$301,065** |

LightGBM achieved the strongest overall R², MAE, MAPE, and RMSE. Random Forest produced the lowest MdAPE, indicating slightly better performance for the typical property-level percentage error.

Additional evaluation showed that:

- Random Forest performed best in the three lowest price quintiles.
- XGBoost was marginally strongest in the fourth quintile.
- LightGBM performed best in the highest-price quintile.
- All five models generally underpredicted properties in the highest-price quintile.
- LightGBM produced the lowest P90 and P95 absolute percentage errors.

---

# 🧠 Preprocessing and Feature Engineering

The preprocessing workflow includes:

- California residential-property filtering
- Data-type standardization and duplicate removal
- Chronological train, validation, and test splitting
- Training-derived target outlier filtering
- Leakage-feature removal
- Training-fitted missing-value imputation
- Categorical and multi-value encoding
- Training-fitted numerical scaling

Engineered features include:

- Property age and age groups
- Bedroom, bathroom, living-area, lot-size, and garage ratios
- Amenity counts and property-feature interactions
- Flooring and property-level indicators
- California school-district system indicators

The final encoded model schema contains **127 predictors**. Learned preprocessing parameters were fitted only on the training data and applied unchanged to validation and testing data.

---

# 📂 Repository Structure

```text
project/
│
├── app.py
│
├── data/
│   ├── raw/                  # Raw CRMLS files, excluded from GitHub
│   ├── cleaned/              # Sample processed datasets
│   ├── cleaned_full/         # Full processed datasets, excluded from GitHub
│   ├── reference/            # School-district reference files
│   └── results/
│       └── metrics_summary.csv
│
├── models/                   # Local joblib artifacts, excluded from GitHub
│
├── notebook/
│   ├── notebook_01_exploration.ipynb
│   ├── notebook_02_preprocessing.ipynb
│   ├── notebook_03_baseline_model.ipynb
│   ├── notebook_04_model_comparison.ipynb
│   ├── notebook_05_advanced_models.ipynb
│   └── notebook_06_evaluation.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 📓 Notebook Workflow

| Notebook | Purpose |
|---|---|
| 01 - Exploration | Explore CRMLS records, distributions, missing values, and relationships with ClosePrice |
| 02 - Preprocessing | Clean data, engineer features, create chronological splits, encode features, and export datasets |
| 03 - Baseline Model | Train and save the final Linear Regression model |
| 04 - Model Comparison | Select, evaluate, and save Decision Tree and Random Forest models |
| 05 - Advanced Models | Tune, evaluate, and save XGBoost and LightGBM models |
| 06 - Evaluation | Compare all models, analyze price-band errors and residuals, and export the metrics summary |

Notebooks 3–5 save the fitted models and feature schema under `models/`. Notebook 6 reloads these artifacts without retraining them.

---

# 🖥️ Streamlit Application

The Streamlit application provides three views:

- **Single Property:** transforms form inputs into the 127-feature model schema and returns estimates from all five models.
- **Batch Prediction:** accepts an already encoded 127-feature CSV and exports model predictions.
- **Model Performance:** displays the held-out June 2026 evaluation metrics.

LightGBM is used as the primary estimate because it achieved the strongest overall testing performance.

The displayed MAPE-based price range is a descriptive reference. It is not a statistical confidence interval or professional appraisal.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/jasperfc/IDX-Exchange.git
cd IDX-Exchange
```

Create and activate a Python 3.11 virtual environment:

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 🔁 Reproducing the Results

Run the notebooks from the `notebook/` directory in numerical order:

```bash
cd notebook
jupyter notebook
```

The workflow order is:

1. Exploration
2. Preprocessing and dataset export
3. Linear Regression training
4. Decision Tree and Random Forest training
5. XGBoost and LightGBM training
6. Final evaluation and metrics export

Raw CRMLS files are excluded from GitHub. Full preprocessing therefore requires access to the source files expected under `data/raw/`.

The sample datasets under `data/cleaned/` are included for inspecting the downstream modeling workflow.

---

# 🚀 Launching the App

The application requires these local artifacts:

```text
models/
├── linear_regression.joblib
├── decision_tree.joblib
├── random_forest.joblib
├── xgboost.joblib
├── lightgbm.joblib
└── model_features.joblib
```

These files can be generated by running Notebooks 3–5. They are excluded from GitHub because of their file sizes.

Launch the application from the repository root:

```bash
streamlit run app.py
```

The app also requires:

```text
data/results/metrics_summary.csv
```

---

# ⚠️ Limitations

- Results represent one held-out testing month.
- Price-band boundaries are testing-month quintiles rather than fixed business thresholds.
- High-priced properties remain more difficult to predict.
- The single-property App manually recreates the current preprocessing schema and must be updated if model features or scaling parameters change.
- Predictions are educational estimates and should not be treated as professional appraisals, lending advice, or investment advice.

---

# 📅 Project Progress

| Week | Deliverable | Status |
|---|---|:---:|
| Weeks 1–7 | Exploration, preprocessing, feature engineering, and modeling | ✅ |
| Week 8 | Expanded evaluation and `metrics_summary.csv` | ✅ |
| Week 9 | Optional Streamlit application | ✅ |
| Week 10 | Documentation and reproducibility instructions | ✅ |
| Week 11 | Presentation draft and rehearsal | ✅ |
| Week 12 | Final presentation and repository handoff | ⏳ |

The remaining scheduled work is presentation preparation, rehearsal, final delivery, and repository handoff.

---

# 🛠️ Technologies

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, joblib, GeoPandas, Shapely, matplotlib, seaborn, Jupyter Notebook, and Streamlit.

---

# 👤 Author

**Jasper Fan-Chiang**

M.S. in Applied Data Science  
University of Southern California

IDX Exchange — Data Science Internship
