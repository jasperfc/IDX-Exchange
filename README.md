# 🏠 IDX Exchange - California Home Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Week%205-success)
![License](https://img.shields.io/badge/Project-Internship-lightgrey)

Machine learning project for predicting California residential property sale prices using **CRMLS real estate transaction data**.

---

# 📖 Project Overview

This project is part of the **IDX Exchange Data Science Internship**.

The objective is to build an end-to-end machine learning pipeline that predicts residential property sale prices based on property characteristics and transaction information from the California Regional Multiple Listing Service (CRMLS).

Current work includes:

- Exploratory Data Analysis (EDA)
- Leakage-aware data preprocessing
- Chronological train/test splitting
- Baseline Linear Regression modeling
- Decision Tree regression modeling
- Random Forest regression modeling
- Hyperparameter exploration for Decision Trees
- Model evaluation and performance benchmarking

## Machine Learning Pipeline

The project follows a chronological, leakage-aware machine learning workflow.

All learned preprocessing operations—including missing-value imputation, outlier threshold estimation, categorical encoding, and feature scaling—are fitted exclusively on the training data and then applied unchanged to the held-out testing period.

The training-window length is configurable through the `TRAIN_MONTHS` parameter, allowing different amounts of historical transaction data to be evaluated while preserving a realistic future-period testing scenario.

---

# 📂 Repository Structure

```text
project/
│
├── data/
│   ├── raw/                  # Raw source data (excluded from GitHub)
│   ├── cleaned/              # Sample processed datasets
│   └── cleaned_full/         # Full processed datasets (local only)
│
├── notebooks/
│   ├── notebook_01_data_exploration.ipynb
│   ├── notebook_02_data_preprocessing.ipynb
│   ├── notebook_03_baseline_model.ipynb
│   └── notebook_04_model_comparison.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🔄 Project Workflow

```text
Raw CRMLS Dataset
        │
        ▼
Residential Property Filtering
        │
        ▼
Data Cleaning
        │
        ▼
Data Type Standardization
        │
        ▼
Chronological Train/Test Split
        │
        ▼
Training-Based Outlier Filtering
        │
        ▼
Missing Value Imputation
        │
        ▼
Categorical Encoding
        │
        ▼
Feature Scaling
        │
        ▼
Linear Regression Baseline
        │
        ▼
Decision Tree Regression
        │
        ▼
Random Forest Regression
        │
        ▼
Model Comparison
```

---

# 📈 Project Progress

| Week | Task | Status |
|------|------|:------:|
| Week 1 | Orientation & Setup | ✅ |
| Week 2 | Exploratory Data Analysis | ✅ |
| Week 3 | Data Preprocessing | ✅ |
| Week 4 | Linear Regression Baseline | ✅ |
| Week 5 | Decision Tree and Random Forest Regressors | ✅ |
| Week 6 | Feature Engineering | ⏳ |
| Week 6+ | Advanced Models & Optimization | ⏳ |

---

# 📓 Notebook Overview

## 01 - Data Exploration

Performed exploratory data analysis on the CRMLS dataset.

### Tasks Completed

- Imported and explored the dataset
- Examined data types
- Reviewed missing values
- Generated descriptive statistics
- Visualized feature distributions
- Investigated relationships between housing characteristics and sale price

---

## 02 - Data Preprocessing

Prepared the dataset for machine learning.

### Tasks Completed

- Filtered residential single-family transactions
- Standardized data types
- Removed duplicate and logically invalid records
- Applied chronological train/test splitting
- Configured a flexible historical training window
- Performed training-based outlier filtering
- Excluded target leakage and non-modeling features
- Imputed missing values using training-set statistics
- Applied categorical and multi-value encoding
- Standardized continuous numerical features
- Exported cleaned and encoded datasets

---

## 03 - Baseline Model

Established the first baseline model for close price prediction.

### Tasks Completed

- Loaded the processed training and testing datasets
- Prepared the feature matrix and prediction target
- Trained a baseline Linear Regression model
- Generated predictions on the held-out test set
- Evaluated model performance using R², MAE, MAPE, MdAPE, and RMSE
- Visualized prediction performance
- Established baseline results for future model comparison

---

## 04 - Model Comparison (Decision Tree & Random Forest)

Developed and evaluated tree-based machine learning models for residential property price prediction.

### Tasks Completed

- Trained a Decision Tree Regressor
- Evaluated multiple maximum tree depths
- Selected the best-performing Decision Tree configuration
- Visualized the Decision Tree structure
- Analyzed Decision Tree feature importance
- Trained a Random Forest Regressor
- Analyzed Random Forest feature importance
- Compared model performance against the Linear Regression baseline
- Evaluated models using R², MAE, MAPE, MdAPE, and RMSE

---

# 📊 Current Model Performance

The table below summarizes the predictive performance of the regression models evaluated on the held-out testing dataset.

| Model | R² | MAE | MAPE | MdAPE | RMSE |
|------|------:|------:|------:|------:|------:|
| Linear Regression | 0.6405 | $368,707.93 | 33.8549% | 25.1820% | $586,081.35 |
| Decision Tree Regressor | 0.8130 | $208,460.26 | 15.1352% | **9.9675%** | $422,718.61 |
| Random Forest Regressor | **0.8609** | **$192,948.57** | **14.8471%** | 10.1211% | **$364,520.98** |

# 🎯 Prediction Target

The prediction target is:

```text
ClosePrice
```

The following pricing variables will be excluded during model training to prevent target leakage:

- `ListPrice`
- `OriginalListPrice`

---

# 💾 Processed Datasets

The preprocessing notebook exports **two versions** of the processed datasets.

## Sample Datasets

Location:

```text
data/cleaned/
```

These lightweight datasets are included in the repository for demonstration purposes.

Generated files (example for a 30-month training window):

- cleaned_crmls_30m.csv
- train_clean_30m.csv
- test_clean_30m.csv
- train_encoded_30m.csv
- test_encoded_30m.csv

The training-window length is configurable through the `TRAIN_MONTHS` parameter in the preprocessing notebook.

---

## Full Datasets

Location:

```text
data/cleaned_full/
```

These datasets contain the complete processed CRMLS records.

The files are generated locally and excluded from GitHub because they exceed GitHub's file size limit.

These datasets will be used for all model training and evaluation in subsequent notebooks.

---

# 🛠️ Technologies

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

---

# ⚙️ Installation

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# 📌 Repository Notes

- Raw CRMLS source data is excluded from GitHub.
- Metadata PDF files are excluded.
- Sample processed datasets are included for demonstration.
- Full processed datasets are generated locally.
- Running the preprocessing notebook will regenerate the complete processed datasets.

---

# 🚀 Future Work

- ✅ Data Exploration
- ✅ Data Preprocessing
- ✅ Linear Regression
- ✅ Model Evaluation (RMSE, MAE, MAPE, R²)
- ✅ Decision Tree Regression
- ✅ Random Forest Regression
- ⏳ Feature Engineering
- ⏳ Training-window comparison
- ⏳ XGBoost
- ⏳ Hyperparameter Tuning
- ⏳ Model Comparison

---

# 👤 Author

**Jasper Fan-Chiang**

M.S. in Applied Data Science  
University of Southern California

IDX Exchange — Data Science Internship