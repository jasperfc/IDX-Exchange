# 🏠 IDX Exchange - California Home Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Week%203-yellow)
![License](https://img.shields.io/badge/Project-Internship-lightgrey)

Machine learning project for predicting California residential property sale prices using **CRMLS real estate transaction data**.

---

# 📖 Project Overview

This project is part of the **IDX Exchange Data Science Internship**.

The objective is to build an end-to-end machine learning pipeline that predicts residential property sale prices based on property characteristics and transaction information from the California Regional Multiple Listing Service (CRMLS).

Current work focuses on:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature preparation
- Building a baseline prediction model

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
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_linear_regression.ipynb
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
Data Type Conversion
        │
        ▼
Missing Value Handling
        │
        ▼
Outlier Removal
        │
        ▼
Train/Test Split
        │
        ▼
One-Hot Encoding
        │
        ▼
Feature Selection
        │
        ▼
Model Development
        │
        ▼
Model Evaluation
```

---

# 📈 Project Progress

| Week | Task | Status |
|------|------|:------:|
| Week 2 | Exploratory Data Analysis | ✅ |
| Week 3 | Data Preprocessing | ✅ |
| Week 4 | Linear Regression Baseline | ⏳ |
| Week 5 | Model Evaluation & Improvement | ⏳ |
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

- Filtered residential single-family properties
- Converted columns to appropriate data types
- Handled missing values
- Removed unrealistic observations and outliers
- Split the dataset into training and testing sets
- Applied One-Hot Encoding to categorical variables
- Exported processed datasets

---

## 03 - Linear Regression *(Week 4)*

Planned tasks include:

- Feature selection
- Baseline Linear Regression
- Model evaluation
- Residual analysis
- Performance comparison

---

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

Generated files:

- cleaned_crmls.csv
- train_clean.csv
- test_clean.csv
- train_encoded.csv
- test_encoded.csv

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
- ⏳ Feature Scaling
- ⏳ Linear Regression
- ⏳ Model Evaluation (RMSE, MAE, MAPE, R²)
- ⏳ Feature Engineering
- ⏳ Decision Tree
- ⏳ Random Forest
- ⏳ XGBoost
- ⏳ Hyperparameter Tuning
- ⏳ Model Comparison

---

# 👤 Author

**Jasper Fan-Chiang**

M.S. in Applied Data Science  
University of Southern California

IDX Exchange — Data Science Internship