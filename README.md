# 🏠 IDX Exchange - California Home Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Week%206-success)
![License](https://img.shields.io/badge/Project-Internship-lightgrey)

Machine learning project for predicting California residential property sale prices using **CRMLS real estate transaction data**.

---

# 📖 Project Overview

This project is part of the **IDX Exchange Data Science Internship**.

The objective is to build an end-to-end machine learning pipeline that predicts residential property sale prices based on property characteristics and transaction information from the California Regional Multiple Listing Service (CRMLS).

Current work includes:

- Exploratory Data Analysis
- Leakage-aware data preprocessing
- Chronological train/test splitting
- Property and geographic feature engineering
- Linear Regression, Decision Tree, and Random Forest modeling
- Decision Tree depth exploration
- Model evaluation and performance comparison

## Machine Learning Pipeline

The project follows a chronological, leakage-aware workflow. All learned preprocessing operations are fitted on the training data and applied unchanged to the held-out testing period.

The training window is configurable through `TRAIN_MONTHS`, and `ClosePrice` outlier thresholds are calculated from the training set only.

---

# 📂 Repository Structure

```text
project/
│
├── data/
│   ├── raw/                  # Raw source data (excluded from GitHub)
│   ├── cleaned/              # Sample processed datasets
│   ├── cleaned_full/         # Full processed datasets
│   └── reference/            # Reference files
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
Property & Temporal Feature Engineering
        │
        ▼
Geographic Feature Engineering
        │
        ▼
Categorical & Multi-Value Encoding
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
| Week 6 | Property & Geographic Feature Engineering | ✅ |
| Week 7+ | Advanced Models & Optimization | ⏳ |

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

- Filtered and cleaned residential transaction records
- Standardized data types and removed invalid records
- Applied chronological train/test splitting
- Used training-derived thresholds for target outlier filtering
- Removed leakage and non-modeling features
- Imputed missing values using training-set statistics
- Performed property and geographic feature engineering
- Encoded categorical and multi-value features
- Scaled selected numerical features
- Exported cleaned and encoded datasets

### Feature Engineering

- Extracted year and month features from `CloseDate`
- Created `PropertyAge`, `BathroomsPerBedroom`, `LivingAreaPerBedroom`, and `LivingAreaLotRatio`
- Split and encoded multi-value `Flooring` and `Levels` fields
- Spatially joined properties with California school district boundaries
- Created elementary, high, and unified district features
- Combined multiple district matches into one property row

---

## 03 - Baseline Model

Developed a Linear Regression baseline for close-price prediction.

### Tasks Completed

- Loaded the processed training and testing datasets
- Prepared the feature matrix and prediction target
- Trained a baseline Linear Regression model
- Generated predictions on the held-out test set
- Evaluated model performance using R², MAE, MAPE, MdAPE, and RMSE
- Visualized prediction performance

---

## 04 - Model Comparison (Decision Tree & Random Forest)

Developed and evaluated tree-based machine learning models for residential property price prediction.

### Tasks Completed

- Trained Decision Tree and Random Forest regressors
- Evaluated multiple Decision Tree depths
- Selected the strongest tested tree configuration
- Visualized the Decision Tree structure
- Analyzed model feature importance
- Regenerated the Linear Regression baseline using the updated pipeline
- Compared all models using R², MAE, MAPE, MdAPE, and RMSE
- Visualized model performance and predictions

---

# 📊 Current Model Performance

The table below summarizes the predictive performance of the regression models evaluated on the held-out testing dataset.

| Model                     | R²     | MAE            | MAPE    | MdAPE   | RMSE           |
|---------------------------|-------:|---------------:|---------:|---------:|---------------:|
| Linear Regression         | 0.6415 | USD 358,586.84 | 32.9718% | 24.8969% | USD 587,883.67 |
| Decision Tree Regressor   | 0.8198 | USD 208,838.71 | 15.4044% | 10.2098% | USD 416,794.21 |
| Random Forest Regressor   | **0.8602** | **USD 191,428.79** | **14.9573%** | **10.1449%** | **USD 367,103.45** |

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

The preprocessing notebook exports both sample and full processed datasets.

## Sample Datasets

Location:

```text
data/cleaned/
```

Files:

- `train_encoded_30m.csv`
- `test_encoded_30m.csv`

These lightweight datasets are included in the repository for demonstration purposes.

---

## Full Datasets

Location:

```text
data/cleaned_full/
```

Files:

- `cleaned_crmls_30m.csv`
- `train_encoded_30m.csv`
- `test_encoded_30m.csv`

The training window is configurable through the `TRAIN_MONTHS` parameter.

---

# 🛠️ Technologies

- Python
- pandas
- NumPy
- scikit-learn
- GeoPandas
- Shapely
- matplotlib
- seaborn
- Jupyter Notebook

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/jasperfc/IDX-Exchange.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📌 Repository Notes

- Raw CRMLS source data is excluded from GitHub.
- Sample processed datasets are included for demonstration.
- School district boundary shapefiles used for geographic feature engineering are stored in `data/reference/school_districts/`.
- Running the preprocessing notebook regenerates the processed datasets.

---

# 🚀 Future Work

- ⏳ XGBoost
- ⏳ Hyperparameter Tuning
- ⏳ Training-window comparison

---

# 👤 Author

**Jasper Fan-Chiang**

M.S. in Applied Data Science  
University of Southern California

IDX Exchange — Data Science Internship