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
- Chronological train/validation/test splitting
- Property and geographic feature engineering
- Linear Regression, Decision Tree, and Random Forest modeling
- Validation-based Decision Tree and Random Forest depth selection
- Model evaluation and performance comparison

## Machine Learning Pipeline

The project follows a chronological, leakage-aware workflow. The second-most-recent complete month is reserved for validation and the latest complete month is reserved for final testing. All learned preprocessing operations are fitted on the training data and applied unchanged to both held-out periods.

Decision Tree and Random Forest hyperparameters are selected using the validation month. After model choices are frozen, the selected models are refitted on the combined training and validation data and evaluated once on the untouched test month.

The training window is configurable through `TRAIN_MONTHS`, and `ClosePrice` outlier thresholds are calculated from the training set only.

The current 30-month experiment uses:

- **Training:** November 2023 through April 2026 (30 months)
- **Validation:** May 2026
- **Test:** June 2026

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
├── notebook/
│   ├── notebook_01_exploration.ipynb
│   ├── notebook_02_preprocessing.ipynb
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
Chronological Train/Validation/Test Split
        │
        ▼
Property & Temporal Feature Engineering
        │
        ▼
Geographic Feature Engineering
        │
        ▼
Training-Derived Target Outlier Filtering
        │
        ▼
Leakage & Non-Modeling Feature Exclusion
        │
        ▼
Training-Fitted Missing Value Imputation
        │
        ▼
Categorical & Multi-Value Encoding
        │
        ▼
Training-Fitted Feature Scaling
        │
        ▼
Processed Dataset Export
        │
        ▼
Candidate Models Trained on Training Set
        │
        ▼
Validation-Month Model Selection
        │
        ▼
Selected Models Refit on Training + Validation
        │
        ▼
One-Time Test-Month Evaluation & Model Comparison
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
- Applied chronological train/validation/test splitting
- Used training-derived thresholds for target outlier filtering
- Removed leakage and non-modeling features
- Imputed missing values using training-set statistics
- Performed property and geographic feature engineering
- Encoded categorical and multi-value features
- Scaled selected numerical features
- Exported cleaned and encoded datasets

### Feature Engineering

- Extracted year and month features from `CloseDate`
- Created five property-age predictors: `PropertyAge` and four encoded age ranges
- Created six housing and space-ratio predictors
- Created seven amenity-summary and property-interaction predictors
- Split and encoded multi-value `Flooring` and `Levels` fields
- Spatially joined properties with California school district boundaries
- Retained individual school-district codes in the cleaned property-level datasets for possible future experiments
- Created three lower-dimensional district-type indicators for the current model feature set
- Combined multiple district matches into one property row

The Week 6 feature-engineering update increased the encoded model schema from **106 to 127 predictors**. All 106 original predictors were retained, 21 were added, and none were removed.

---

## 03 - Baseline Model

Developed a Linear Regression baseline for close-price prediction.

### Tasks Completed

- Loaded the processed training, validation, and testing datasets
- Prepared the feature matrix and prediction target
- Refit the baseline Linear Regression model on the combined training and validation data
- Generated predictions on the held-out test set
- Evaluated model performance using R², MAE, MAPE, MdAPE, and RMSE
- Visualized prediction performance

---

## 04 - Model Comparison (Decision Tree & Random Forest)

Developed and evaluated tree-based machine learning models for residential property price prediction.

### Tasks Completed

- Trained candidate Decision Tree and Random Forest regressors using the training set
- Evaluated candidate depths on the validation month
- Applied the near-best validation rule and selected the shallowest qualifying configuration
- Refit each selected tree-based model on the combined training and validation data
- Evaluated the refitted models once on the untouched test month
- Visualized the Decision Tree structure
- Analyzed model feature importance
- Regenerated the Linear Regression baseline using the updated pipeline
- Compared all models using R², MAE, MAPE, MdAPE, and RMSE
- Visualized model performance and predictions

---

# 📊 Current Model Performance

The table below reports the current final performance on the untouched June 2026 test set. Model configurations were selected using May 2026 validation results; the final models were then refitted on training plus validation data before test evaluation.

| Model                     | R²     | MAE            | MAPE    | MdAPE   | RMSE           |
|---------------------------|-------:|---------------:|---------:|---------:|---------------:|
| Linear Regression         | 0.6410 | USD 358,844.63 | 33.0237% | 24.9839% | USD 588,238.00 |
| Decision Tree Regressor   | 0.8131 | USD 210,221.47 | 15.5580% | 10.1449% | USD 424,456.10 |
| Random Forest Regressor   | **0.8749** | **USD 168,931.31** | **12.4763%** | **7.9706%** | **USD 347,200.70** |

Random Forest produced the strongest result across all reported metrics in the current evaluation.

---

# 🧩 Old vs. New Feature Set

The comparison below describes the encoded predictor schemas before and after the Week 6 feature-engineering update. `ClosePrice` is excluded from these counts.

| Feature-set measure | Old feature set | Current feature set |
|---------------------|----------------:|--------------------:|
| Encoded predictors | 106 | 127 |
| Predictors retained from the old set | 106 | 106 |
| Newly added predictors | 0 | 21 |
| Predictors removed from the old set | 0 | 0 |

| Added feature group | Predictors added | Purpose |
|---------------------|-----------------:|---------|
| Property age | 5 | Represent continuous property age and nonlinear age ranges |
| Housing and space ratios | 6 | Capture layout, space utilization, and parking capacity relative to property size |
| Amenity summaries and interactions | 7 | Represent amenity richness and selected property-feature combinations |
| School-district system type | 3 | Add lower-dimensional geographic context without using individual district identity |
| **Total** | **21** | |

This is a schema comparison, not a controlled estimate of performance improvement. The earlier and current results used different preprocessing, validation, tuning, and feature-engineering workflows, so their metric differences cannot be attributed solely to the 21 added predictors. A controlled comparison must hold the split, preprocessing rules, model configuration, and random seed constant.

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
- `validation_encoded_30m.csv`
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
- `validation_encoded_30m.csv`
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
cd IDX-Exchange
```

Create and activate a Python 3.11 virtual environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` currently records the required packages but does not pin exact versions. The Python 3.11 environment should therefore be treated as the supported setup rather than a fully locked environment. Exact version pinning or a lock file remains necessary for byte-for-byte reproducible dependency installation.

---

# 📌 Repository Notes

- Raw CRMLS source data is excluded from GitHub.
- Sample processed datasets are included for demonstration.
- School district boundary shapefiles used for geographic feature engineering are stored in `data/reference/school_districts/`.
- Individual school-district identity codes are retained in cleaned property-level data but are not used by the current models; they remain available for future experiments.
- The current models use only the three lower-dimensional district-system indicators: elementary, high, and unified.
- Running the preprocessing notebook regenerates the processed datasets.

---

# 🚀 Future Work

- ⏳ XGBoost
- ⏳ Broader hyperparameter optimization beyond the current depth search
- ⏳ Rolling-origin validation across multiple historical cutoffs
- ⏳ Controlled 106-versus-127 feature-set performance comparison
- ⏳ Training-window comparison
- ⏳ Exact dependency-version pinning or lock-file generation

---

# 👤 Author

**Jasper Fan-Chiang**

M.S. in Applied Data Science  
University of Southern California

IDX Exchange — Data Science Internship
