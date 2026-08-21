# 🏠 IDX Exchange - California Home Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Week%2010-success)
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
- Linear Regression, Decision Tree, Random Forest, XGBoost, and LightGBM modeling
- Validation-based hyperparameter selection for tree-based models
- Three-version advanced-model experiments: baseline, tuned, and final test evaluation
- Model evaluation and performance comparison
- Saved final-model artifacts with feature-schema validation
- Evaluation expansion by close-price quintile, residual direction, and APE distribution
- Streamlit application for single-property estimates, encoded batch prediction, and model-performance review

## Machine Learning Pipeline

The project follows a chronological, leakage-aware workflow. The second-most-recent complete month is reserved for validation and the latest complete month is reserved for final testing. All learned preprocessing operations are fitted on the training data and applied unchanged to both held-out periods.

Decision Tree, Random Forest, XGBoost, and LightGBM hyperparameters are selected using the validation month. After model choices are frozen, the selected models are refitted on the combined training and validation data and evaluated once on the untouched test month.

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
│   ├── reference/            # Reference files
│   └── results/              # Model evaluation summaries
│
├── models/                   # Local trained-model artifacts (excluded from GitHub)
│
├── notebook/
│   ├── notebook_01_exploration.ipynb
│   ├── notebook_02_preprocessing.ipynb
│   ├── notebook_03_baseline_model.ipynb
│   ├── notebook_04_model_comparison.ipynb
│   ├── notebook_05_advanced_models.ipynb
│   └── notebook_06_evaluation.ipynb
│
├── app.py                    # Streamlit prediction and model-performance app
├── README.md
├── requirements.txt
└── .gitignore
```

## Streamlit Prediction App

The project includes a single-property estimator, encoded-CSV batch prediction,
and a held-out model comparison view. Install the dependencies and launch it from
the repository root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The single-property form recreates the engineered ratios, interaction features,
categorical indicators, and training-fitted scaling used by the saved models.
Batch prediction expects the already encoded 127-feature schema and provides a
downloadable CSV template in the app.

The app requires all five fitted model files plus `model_features.joblib` under
`models/`, as well as `data/results/metrics_summary.csv`. Because trained model
artifacts are intentionally excluded from GitHub, run Notebooks 3-5 locally or
copy the matching artifacts into `models/` before launching the app on a fresh
clone.

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
        │
        ▼
Saved Final Models & Feature Schema
        │
        ▼
Price-Band, Residual & APE Distribution Analysis
        │
        ▼
Metrics Summary Export
        │
        ▼
Streamlit Prediction & Model-Performance App
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
| Week 7 | Advanced Models (XGBoost and LightGBM), Hyperparameter Tuning, and Final Model Comparison | ✅ |
| Week 8 | Evaluation Expansion, Price-Band Error Analysis, and Metrics Summary | ✅ |
| Week 9 | Optional Simple Prediction App (Streamlit) | ✅ |
| Week 10 | Documentation and Reproducibility Instructions | ✅ |
| Week 11 | Practice Presentation and Slide Draft | ⏳ |
| Week 12 | Final Presentation and Repository Handoff | ⏳ |

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
- Restricted the modeling data to California records (`StateOrProvince == "CA"`)
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

## 05 - Advanced Models (XGBoost and LightGBM)

Developed and evaluated XGBoost and LightGBM regressors using the same chronological, leakage-aware workflow. Each model is organized as a three-version experiment:

- **Version A — Baseline:** evaluates an initial model on the validation set
- **Version B — Tuned:** tests 27 hyperparameter combinations and selects a near-best validation configuration
- **Version C — Final:** refits the selected configuration on training plus validation data and evaluates it once on the untouched test set

### Tasks Completed

- Trained baseline XGBoost and LightGBM models using the training period
- Evaluated 27 parameter combinations for each model on the May 2026 validation month
- Used validation R² as the primary selection metric while also recording MAE, MAPE, MdAPE, RMSE, and fitting time
- Visualized the tuning results with three-panel heatmaps separated by learning rate, with R² displayed in every cell
- Selected XGBoost parameters: `max_depth = 8`, `learning_rate = 0.12`, and `n_estimators = 1200`
- Selected LightGBM parameters: `num_leaves = 127`, `learning_rate = 0.08`, and `n_estimators = 900`
- Refit each selected model using the combined training and validation data
- Evaluated each final model once on the untouched June 2026 test month
- Analyzed model feature importance and visualized actual versus predicted close prices
- Compared the best XGBoost and LightGBM results with the Linear Regression, Decision Tree, and Random Forest benchmarks

The selection rule chooses the fastest candidate within 0.001 validation R² of the best result. XGBoost B achieved validation R² of 0.9026, while LightGBM B achieved 0.9037. On the untouched test month, LightGBM C produced the strongest overall result with R² of 0.9060.

### Advanced-Model Experiment Results

| Version    | Model                         | Evaluation Set | R²     | MAE            | MAPE     | MdAPE    | RMSE           |
|------------|-------------------------------|----------------|-------:|---------------:|---------:|---------:|---------------:|
| XGBoost A  | Baseline XGBoost Regressor    | Validation     | 0.8678 | USD 198,103.15 | 15.7776% | 11.3324% | USD 355,429.73 |
| XGBoost B  | Tuned XGBoost Regressor       | Validation     | 0.9026 | USD 157,473.04 | 11.9034% | 8.0322%  | USD 305,146.10 |
| XGBoost C  | Final XGBoost Regressor       | Test           | 0.9009 | USD 160,000.81 | 12.5889% | 8.3710%  | USD 309,013.08 |
| LightGBM A | Baseline LightGBM Regressor   | Validation     | 0.8527 | USD 211,672.36 | 17.1957% | 12.2529% | USD 375,225.27 |
| LightGBM B | Tuned LightGBM Regressor      | Validation     | 0.9037 | USD 159,114.15 | 12.2673% | 8.3805%  | USD 303,347.21 |
| LightGBM C | Final LightGBM Regressor      | Test           | 0.9060 | USD 156,025.13 | 12.1622% | 8.3320%  | USD 301,065.10 |

---

## 06 - Evaluation Expansion

Expanded the final test-month evaluation beyond top-line model metrics and examined how prediction performance changes across price levels and error distributions.

### Tasks Completed

- Reloaded the five saved final models and verified the saved feature schema before prediction
- Evaluated every model on the same untouched June 2026 testing month
- Reported R², MAE, MAPE, MdAPE, and RMSE in a consolidated model scorecard
- Compared overall model fit, percentage errors, and dollar errors using separate visualizations
- Divided actual ClosePrice into five testing-month quintiles and compared MdAPE by model and price band
- Calculated median residuals by price band to identify systematic overprediction and underprediction
- Compared property-level APE distributions using median, P90, and P95 statistics
- Exported `data/results/metrics_summary.csv`

### Evaluation Findings

- LightGBM produced the strongest overall R², MAE, MAPE, and RMSE
- Random Forest produced the lowest overall MdAPE at 7.9310%
- Random Forest achieved the lowest price-band MdAPE in the three lowest quintiles, XGBoost was marginally strongest in the fourth quintile, and LightGBM was strongest in the highest quintile
- All five models typically underpredicted the highest-price quintile, which ranged from approximately USD 1.65 million to USD 8.20 million in the June 2026 testing month
- LightGBM produced the lowest P90 and P95 APE, indicating the strongest control of upper-tail percentage errors

The price bands are testing-month quintiles rather than fixed business thresholds, so their dollar boundaries will change when the evaluation dataset changes. Results also represent one held-out month; rolling-origin evaluation remains necessary to assess performance stability across multiple market periods.

---

# 📊 Current Model Performance

The table below reports the current final performance on the untouched June 2026 test set. Model configurations were selected using May 2026 validation results; the final models were then refitted on training plus validation data before test evaluation.

| Model                     | R²         | MAE                | MAPE        | MdAPE       | RMSE               |
|---------------------------|-----------:|-------------------:|------------:|------------:|-------------------:|
| Linear Regression         | 0.6409     | USD 358,818.47     | 33.0190%    | 24.9885%    | USD 588,365.25     |
| Decision Tree Regressor   | 0.8134     | USD 210,000.70     | 15.5529%    | 10.1457%    | USD 424,092.20     |
| Random Forest Regressor   | 0.8746     | USD 168,733.01     | 12.4580%    | **7.9310%** | USD 347,627.59     |
| XGBoost Regressor         | 0.9009     | USD 160,000.81     | 12.5889%    | 8.3710%     | USD 309,013.08     |
| **LightGBM Regressor**    | **0.9060** | **USD 156,025.13** | **12.1622%**| 8.3320%     | **USD 301,065.10** |

LightGBM produced the strongest R², MAE, MAPE, and RMSE. Random Forest retained the lowest MdAPE, at 7.9310% compared with LightGBM's 8.3320%. LightGBM is therefore the strongest final model when prioritizing overall fit and large-error reduction, while Random Forest remains slightly better for the median percentage error. The actual-versus-predicted plots also show that both boosted models tend to underestimate some of the highest-priced properties.

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

## Evaluation Results

Location:

```text
data/results/
```

Files:

- `metrics_summary.csv`

This model scorecard contains the final June 2026 test metrics for Linear Regression, Decision Tree, Random Forest, XGBoost, and LightGBM.

---

# 🛠️ Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- joblib
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

# 🔁 Reproducing the Results

Run the notebooks from the `notebook/` directory in numerical order because later notebooks depend on datasets or model artifacts produced by earlier notebooks:

1. `notebook_01_exploration.ipynb` explores and filters the raw CRMLS data.
2. `notebook_02_preprocessing.ipynb` regenerates the processed datasets under `data/cleaned/` and `data/cleaned_full/`.
3. `notebook_03_baseline_model.ipynb` trains and saves the final Linear Regression model and feature schema.
4. `notebook_04_model_comparison.ipynb` trains and saves the final Decision Tree and Random Forest models.
5. `notebook_05_advanced_models.ipynb` trains and saves the final XGBoost and LightGBM models.
6. `notebook_06_evaluation.ipynb` reloads all five saved models, creates the evaluation figures, and exports `data/results/metrics_summary.csv`.

The notebooks use paths relative to the `notebook/` directory. Launch Jupyter from that directory to reproduce the current workflow:

```bash
cd notebook
jupyter notebook
```

Raw CRMLS files are excluded from GitHub, so reproducing the full preprocessing workflow requires access to the source files expected under `data/raw/`. The lightweight processed datasets in `data/cleaned/` can be used to inspect the downstream modeling workflow without publishing the raw source data.

## Launching the Streamlit App

After the model artifacts have been created by Notebooks 3-5, return to the
repository root and run:

```bash
streamlit run app.py
```

The app provides three views:

- **Single Property:** converts raw form inputs into the saved 127-feature schema and compares estimates from all five models.
- **Batch Prediction:** validates an already encoded 127-feature CSV and returns predictions from all five models.
- **Model Performance:** displays the held-out June 2026 metrics exported by Notebook 6.

The displayed MAPE-based price range is a descriptive reference only; it is not
a statistical confidence interval or a professional appraisal.

---

# 📌 Repository Notes

- Raw CRMLS source data is excluded from GitHub.
- Sample processed datasets are included for demonstration.
- School district boundary shapefiles used for geographic feature engineering are stored in `data/reference/school_districts/`.
- Individual school-district identity codes are retained in cleaned property-level data but are not used by the current models; they remain available for future experiments.
- The current models use only the three lower-dimensional district-system indicators: elementary, high, and unified.
- Running the preprocessing notebook regenerates the processed datasets.
- Notebooks 3-5 save the final fitted models and feature schema under `models/` for reuse in evaluation and deployment; these local joblib artifacts are excluded from GitHub.
- Notebook 6 reloads the saved models and exports the required Week 8 scorecard to `data/results/metrics_summary.csv`.
- `app.py` reloads those local artifacts for single-property and batch prediction without retraining the models.

---

# 🚀 Remaining Internship Milestones

- ✅ **Week 9:** Complete the optional Streamlit application with single-property estimation, encoded batch prediction, and model-performance views.
- ✅ **Week 10:** Complete the README documentation covering the dataset source, preprocessing, models, results, setup, notebook rerun sequence, and Streamlit launch instructions.
- ⏳ **Week 11:** Prepare a slide draft covering the data, methods, models, evaluation findings, and an optional Streamlit demonstration, then rehearse the presentation.
- ⏳ **Week 12:** Deliver the final presentation and complete the repository handoff with the finalized documentation, slides, and any implemented app or demo materials.

Additional model tuning, rolling-origin validation, geographic error analysis, neural-network experiments, and controlled feature-set comparisons are possible technical extensions, but they are not part of the remaining required internship milestones in the current project schedule.

The technical implementation and core documentation are complete. The remaining
scheduled work is presentation preparation, rehearsal, final delivery, and
repository handoff.

---

# 👤 Author

**Jasper Fan-Chiang**

M.S. in Applied Data Science  
University of Southern California

IDX Exchange — Data Science Internship
