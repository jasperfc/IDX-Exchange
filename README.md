# IDX-Exchange: Real Estate Price Prediction

## Project Overview

This project is part of the IDX Exchange Data Science Internship. The goal is to develop a machine learning pipeline for predicting residential property sale prices using the CRMLS real estate dataset.

---

## Project Structure

```text
project/
│
├── data/
│   ├── raw/                  # Raw source data (excluded from GitHub)
│   ├── cleaned/              # Sample processed datasets (included in GitHub)
│   └── cleaned_full/         # Full processed datasets (local only)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── ......
│
├── README.md
└── .gitignore
```

---

## Project Workflow

```text
Raw Data
    ↓
Data Filtering
    ↓
Data Type Conversion
    ↓
Missing Value Handling
    ↓
Outlier Removal
    ↓
Train/Test Split
    ↓
One-Hot Encoding
    ↓
Export Processed Datasets
    ↓
Linear Regression (Week 4)
```

---

## Notebook Overview

### Notebook 1 – Data Exploration

* Loaded and explored the CRMLS real estate dataset.
* Reviewed data types, missing values, and data quality.
* Performed exploratory data analysis (EDA) with descriptive statistics and visualizations.
* Examined relationships between housing characteristics and sale prices.

### Notebook 2 – Data Preprocessing

* Filtered residential single-family properties.
* Converted features to appropriate data types.
* Handled missing values and removed outliers.
* Split the dataset into training and testing sets.
* Applied One-Hot Encoding to selected categorical features.
* Exported cleaned datasets for model development.

---

## Processed Datasets

The preprocessing notebook generates two versions of the processed datasets:

### Sample datasets (`data/cleaned/`)

These datasets are included in this repository for demonstration purposes.

* `cleaned_crmls.csv`
* `train_clean.csv`
* `test_clean.csv`
* `train_encoded.csv`
* `test_encoded.csv`

### Full datasets (`data/cleaned_full/`)

The preprocessing notebook also generates the complete processed datasets locally. These files are excluded from GitHub because they exceed GitHub's 100 MB file size limit and are used for model training and evaluation in subsequent notebooks.

---

## Requirements

* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Repository Notes

* Raw source data is excluded from this repository.
* Metadata PDF files are not included.
* Sample processed datasets are provided in `data/cleaned/`.
* Full processed datasets are generated locally in `data/cleaned_full/` and are used for model development.
The notebooks can be executed to regenerate the complete processed datasets locally.

---

## Future Work

- Feature Scaling
- Linear Regression
- Model Evaluation (RMSE, MAE, R²)
- Feature Engineering
- Model Comparison and Optimization
