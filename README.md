# D100: Loan Default Prediction

This project implements an end-to-end machine learning pipeline to predict loan defaults in [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset/data) from [Kaggle](https://www.kaggle.com/). It demonstrates a rigorous data science workflow, including automated data acquisition, custom feature engineering, model benchmarking (GLM vs. LightGBM), hyperparameter tuning, and in-depth model interpretation using Dalex.

## 📌 Project Overview

The goal of this project is to predict the `Status` variable (Default vs. Non-Default) based on various borrower and loan characteristics.

**Key Features:**

  * **Automated Pipeline:** Data is automatically downloaded from Kaggle using `kagglehub`.
  * **Custom Transformers:** Implementation of a custom scikit-learn compatible `CustomStandardScaler`.
  * **Dual Modeling Approach:** Compares a linear baseline (**Logistic Regression with ElasticNet**) against a non-linear tree-based model (**LightGBM**).
  * **Hyperparameter Tuning:** Utilizes `GridSearchCV` and `RandomizedSearchCV` for optimal model performance.
  * **Model Interpretation:** Uses the `dalex` library to generate ROC curves, Feature Importance, and Partial Dependence Plots (PDP).
  * **Robust Testing:** Includes unit tests for feature engineering components.

## 📂 Project Structure

```text
├── data/
│   ├── raw/                   # Raw CSV files (downloaded automatically)
│   └── processed/             # Cleaned parquet files
├── models/                    # Serialized models (.joblib)
├── notebooks/
│   ├── eda_cleaning.ipynb     # Exploratory Data Analysis & Cleaning
│   └── evaluation.ipynb       # Interactive Model Interpretation
├── reports/
│   └── figures/               # Generated plots (ROC, PDP, etc.)
├── src/
│   ├── data.py                # Data loading and downloading logic
│   ├── feature_engineering.py # Custom transformers (CustomStandardScaler)
│   └── model_training.py      # Training pipelines and hyperparameter tuning
├── tests/
│   └── test_feature_engineering.py # Unit tests for custom transformers
├── environment.yml            # Conda environment definition
├── setup.py                   # Package setup
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### Prerequisites

  * Python 3.12
  * Conda (recommended)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Joy-Quan/d100_loan_default_prediction/
    cd d100_loan_default_prediction
    ```

2.  **Create and activate the environment:**

    ```bash
    conda env create -f environment.yml
    conda activate d100_env
    ```

3.  **Install the package in editable mode:**

    ```bash
    pip install -e .
    ```

## 💻 Usage

### 1. EDA and Cleaning

Run all the cells in [notebooks/eda_cleaning.ipynb](notebooks/eda_cleaning.ipynb) to explore the raw data, perform cleaning, and save the processed data to parquet format. This step is mandatory before training.

Once finished, the processed data will be saved to `data/processed/cleaned_data.parquet`.

### 2. Train Models

Run the training script to tune hyperparameters, train the models, and save the best models:

```bash
python src/model_training.py
```

### 3. Evaluate Models

Run all the cells in [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb) to evaluate the models, generate performance metrics and explainability plots (ROC, Feature Importance, PDP).

The result figures will **not** automatically be saved, but you can view them under [reports/figures/](reports/figures/) since I've manually saved them before.

### 4. Run Tests

Ensure custom components work as expected:

```bash
python -m pytest tests/
```

## 📊 Methodology & Results

### Data Preprocessing

  * **Leakage Removal:** Columns like `Interest_rate_spread` were removed as they are direct indicators of loan approval status (data leakage).
  * **Imputation:** Numerical missing values filled with Median; Categorical with Mode.
  * **Encoding:** One-Hot Encoding for categorical variables.

### Model Performance

We compared a Generalized Linear Model (GLM) against LightGBM.

  * **GLM (ElasticNet):** Serves as a strong linear baseline.
  * **LightGBM:** Significantly outperformed the linear model, capturing non-linear relationships in the financial data.

**Receiver Operating Characteristic (ROC) Curve:**
*The LightGBM model achieves a near-perfect AUC on the test set, demonstrating superior predictive power.*

### Model Interpretation

**Feature Importance:**
We utilized permutation-based feature importance to identify key drivers of default.

  * **Upfront_charges**, **credit_type** are among the most influential predictors.
  * The high importance of `Upfront_charges` in LightGBM suggests a strong non-linear signal that the linear model failed to fully capture.

**Partial Dependence Plots (PDP):**
PDPs illustrate how the predicted probability of default changes as a single feature varies.

  * **LTV:** Higher Loan-to-Value ratios correlate with increased risk (as expected).
  * **Upfront_charges:** Shows a non-linear relationship where risk decreases as charges increase (potentially indicating borrower liquidity).

## 🛠️ Technologies Used

  * **Core:** Python 3.12, Pandas, NumPy
  * **Machine Learning:** Scikit-learn, LightGBM
  * **Interpretation:** Dalex
  * **Visualization:** Matplotlib, Seaborn, Plotly (via Dalex)
  * **Data Format:** Parquet (for efficient I/O)

## 📝 License

This project is open-source. Feel free to use and modify the code.
