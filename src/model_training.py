import sys
import warnings
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.exceptions import ConvergenceWarning
import lightgbm as lgb

# ---------------------------------------------------
# 1. Configuration & Paths
# ---------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_FILE = PROJECT_DIR / "data" / "processed" / "cleaned_data.parquet"
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Define feature groups based on the dataset
NUMERIC_FEATURES = [
    'loan_amount', 'income', 'Credit_Score', 'LTV', 'dtir1', 
    'term', 'Upfront_charges', 'property_value', 'age'
]

CATEGORICAL_FEATURES = [
    'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
    'Credit_Worthiness', 'open_credit', 'business_or_commercial',
    'Neg_ammortization', 'interest_only', 'lump_sum_payment',
    'construction_type', 'occupancy_type', 'Secured_by', 'total_units',
    'credit_type', 'co-applicant_credit_type', 'submission_of_application',
    'Region', 'Security_Type'
]

# --- Filter Warnings ---
# 1. Deprecation Warning (keep from before)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*'penalty' was deprecated.*")

# 2. GLM Convergence Warning (expected because we limited `max_iter`)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 3. LGBM Feature Name Warning (expected because Pipelines return numpy arrays)
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

# ---------------------------------------------------
# 2. Data Loading & Splitting
# ---------------------------------------------------
def load_and_split_data(test_size=0.2, random_state=42, max_rows=30000):
    """
    Load cleaned data and perform stratified split.
    
    Args:
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.
        max_rows (int): Maximum number of rows to use. If None, use full dataset.
                        Defaults to 30000 to speed up training during development/grading.
    """
    if not PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(f"File not found: {PROCESSED_DATA_FILE}")
    
    print(f"Loading data from {PROCESSED_DATA_FILE}...")
    df = pd.read_parquet(PROCESSED_DATA_FILE)
    target_col = 'Status'
    
    # Downsample using stratified sampling since data is too large
    if max_rows and len(df) > max_rows:
        print(
            f"⚠️  Downsampling data from {len(df)} to {max_rows} rows (Stratified) "
                + "since the training time may be too long..."
        )
        df, _ = train_test_split(
            df, 
            train_size=max_rows, 
            stratify=df[target_col], 
            random_state=random_state
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Validate columns exist
    valid_num = [c for c in NUMERIC_FEATURES if c in X.columns]
    valid_cat = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    
    print(f"Features selected: {len(valid_num)} numerical, {len(valid_cat)} categorical.")
    print(f"Total samples for training/testing: {len(X)}")

    # Stratified split ensures class balance is maintained
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y), valid_num, valid_cat

# ---------------------------------------------------
# 3. Pipeline Construction
# ---------------------------------------------------
def build_glm_pipeline(num_cols, cat_cols):
    """
    Builds a Logistic Regression pipeline with ElasticNet penalty.
    """
    # Pipeline for numerical features: Imputation -> Custom Scaling
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', CustomStandardScaler())
    ])

    # Pipeline for categorical features: Imputation -> One-Hot Encoding
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )

    # ElasticNet requires 'saga' solver
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='saga', penalty='elasticnet', max_iter=1000, random_state=42))
    ])
    
    return pipeline

def build_lgbm_pipeline(num_cols, cat_cols):
    """
    Builds a LightGBM pipeline.
    """
    # Tree models handle unscaled data well, but simple imputation is needed
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1))
    ])
    
    return pipeline

# ---------------------------------------------------
# 4. Hyperparameter Tuning
# ---------------------------------------------------
def tune_glm_model(X_train, y_train, num_cols, cat_cols):
    """
    Tune GLM using GridSearchCV.
    """
    print("\n--- Tuning GLM (Logistic Regression) ---")
    pipeline = build_glm_pipeline(num_cols, cat_cols)

    # Define hyperparameter grid
    param_grid = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__l1_ratio': [0.1, 0.5, 0.9]
    }

    # GridSearchCV uses Cross-Validation to find best params
    search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        scoring='f1', 
        n_jobs=-1,
        verbose=2,
    )
    
    search.fit(X_train, y_train)
    print(f"Best GLM Params: {search.best_params_}")
    print(f"Best GLM CV F1 Score: {search.best_score_:.4f}")
    
    return search.best_estimator_

def tune_lgbm_model(X_train, y_train, num_cols, cat_cols):
    """
    Tune LGBM using RandomizedSearchCV for efficiency.
    """
    print("\n--- Tuning LGBM Classifier ---")
    pipeline = build_lgbm_pipeline(num_cols, cat_cols)

    # Define hyperparameter distribution
    param_dist = {
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__n_estimators': [100, 200, 500],
        'classifier__num_leaves': [31, 50, 100],
        'classifier__min_child_weight': [1, 5, 10]
    }

    # RandomizedSearchCV allows exploring a wider range efficiently
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,  # Number of parameter settings that are sampled
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    search.fit(X_train, y_train)
    print(f"Best LGBM Params: {search.best_params_}")
    print(f"Best LGBM CV F1 Score: {search.best_score_:.4f}")
    
    return search.best_estimator_

# ---------------------------------------------------
# 5. Main Execution Flow
# ---------------------------------------------------
if __name__ == "__main__":
    # 0. Setup Project Path
    if str(PROJECT_DIR) not in sys.path:
        sys.path.append(str(PROJECT_DIR))
    from src.feature_engineering import CustomStandardScaler

    # 1. Load Data
    (X_train, X_test, y_train, y_test), num_cols, cat_cols = load_and_split_data(
        test_size=0.2, random_state=42, max_rows=30000,
    )
    print(f"Training set shape: {X_train.shape}")

    # 2. Tune GLM
    best_glm = tune_glm_model(X_train, y_train, num_cols, cat_cols)
    
    # 3. Tune LGBM
    best_lgbm = tune_lgbm_model(X_train, y_train, num_cols, cat_cols)
    
    # 4. Final Evaluation on Test Set
    print("\n--- Final Evaluation on Test Set ---")
    
    print("GLM Report:")
    y_pred_glm = best_glm.predict(X_test)
    print(classification_report(y_test, y_pred_glm))
    
    print("LGBM Report:")
    y_pred_lgbm = best_lgbm.predict(X_test)
    print(classification_report(y_test, y_pred_lgbm))
    
    # 5. Save Models
    print("\n--- Saving Models ---")
    
    glm_path = MODELS_DIR / "glm_model.joblib"
    lgbm_path = MODELS_DIR / "lgbm_model.joblib"
    
    print(f"Saving GLM model to {glm_path}...")
    joblib.dump(best_glm, glm_path)
    
    print(f"Saving LGBM model to {lgbm_path}...")
    joblib.dump(best_lgbm, lgbm_path)
    
    print("Models saved successfully!")
