import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
 
from heart import config as cfg

# Preprocessing function to replace Cholesterol and RestingBPzeros with NaN and create missing value flags
def zeros_to_nan(X):
    X = X.copy()
    X[cfg.MISSING_FLAG] = (X["Cholesterol"] == 0).astype(int)
    X[cfg.ZERO_AS_MISSING] = X[cfg.ZERO_AS_MISSING].replace(0, np.nan)
    return X

# Build a preprocessing pipeline that imputes missing values, scales numeric features, and one-hot encodes categorical features.
def build_preprocessor(scale=True, chol_flag=True):
    # Median imputation (Cholesterol is right-skewed), then optional scaling.
    num_steps = [SimpleImputer(strategy="median")]
    if scale:
        num_steps.append(StandardScaler())
 
    # drop="if_binary" gives one 0/1 column for two-level columns (Sex, ExerciseAngina,
    # FastingBS) and keeps every level for the rest (ChestPainType, RestingECG, ST_Slope).
    # Levels not seen in training become all zeros instead of raising an error.
    onehot = OneHotEncoder(
        drop="if_binary", handle_unknown="ignore", sparse_output=False
    )
 
    transformers = [
        ("num", make_pipeline(*num_steps), cfg.NUM_COLS),
        ("cat", onehot, cfg.BINARY_COLS + cfg.NOMINAL_COLS),
    ]
    if chol_flag:
        transformers.append(("flag", "passthrough", [cfg.MISSING_FLAG]))
 
    # Columns are picked by name; any column not listed is dropped.
    encode = ColumnTransformer(transformers, verbose_feature_names_out=False)
    # Return a DataFrame with real column names (useful for SHAP later).
    encode.set_output(transform="pandas")
 
    return Pipeline([("clean", FunctionTransformer(zeros_to_nan)), ("encode", encode)])

