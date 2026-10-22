import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.utils.validation import validate_data
 
from heart import config as cfg

# Preprocessing function to replace Cholesterol and RestingBPzeros with NaN and create missing value flags
def zeros_to_nan(X):
    X = X.copy()
    X[cfg.MISSING_FLAG] = (X["Cholesterol"] == 0).astype(int)
    X[cfg.ZERO_AS_MISSING] = X[cfg.ZERO_AS_MISSING].replace(0, np.nan)
    return X

# IQR Outlier clipper 
class IQRClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    # Clip each column to [Q1 - k*IQR, Q3 + k*IQR].
    def __init__(self, k=1.5):
        self.k = k
 
    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite="allow-nan")
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        self.lower_ = q1 - self.k * (q3 - q1)
        self.upper_ = q3 + self.k * (q3 - q1)
        return self
 
    def transform(self, X):
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        return np.clip(X, self.lower_, self.upper_)

# Build the preprocessing pipeline
def build_preprocessor(scale=True, chol_flag=True, clip_outliers=True):
    # Clip on the original scale (limits stay in clinical units), before imputing so
    # the limits are computed on recorded values only, not on the imputed median.
    num_steps = []
    if clip_outliers:
        num_steps.append(IQRClipper())
    # Median imputation (Cholesterol is right-skewed).
    num_steps.append(SimpleImputer(strategy="median"))
    if scale:
        num_steps.append(PowerTransformer(method="yeo-johnson", standardize=False))
        num_steps.append(RobustScaler())
 
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

