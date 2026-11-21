import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted, validate_data

from heart import config as cfg


# Preprocessing function to replace Cholesterol and RestingBPzeros with NaN and create missing value flags
def zeros_to_nan(X):
    X = X.copy()
    X[cfg.MISSING_FLAG] = (X["Cholesterol"] == 0).astype(int)
    X[cfg.ZERO_AS_MISSING] = X[cfg.ZERO_AS_MISSING].replace(0, np.nan)
    return X


# Variant for 04_preprocessing.ipynb: add the flag but keep the zeros as recorded values
def flag_only(X):
    X = X.copy()
    X[cfg.MISSING_FLAG] = (X["Cholesterol"] == 0).astype(int)
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


# Supervised discretiser (04_preprocessing.ipynb): for each column, fit a small decision tree
# on that column alone and use its split points as bin edges. A stand-in for MDL
# discretisation, which scikit-learn does not provide. Outputs the bin index per column.
class TreeBinner(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, n_bins=4, min_samples_leaf=20, random_state=cfg.SEED):
        self.n_bins = n_bins
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        X = validate_data(self, X)
        self.edges_ = []
        for j in range(X.shape[1]):
            tree = DecisionTreeClassifier(max_leaf_nodes=self.n_bins,
                                          min_samples_leaf=self.min_samples_leaf,
                                          random_state=self.random_state)
            tree.fit(X[:, [j]], y)
            t = tree.tree_
            self.edges_.append(np.sort(t.threshold[t.children_left != -1]))
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return np.column_stack([np.searchsorted(e, X[:, j], side="right")
                                for j, e in enumerate(self.edges_)]).astype(float)


SCALERS = {
    "yj_robust": lambda: [PowerTransformer(method="yeo-johnson", standardize=False), RobustScaler()],
    "standard": lambda: [StandardScaler()],
    "minmax": lambda: [MinMaxScaler()],
    "robust": lambda: [RobustScaler()],
}


def _binner(discretize):
    # discretize = (strategy, n_bins); bins are one-hot encoded afterwards.
    strategy, k = discretize
    if strategy == "tree":
        return TreeBinner(n_bins=k)
    return KBinsDiscretizer(n_bins=k, encode="ordinal", strategy=strategy,
                            quantile_method="averaged_inverted_cdf")


# Build the preprocessing pipeline
# The first three arguments are the ones used by notebooks 02 and 03; their defaults are unchanged.
# The others are only used by 04_preprocessing.ipynb, and their defaults reproduce the base pipeline:
#   missing    "median" (zeros -> NaN -> median) | "knn" (zeros -> NaN -> KNN imputer) | "zero" (keep zeros)
#   scaler     key of SCALERS; ignored when scale=False
#   discretize None | ("quantile" | "uniform" | "tree", n_bins): replaces scaling of the numeric columns
#   drop_cols  numeric columns to leave out (e.g. ("Cholesterol",))
def build_preprocessor(scale=True, chol_flag=True, clip_outliers=True,
                       missing="median", scaler="yj_robust", discretize=None, drop_cols=()):
    num_cols = [c for c in cfg.NUM_COLS if c not in drop_cols]

    # Clip on the original scale (limits stay in clinical units), before imputing so
    # the limits are computed on recorded values only, not on the imputed median.
    num_steps = []
    if clip_outliers:
        num_steps.append(IQRClipper())
    if missing == "knn":
        # KNN imputation needs comparable units, so standardise first
        num_steps += [StandardScaler(), KNNImputer(n_neighbors=5)]
    else:
        # Median imputation (Cholesterol is right-skewed). With missing="zero" there is
        # nothing left to impute except never-seen NaNs.
        num_steps.append(SimpleImputer(strategy="median"))
    if discretize is not None:
        num_steps += [_binner(discretize),
                      OneHotEncoder(handle_unknown="ignore", sparse_output=False)]
    elif scale:
        num_steps += SCALERS[scaler]()

    # drop="if_binary" gives one 0/1 column for two-level columns (Sex, ExerciseAngina,
    # FastingBS) and keeps every level for the rest (ChestPainType, RestingECG, ST_Slope).
    # Levels not seen in training become all zeros instead of raising an error.
    onehot = OneHotEncoder(
        drop="if_binary", handle_unknown="ignore", sparse_output=False
    )

    transformers = [
        ("num", make_pipeline(*num_steps), num_cols),
        ("cat", onehot, cfg.BINARY_COLS + cfg.NOMINAL_COLS),
    ]
    if chol_flag:
        transformers.append(("flag", "passthrough", [cfg.MISSING_FLAG]))

    # Columns are picked by name; any column not listed is dropped.
    encode = ColumnTransformer(transformers, verbose_feature_names_out=False)
    # Return a DataFrame with real column names (useful for SHAP later).
    encode.set_output(transform="pandas")

    clean = zeros_to_nan if missing != "zero" else flag_only
    return Pipeline([("clean", FunctionTransformer(clean)), ("encode", encode)])
