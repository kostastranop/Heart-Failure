"""Training-only resampling for 04_preprocessing.ipynb.

scikit-learn pipelines cannot change the number of rows, so the balancing and row-removal
variants wrap an estimator: `Resampled(estimator, method).fit(X, y)` resamples (X, y) and
then fits `estimator` on the result. Prediction is passed straight through. Because the
resampling happens inside fit, cross-validation only ever resamples the training folds;
validation folds and the test set keep their real rows and class balance (no leakage).

Methods
  "oversample"   duplicate random minority rows until both classes are the same size
  "undersample"  drop random majority rows until both classes are the same size
  "smote"        SMOTE (Chawla et al., 2002) with nominal columns copied, not interpolated
  "remove_outliers"  drop training rows with a numeric value outside the 1.5 x IQR fences
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.metaestimators import available_if

from heart import config as cfg


def _take(X, y, idx):
    Xs = X.iloc[idx] if hasattr(X, "iloc") else np.asarray(X)[idx]
    ys = y.iloc[idx] if hasattr(y, "iloc") else np.asarray(y)[idx]
    return Xs, ys


def _class_counts(y):
    classes, counts = np.unique(np.asarray(y), return_counts=True)
    return classes[counts.argmin()], classes[counts.argmax()], counts.max() - counts.min()


def oversample(X, y, rng, **_):
    ya = np.asarray(y)
    minority, _, n_new = _class_counts(ya)
    extra = rng.choice(np.flatnonzero(ya == minority), n_new, replace=True)
    return _take(X, y, np.concatenate([np.arange(len(ya)), extra]))


def undersample(X, y, rng, **_):
    ya = np.asarray(y)
    _, majority, n_drop = _class_counts(ya)
    drop = rng.choice(np.flatnonzero(ya == majority), n_drop, replace=False)
    return _take(X, y, np.setdiff1d(np.arange(len(ya)), drop))


def smote(X, y, rng, k_neighbors=5, **_):
    """SMOTE on the encoded features.

    Continuous columns (the numeric features) are interpolated between a minority row and one
    of its k nearest minority neighbours. All other columns (one-hot levels, binary features,
    the missing-value flag) are copied as a block from whichever of the two rows is closer to
    the interpolation point, so every synthetic row still has exactly one level per category
    (the idea behind SMOTE-NC). Neighbours are found on z-scored columns so that no feature
    dominates the distance through its units.
    """
    columns = list(X.columns) if hasattr(X, "columns") else None
    Xa, ya = np.asarray(X, float), np.asarray(y)
    minority, _, n_new = _class_counts(ya)
    Xm = Xa[ya == minority]

    # Numeric features keep their names after encoding; one-hot outputs are named "<col>_<level>".
    cont = (np.array([c in cfg.NUM_COLS for c in columns]) if columns is not None
            else np.ones(Xa.shape[1], bool))
    sd = Xm.std(axis=0)
    Z = (Xm - Xm.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(Z)
    neighbours = nn.kneighbors(Z, return_distance=False)[:, 1:]  # drop the row itself

    base = rng.integers(0, len(Xm), n_new)
    other = neighbours[base, rng.integers(0, k_neighbors, n_new)]
    gap = rng.random(n_new)[:, None]
    interpolated = Xm[base] + gap * (Xm[other] - Xm[base])
    copied = np.where(gap < 0.5, Xm[base], Xm[other])
    synthetic = np.where(cont, interpolated, copied)

    X_new = np.vstack([Xa, synthetic])
    y_new = np.concatenate([ya, np.full(n_new, minority)])
    if columns is not None:
        X_new = pd.DataFrame(X_new, columns=columns)
    return X_new, y_new


def remove_outliers(X, y, rng=None, k=1.5, **_):
    """Drop rows with any numeric feature outside [Q1 - k*IQR, Q3 + k*IQR] (raw data).

    Zeros in Cholesterol / RestingBP are treated as missing (as in the base pipeline), so they
    are not counted as outliers; missing values never cause a row to be removed.
    """
    num = X[cfg.NUM_COLS].astype(float).copy()
    num[cfg.ZERO_AS_MISSING] = num[cfg.ZERO_AS_MISSING].replace(0, np.nan)
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    outside = (num.lt(lo) | num.gt(hi)).any(axis=1).to_numpy()
    return _take(X, y, np.flatnonzero(~outside))


METHODS = {"oversample": oversample, "undersample": undersample, "smote": smote,
           "remove_outliers": remove_outliers}


class Resampled(ClassifierMixin, BaseEstimator):
    """Resample the training data inside fit, then fit `estimator` on it."""

    def __init__(self, estimator, method="smote", k_neighbors=5, random_state=cfg.SEED):
        self.estimator = estimator
        self.method = method
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X_res, y_res = METHODS[self.method](X, y, rng, k_neighbors=self.k_neighbors)
        self.estimator_ = clone(self.estimator).fit(X_res, y_res)
        self.classes_ = self.estimator_.classes_
        self.n_train_in_ = len(y)
        self.n_train_resampled_ = len(y_res)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X):
        return self.estimator_.decision_function(X)
