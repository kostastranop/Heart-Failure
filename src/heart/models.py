from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
 
from heart import config as cfg
from heart.preprocessing import build_preprocessor

# Trees need no scaling, and no clipping outliers 
TREE_PREP = {"scale": False, "clip_outliers": False}

# function to return the models along with their fitting preprocessing
def get_specs(seed=cfg.SEED):
    
    # Return {name: (estimator, preprocessing options)}.
    return {
        # Baselines
        "dummy_majority": (DummyClassifier(strategy="most_frequent"), {}),
        "dummy_stratified": (DummyClassifier(strategy="stratified", random_state=seed), {}),
        "logreg": (LogisticRegression(max_iter=1000), {}),
        # Distance-based: needs scaled inputs, so it keeps the default preprocessing.
        "knn": (KNeighborsClassifier(n_neighbors=15), {}),
        # Trees: no scaling, no clipping.
        "decision_tree": (DecisionTreeClassifier(max_depth=4, random_state=seed), TREE_PREP),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
            TREE_PREP,
        ),
        # Boosters: modest settings, because the library defaults are meant for
        # much larger datasets than 734 rows.
        "xgboost": (
            XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=3,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=seed, n_jobs=1,
            ),
            TREE_PREP,
        ),
        "lightgbm": (
            LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=15,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                random_state=seed, n_jobs=1, verbose=-1,
            ),
            TREE_PREP,
        ),
        "catboost": (
            CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=4,
                random_seed=seed, thread_count=1, verbose=0, allow_writing_files=False,
            ),
            TREE_PREP,
        ),
    }

# Function to build the models
def build_model(name, seed=cfg.SEED, **prep_overrides):
    # Preprocessing + estimator as one pipeline.
 
    estimator, prep = get_specs(seed)[name]
    return make_pipeline(build_preprocessor(**{**prep, **prep_overrides}), estimator)
