from pathlib import Path

# Paths 
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "heart.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
TRAIN_CSV = DATA_PROCESSED / "train.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"
FIGURES = ROOT / "reports" / "figures"

# MLflow: runs and model registry in a local SQLite file, run artifacts in mlruns/
# (both gitignored). The registry needs a database store, a plain folder is not enough.
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = ROOT / "mlruns"
EXPERIMENT_NAME = "heart-disease"
# Registry name; the app loads f"models:/{REGISTERED_MODEL}@champion"
REGISTERED_MODEL = "heart-disease"

# Constants
SEED = 42
TEST_SIZE = 0.2
TARGET = "HeartDisease"

# Columns
NUM_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
BINARY_COLS = ["Sex", "ExerciseAngina", "FastingBS"]
NOMINAL_COLS = ["ChestPainType", "RestingECG", "ST_Slope"]
FEATURE_COLS = NUM_COLS + BINARY_COLS + NOMINAL_COLS

# Missing values
ZERO_AS_MISSING = ["Cholesterol", "RestingBP"]
MISSING_FLAG = "chol_missing"

# Categorical levels
CATEGORY_LEVELS = {
    "Sex": ["M", "F"],
    "ChestPainType": ["ATA", "NAP", "ASY", "TA"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["Y", "N"],
    "ST_Slope": ["Up", "Flat", "Down"],
    "FastingBS": [0, 1],
    TARGET: [0, 1],
}

# API input limits (app/schemas.py): (min, max) of what the API accepts for each numeric feature.
# These are physiologically plausible limits, deliberately wider than the training data
# (shown in comments), so real but unusual patients are accepted while typos and unit mix-ups
# (e.g. cholesterol in mmol/L, temperature typed into MaxHR) are rejected with a 422.
INPUT_RANGES = {
    "Age": (18, 100),           # years, train 28-77; the dataset covers adults only
    "RestingBP": (60, 250),     # mmHg, train 80-200 (one 0 = recording error)
    "Cholesterol": (50, 700),   # mg/dL, train 85-603; "not measured" is sent as null, not 0
    "MaxHR": (50, 230),         # bpm, train 60-202
    "Oldpeak": (-3.0, 10.0),    # mm of ST depression, train -2.6-6.2
}

# Cross-validation: same repeated 10-fold scheme as 02_modeling.ipynb,
# so the CV numbers logged by train.py match the notebook.
CV_SPLITS = 10
CV_REPEATS = 5
# Bootstrap resamples of the test set, for the 95% interval around each test score
N_BOOTSTRAP = 1000
# Shuffles per feature for permutation importance (gives the error bars)
N_PERMUTATION_REPEATS = 30

# Chosen models (Phase 3 summary in 02_modeling.ipynb)
# logreg is the champion: tied with the forest on CV, but simpler and fully interpretable.
# random_forest is kept as a challenger to compare against in MLflow.
CHAMPION = "logreg"
CHALLENGER = "random_forest"
# Default threshold, the same one used in CV and on the test set.
DECISION_THRESHOLD = 0.5
# Full-precision Optuna best params, so a retrain reproduces the notebook exactly.
BEST_PARAMS = {
    "logreg": {
        "C": 0.1685178451496271,
        "l1_ratio": 0.13565654471298333,
        "solver": "saga",
        "max_iter": 5000,
    },
    "random_forest": {
        "n_estimators": 400,
        "max_depth": 7,
        "min_samples_leaf": 2,
        "max_features": 0.20724690147815145,
        "random_state": SEED,
        "n_jobs": 1,
    },
}
