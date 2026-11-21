from pathlib import Path

# Paths 
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "heart.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
TRAIN_CSV = DATA_PROCESSED / "train.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"
FIGURES = ROOT / "reports" / "figures"

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
