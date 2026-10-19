from pathlib import Path

# Paths 
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "heart.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
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
MISSING_FLAG = {"Cholesterol": "chol_missing"}

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
