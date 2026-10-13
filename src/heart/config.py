from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "heart.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

SEED = 42
TARGET = "HeartDisease"

NUM_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
BINARY_COLS = ["Sex", "ExerciseAngina"]
NOMINAL_COLS = ["ChestPainType", "RestingECG", "ST_Slope"]