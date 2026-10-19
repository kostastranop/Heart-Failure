import pandas as pd
from sklearn.model_selection import train_test_split
 
from heart import config as cfg

# Data validation function
def validate(df):
    expected = cfg.FEATURE_COLS + [cfg.TARGET]
 
    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
 
    if df[expected].isna().any().any():
        raise ValueError("Unexpected NaN values in the data")
 
    for col, levels in cfg.CATEGORY_LEVELS.items():
        unexpected = set(df[col].unique()) - set(levels)
        if unexpected:
            raise ValueError(f"{col}: unexpected levels {sorted(map(str, unexpected))}")

# Data loading function
def load_raw(path=cfg.RAW_PATH):
    df = pd.read_csv(path)
    validate(df)
    return df

# Data splitting function
def make_splits(df, test_size=cfg.TEST_SIZE, seed=cfg.SEED):
    X = df[cfg.FEATURE_COLS]
    y = df[cfg.TARGET]

    # Create a stratification variable that combines the target and whether cholesterol is zero
    strata = y.astype(str) + "_" + (df["Cholesterol"] == 0).astype(str)
 
    return train_test_split(
        X, y, test_size=test_size, stratify=strata, random_state=seed
    )
