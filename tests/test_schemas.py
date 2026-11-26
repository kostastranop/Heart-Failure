"""The API input schema accepts real patients and rejects bad input before it reaches the model."""
import pandas as pd
import pytest
from pydantic import ValidationError

from app.schemas import PatientIn, to_model_input
from heart import config as cfg
from heart.data import load_splits

VALID = PatientIn.model_config["json_schema_extra"]["examples"][0]


def test_every_training_row_is_accepted():
    """The limits must never reject a patient like the ones the model was trained on.

    Cholesterol 0 means "not measured" in the data and is sent as null through the API;
    the single RestingBP 0 row is a recording error and is skipped.
    """
    X_train, X_test, _, _ = load_splits()
    X = pd.concat([X_train, X_test])
    X = X[X["RestingBP"] > 0]
    for row in X.to_dict("records"):
        row["Cholesterol"] = row["Cholesterol"] or None
        PatientIn(**row)


@pytest.mark.parametrize("change", [
    {"Age": 5},                    # below range
    {"Cholesterol": 6.2},          # mmol/L instead of mg/dL (and not an integer)
    {"Cholesterol": 0},            # "not measured" must be null, not 0
    {"Sex": "male"},               # not an allowed level
    {"ST_Slope": "up"},            # levels are case-sensitive
    {"FastingBS": 2},
    {"cholesterol": 200},          # unknown field (typo)
])
def test_bad_input_is_rejected(change):
    with pytest.raises(ValidationError):
        PatientIn(**(VALID | change))


def test_missing_required_field_is_rejected():
    payload = {k: v for k, v in VALID.items() if k != "MaxHR"}
    with pytest.raises(ValidationError):
        PatientIn(**payload)


def test_model_input_matches_training_layout():
    X_train, _, _, _ = load_splits()
    df = to_model_input(PatientIn(**VALID))
    assert list(df.columns) == cfg.FEATURE_COLS
    assert df.dtypes.equals(X_train[cfg.FEATURE_COLS].dtypes)


def test_unmeasured_cholesterol_becomes_zero():
    """The pipeline expects 0 for 'not measured' (it becomes NaN + the chol_missing flag)."""
    df = to_model_input(PatientIn(**(VALID | {"Cholesterol": None})))
    assert df.loc[0, "Cholesterol"] == 0
