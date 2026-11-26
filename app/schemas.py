"""Request and response schemas for the prediction API.

The input schema mirrors the 11 raw features the model was trained on, so a client sends a
patient exactly as recorded and the pipeline inside the model does all the preprocessing.
Allowed values and ranges come from config.py, the single source of truth for both the
model and the API.
"""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from heart import config as cfg

LEVELS = cfg.CATEGORY_LEVELS
RANGES = cfg.INPUT_RANGES


def _range(col: str) -> dict:
    """Field bounds for a numeric feature (inclusive on both ends)."""
    low, high = RANGES[col]
    return {"ge": low, "le": high}


class PatientIn(BaseModel):
    """One patient's raw features, as recorded at the clinic."""

    # Reject unknown fields: a typo like "cholesterol" would otherwise be silently ignored
    # while the real field fell back to its default.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{
                "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140,
                "Cholesterol": 239, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 140,
                "ExerciseAngina": "Y", "Oldpeak": 1.2, "ST_Slope": "Flat",
            }]
        },
    )

    Age: int = Field(**_range("Age"), description="Age in years")
    Sex: Literal[tuple(LEVELS["Sex"])] = Field(description="M = male, F = female")
    ChestPainType: Literal[tuple(LEVELS["ChestPainType"])] = Field(
        description="ATA = atypical angina, NAP = non-anginal pain, "
                    "ASY = asymptomatic, TA = typical angina")
    RestingBP: int = Field(**_range("RestingBP"), description="Resting blood pressure, mmHg")
    Cholesterol: int | None = Field(
        default=None, **_range("Cholesterol"),
        description="Serum cholesterol, mg/dL. Leave empty (null) if not measured.")
    FastingBS: Literal[tuple(LEVELS["FastingBS"])] = Field(
        description="Fasting blood sugar > 120 mg/dL: 1 = yes, 0 = no")
    RestingECG: Literal[tuple(LEVELS["RestingECG"])] = Field(
        description="Normal, ST = ST-T wave abnormality, LVH = left ventricular hypertrophy")
    MaxHR: int = Field(**_range("MaxHR"), description="Maximum heart rate achieved, bpm")
    ExerciseAngina: Literal[tuple(LEVELS["ExerciseAngina"])] = Field(
        description="Exercise-induced angina: Y = yes, N = no")
    Oldpeak: float = Field(**_range("Oldpeak"),
                           description="ST depression induced by exercise relative to rest, mm")
    ST_Slope: Literal[tuple(LEVELS["ST_Slope"])] = Field(
        description="Slope of the peak exercise ST segment: Up, Flat, Down")


class PredictionOut(BaseModel):
    """The model's answer for one patient."""

    prediction: Literal[0, 1] = Field(description="1 = heart disease predicted, 0 = not")
    probability: float = Field(ge=0, le=1, description="Predicted probability of heart disease")
    threshold: float = Field(ge=0, le=1,
                             description="Probability cut-off used to turn it into prediction")


def to_model_input(patient: PatientIn) -> pd.DataFrame:
    """Turn a validated patient into the one-row DataFrame the model pipeline expects.

    The training data recorded "cholesterol not measured" as 0, and the pipeline relies on
    that (zeros_to_nan turns it into NaN + the chol_missing flag). The API asks for null
    instead, which is clearer for clients, and translates it back to 0 here.
    """
    row = patient.model_dump()
    if row["Cholesterol"] is None:
        row["Cholesterol"] = 0
    # Same column order and integer dtypes as the training data (and the logged MLflow
    # signature); Oldpeak stays float, the categoricals stay strings.
    int_cols = [c for c in cfg.NUM_COLS if c != "Oldpeak"] + ["FastingBS"]
    return pd.DataFrame([row])[cfg.FEATURE_COLS].astype({c: "int64" for c in int_cols})
