

import pandas as pd
import numpy as np



PLATELET_BINS   = [0, 50_000, 100_000, 150_000, 200_000, float("inf")]
PLATELET_LABELS = [0, 1, 2, 3, 4]  

BINARY_FEATURES = [
    "fever", "chills", "headache", "nausea",
    "body_pain", "fatigue", "rash", "vomiting"
]

ALL_FEATURES = BINARY_FEATURES + ["platelets_disc"]
TARGET       = "disease"


def discretize_platelets(series: pd.Series) -> pd.Series:
    
    return pd.cut(
        series,
        bins=PLATELET_BINS,
        labels=PLATELET_LABELS,
        right=True
    ).astype(int)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

  
    for col in BINARY_FEATURES:
        df[col] = df[col].fillna(0).astype(int).clip(0, 1)

   
    df["platelets_disc"] = discretize_platelets(df["platelets"])

 
    keep_cols = ALL_FEATURES + [TARGET]
    df = df[keep_cols].dropna()

    
    for col in ALL_FEATURES:
        df[col] = df[col].astype(int)

    return df


def preprocess_single(inputs: dict) -> dict:
 
    processed = {}

    for feat in BINARY_FEATURES:
        processed[feat] = int(inputs.get(feat, 0))

    platelets = inputs.get("platelets", 150_000)
    disc = pd.cut(
        [platelets],
        bins=PLATELET_BINS,
        labels=PLATELET_LABELS,
        right=True
    )
    processed["platelets_disc"] = int(disc[0])

    return processed


def compute_severity_score(inputs: dict) -> float:
   
    symptom_score = sum(int(inputs.get(f, 0)) for f in BINARY_FEATURES)
    max_symptoms  = len(BINARY_FEATURES)

    platelets     = inputs.get("platelets", 150_000)
    plat_severity = max(0.0, 1.0 - (platelets / 300_000))

    score = (symptom_score / max_symptoms) * 70 + plat_severity * 30
    return round(score, 1)
