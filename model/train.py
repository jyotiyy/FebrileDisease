

from __future__ import annotations
import os
import pickle
import sys

import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

from utils.preprocessing import preprocess_dataframe, ALL_FEATURES, TARGET


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "bayesian_model.pkl")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


def build_structure() -> list:
    
    edges = [(feat, TARGET) for feat in ALL_FEATURES]
    return edges


def train(data_path: str = DATA_PATH, save_path: str = MODEL_PATH) -> DiscreteBayesianNetwork:
   
    print("[train] Loading dataset ...")
    df = pd.read_csv(data_path)
    print(f"[train] Raw rows: {len(df)}")

    print("[train] Preprocessing ...")
    df_clean = preprocess_dataframe(df)
    print(f"[train] Processed rows: {len(df_clean)}")
    print(df_clean[TARGET].value_counts())

    print("[train] Building Bayesian Network structure ...")
    edges = build_structure()
    model = DiscreteBayesianNetwork(edges)

    print("[train] Fitting CPDs with MaximumLikelihoodEstimator ...")
    model.fit(
        df_clean,
        estimator=MaximumLikelihoodEstimator,
        state_names={col: sorted(df_clean[col].unique().tolist()) for col in df_clean.columns},
    )

    print("[train] Validating model ...")
    assert model.check_model(), "Model validation failed!"
    print("[train] Model is valid.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[train] Model saved to {save_path}")

    return model


if __name__ == "__main__":
    train()
