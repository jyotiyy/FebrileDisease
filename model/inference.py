

from __future__ import annotations
import os
import pickle
from typing import Dict

from pgmpy.inference import VariableElimination

from utils.preprocessing import ALL_FEATURES, TARGET


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "bayesian_model.pkl")

DISEASE_LABELS = ["Malaria", "Rickettsial", "Typhoid", "Viral_Fever"]


def load_model(path: str = MODEL_PATH):
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained model not found at {path}. "
            "Please run model/train.py first."
        )
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def run_inference(
    evidence: Dict[str, int],
    model=None,
) -> Dict[str, float]:
    
    if model is None:
        model = load_model()

   
    valid_keys = set(ALL_FEATURES)
    clean_evidence = {k: v for k, v in evidence.items() if k in valid_keys}

    infer  = VariableElimination(model)
    result = infer.query(variables=[TARGET], evidence=clean_evidence, show_progress=False)

   
    state_names = result.state_names[TARGET]
    probs = {name: float(result.values[i]) for i, name in enumerate(state_names)}

   
    for label in DISEASE_LABELS:
        probs.setdefault(label, 0.0)

    
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    return probs


def top_contributing_symptoms(
    evidence: Dict[str, int],
    top_disease: str,
    model=None,
) -> Dict[str, float]:
   
    if model is None:
        model = load_model()

    baseline = run_inference(evidence, model).get(top_disease, 0.0)

    weights: Dict[str, float] = {}
    for feat in ALL_FEATURES:
        if feat not in evidence:
            weights[feat] = 0.0
            continue

        
        reduced = {k: v for k, v in evidence.items() if k != feat}
        prob_without = run_inference(reduced, model).get(top_disease, 0.0)

        
        lift = baseline - prob_without
        weights[feat] = max(0.0, lift)


    max_w = max(weights.values()) if weights else 1.0
    if max_w > 0:
        weights = {k: round(v / max_w, 3) for k, v in weights.items()}

    return weights
