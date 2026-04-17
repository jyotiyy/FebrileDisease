

from __future__ import annotations
import random
from typing import Dict, List, Tuple


TREATMENT_DB: Dict[str, List[Dict]] = {
    "Typhoid": [
        {
            "name": "Ceftriaxone IV (14 days)",
            "effectiveness": 0.95,
            "indications": {"fever", "vomiting", "nausea"},
            "contraindications": set(),
            "notes": "First-line for severe / complicated typhoid; IV route preferred when vomiting present.",
        },
        {
            "name": "Azithromycin Oral (7 days)",
            "effectiveness": 0.88,
            "indications": {"fever", "headache", "nausea"},
            "contraindications": {"vomiting"},
            "notes": "Preferred for uncomplicated typhoid; good enteric penetration.",
        },
        {
            "name": "Ciprofloxacin Oral (10–14 days)",
            "effectiveness": 0.82,
            "indications": {"fever", "body_pain"},
            "contraindications": set(),
            "notes": "Use only if local susceptibility confirmed; fluoroquinolone resistance rising.",
        },
        {
            "name": "Supportive Care + Antipyretics",
            "effectiveness": 0.40,
            "indications": {"fever", "fatigue"},
            "contraindications": set(),
            "notes": "Adjunct to antibiotics; ensure adequate hydration.",
        },
    ],
    "Malaria": [
        {
            "name": "Artemether-Lumefantrine (AL) 3-day",
            "effectiveness": 0.96,
            "indications": {"fever", "chills", "headache"},
            "contraindications": set(),
            "notes": "WHO first-line ACT for uncomplicated P. falciparum malaria.",
        },
        {
            "name": "Artesunate IV + Doxycycline",
            "effectiveness": 0.97,
            "indications": {"fever", "chills", "vomiting", "body_pain"},
            "contraindications": set(),
            "notes": "First-line for severe malaria; IV artesunate superior to quinine.",
        },
        {
            "name": "Chloroquine Oral (3 days)",
            "effectiveness": 0.75,
            "indications": {"fever", "chills"},
            "contraindications": set(),
            "notes": "For chloroquine-sensitive P. vivax / P. ovale; check local resistance pattern.",
        },
        {
            "name": "Primaquine (14-day radical cure)",
            "effectiveness": 0.70,
            "indications": {"fatigue"},
            "contraindications": set(),
            "notes": "Adjunct for P. vivax/ovale to prevent relapse; screen for G6PD deficiency first.",
        },
    ],
    "Viral_Fever": [
        {
            "name": "Paracetamol 500 mg q6h + Rest",
            "effectiveness": 0.85,
            "indications": {"fever", "headache", "body_pain"},
            "contraindications": set(),
            "notes": "Cornerstone of symptomatic viral fever management.",
        },
        {
            "name": "Oral Rehydration + Electrolytes",
            "effectiveness": 0.80,
            "indications": {"vomiting", "nausea", "fatigue"},
            "contraindications": set(),
            "notes": "Prevents dehydration; essential when GI symptoms present.",
        },
        {
            "name": "NSAIDs (Ibuprofen 400 mg TDS)",
            "effectiveness": 0.75,
            "indications": {"fever", "body_pain", "headache"},
            "contraindications": {"vomiting", "nausea"},
            "notes": "Anti-inflammatory benefit; avoid if dengue suspected (bleeding risk).",
        },
        {
            "name": "Antihistamines (Cetirizine)",
            "effectiveness": 0.55,
            "indications": {"rash"},
            "contraindications": set(),
            "notes": "Symptomatic relief for pruritus / urticaria associated with viral exanthem.",
        },
    ],
    "Rickettsial": [
        {
            "name": "Doxycycline 100 mg BD (7 days)",
            "effectiveness": 0.97,
            "indications": {"fever", "rash", "headache", "body_pain"},
            "contraindications": set(),
            "notes": "Drug of choice for all rickettsial infections including scrub typhus. Start empirically.",
        },
        {
            "name": "Azithromycin 500 mg OD (5 days)",
            "effectiveness": 0.88,
            "indications": {"fever", "rash"},
            "contraindications": set(),
            "notes": "Alternative when doxycycline contraindicated (pregnancy, children < 8 years).",
        },
        {
            "name": "Chloramphenicol IV",
            "effectiveness": 0.80,
            "indications": {"fever", "rash", "vomiting"},
            "contraindications": set(),
            "notes": "Reserve for severe cases or when tetracyclines are contraindicated.",
        },
        {
            "name": "Supportive Antipyretics + Wound Care",
            "effectiveness": 0.35,
            "indications": {"fever", "rash"},
            "contraindications": set(),
            "notes": "Eschar care; do NOT delay specific therapy while awaiting confirmation.",
        },
    ],
}



def _score_treatment(
    treatment: Dict,
    disease_prob: float,
    active_symptoms: set,
    severity_score: float,
) -> float:
    """
    Composite score (0–1):
      base  = effectiveness * disease_prob
      bonus = indication overlap (normalised)
      penalty = contraindication overlap * 0.25
      severity amplification
    """
    base    = treatment["effectiveness"] * disease_prob

    # Indication overlap
    n_indicated = len(treatment["indications"] & active_symptoms)
    n_total_ind = max(len(treatment["indications"]), 1)
    bonus = 0.15 * (n_indicated / n_total_ind)

    # Contraindication penalty
    n_contra = len(treatment["contraindications"] & active_symptoms)
    penalty  = 0.25 * n_contra

    # Severity amplification: higher severity → treatments with better
    # effectiveness get amplified
    sev_factor = 0.5 + 0.5 * (severity_score / 100)

    raw = (base + bonus - penalty) * sev_factor
    return max(0.0, min(1.0, raw))



def _hill_climb(
    candidates: List[Dict],
    disease_prob: float,
    active_symptoms: set,
    severity_score: float,
    restarts: int = 5,
) -> Tuple[Dict, float]:
 
    best_treatment = None
    best_score     = -1.0

    for _ in range(restarts):
        # Random starting point
        current = random.choice(candidates)
        current_score = _score_treatment(
            current, disease_prob, active_symptoms, severity_score
        )

        improved = True
        while improved:
            improved = False
            neighbours = [t for t in candidates if t["name"] != current["name"]]
            for neighbour in neighbours:
                n_score = _score_treatment(
                    neighbour, disease_prob, active_symptoms, severity_score
                )
                if n_score > current_score:
                    current       = neighbour
                    current_score = n_score
                    improved      = True
                    break   # first-improvement strategy

        if current_score > best_score:
            best_treatment = current
            best_score     = current_score

    return best_treatment, best_score



def recommend_treatment(
    disease_probs: Dict[str, float],
    active_symptoms: set,
    severity_score: float,
) -> Dict:
   
    top_disease = max(disease_probs, key=disease_probs.get)
    top_prob    = disease_probs[top_disease]

    candidates  = TREATMENT_DB.get(top_disease, [])
    if not candidates:
        return {
            "disease": top_disease,
            "treatment": "Consult Specialist",
            "score": 0.0,
            "notes": "No treatment protocol found; specialist referral required.",
            "all_options": [],
        }

    random.seed(42)
    best, score = _hill_climb(
        candidates, top_prob, active_symptoms, severity_score
    )

    
    all_options = sorted(
        [
            {
                "name": t["name"],
                "score": round(
                    _score_treatment(t, top_prob, active_symptoms, severity_score), 3
                ),
                "notes": t["notes"],
            }
            for t in candidates
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    return {
        "disease":     top_disease,
        "treatment":   best["name"],
        "score":       round(score, 3),
        "notes":       best["notes"],
        "all_options": all_options,
    }
