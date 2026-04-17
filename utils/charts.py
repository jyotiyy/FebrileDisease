

from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List



DISEASE_COLOURS = {
    "Typhoid":      "rgba(31, 119, 180, 0.85)",
    "Malaria":      "rgba(214, 39, 40,  0.85)",
    "Viral_Fever":  "rgba(44, 160, 44,  0.85)",
    "Rickettsial":  "rgba(255, 127, 14, 0.85)",
}

ACCENT        = "rgba(31, 119, 180, 1.0)"
GRID_COLOR    = "rgba(220, 220, 220, 0.5)"
BG_COLOR      = "rgba(0, 0, 0, 0)"
PAPER_BG      = "rgba(255, 255, 255, 1.0)"
TEXT_COLOR    = "#1a1a2e"
FONT_FAMILY   = "IBM Plex Sans, sans-serif"


def _base_layout(title: str, height: int = 380) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color=TEXT_COLOR, family=FONT_FAMILY),
            x=0.02,
        ),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=11),
        showlegend=False,
    )



def disease_probability_chart(probs: Dict[str, float]) -> go.Figure:
    
    diseases = list(probs.keys())
    values   = [round(v * 100, 1) for v in probs.values()]
    max_idx  = values.index(max(values))

    colours = [
        DISEASE_COLOURS.get(d, "rgba(100, 100, 100, 0.7)")
        for d in diseases
    ]

    colours[max_idx] = colours[max_idx].replace("0.85", "1.0")

    fig = go.Figure(
        go.Bar(
            x=values,
            y=diseases,
            orientation="h",
            marker_color=colours,
            marker_line_width=0,
            text=[f"{v}%" for v in values],
            textposition="outside",
            textfont=dict(size=11, color=TEXT_COLOR),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )

    layout = _base_layout("Diagnostic Probability Distribution", height=320)
    layout.update(
        xaxis=dict(
            title="Probability (%)",
            range=[0, 110],
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
        ),
        yaxis=dict(showgrid=False, autorange="reversed"),
        bargap=0.35,
    )
    fig.update_layout(**layout)
    return fig



def symptom_radar_chart(
    patient_inputs: Dict[str, int],
    feature_weights: Dict[str, float] | None = None,
) -> go.Figure:
  
    symptom_map = {
        "fever":      "Fever",
        "chills":     "Chills",
        "headache":   "Headache",
        "nausea":     "Nausea",
        "body_pain":  "Body Pain",
        "fatigue":    "Fatigue",
        "rash":       "Rash",
        "vomiting":   "Vomiting",
    }

    labels = list(symptom_map.values())
    keys   = list(symptom_map.keys())

    if feature_weights:
        values = [round(feature_weights.get(k, 0.0), 2) for k in keys]
        title  = "Symptom Importance Weights"
    else:
        values = [int(patient_inputs.get(k, 0)) for k in keys]
        title  = "Patient Symptom Profile"

    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.20)",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=5, color=ACCENT),
            hovertemplate="%{theta}: %{r}<extra></extra>",
        )
    )

    layout = _base_layout(title, height=380)
    layout.update(
        polar=dict(
            bgcolor="rgba(245, 248, 252, 0.8)",
            radialaxis=dict(
                visible=True,
                range=[0, 1.1],
                showline=False,
                gridcolor=GRID_COLOR,
                tickfont=dict(size=9),
            ),
            angularaxis=dict(
                gridcolor=GRID_COLOR,
                linecolor=GRID_COLOR,
                tickfont=dict(size=10, color=TEXT_COLOR),
            ),
        )
    )
    fig.update_layout(**layout)
    return fig



def treatment_score_chart(all_options: List[Dict]) -> go.Figure:
    """
    Horizontal bar chart for treatment option scores.
    """
    if not all_options:
        return go.Figure()

    names  = [t["name"] for t in all_options]
    scores = [round(t["score"] * 100, 1) for t in all_options]

    colours = [
        "rgba(31, 119, 180, 1.0)" if i == 0 else "rgba(31, 119, 180, 0.40)"
        for i in range(len(names))
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker_color=colours,
            marker_line_width=0,
            text=[f"{s}%" for s in scores],
            textposition="outside",
            textfont=dict(size=10, color=TEXT_COLOR),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )

    layout = _base_layout("Treatment Option Scores (Hill Climbing Optimised)", height=280)
    layout.update(
        xaxis=dict(
            title="Composite Score (%)",
            range=[0, 115],
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
        ),
        yaxis=dict(showgrid=False, autorange="reversed"),
        bargap=0.35,
    )
    fig.update_layout(**layout)
    return fig
