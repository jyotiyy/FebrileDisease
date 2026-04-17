from __future__ import annotations
import os
import sys
import json
from datetime import datetime

import streamlit as st
import pandas as pd


sys.path.insert(0, os.path.dirname(__file__))

from model.train     import train, DATA_PATH, MODEL_PATH
from model.inference import load_model, run_inference, top_contributing_symptoms
from utils.preprocessing import preprocess_single, compute_severity_score
from utils.treatment import recommend_treatment
from utils.charts    import (
    disease_probability_chart,
    symptom_radar_chart,
    treatment_score_chart,
)


st.set_page_config(
    page_title="Febrile DSS – Clinical Decision Support",
    page_icon="assets/icon.png" if os.path.exists("assets/icon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* ── Base typography ── */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    .stApp {
        background-color: #f0f4f8;
        color: #111827;
    }

    /* ── Sidebar shell ── */
    section[data-testid="stSidebar"] {
        background-color: #0d1b2a !important;
    }

    /* All text inside sidebar → light */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }

    /* Widget labels (checkbox text, slider label, number input label) */
    section[data-testid="stSidebar"] .stCheckbox label p,
    section[data-testid="stSidebar"] .stCheckbox span,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #cbd5e1 !important;
        font-size: 0.85rem !important;
    }

    /* Number input text */
    section[data-testid="stSidebar"] input[type="number"] {
        color: #111827 !important;
        background-color: #ffffff !important;
        border: 1px solid #64748b !important;
    }

    /* Checkbox tick box border */
    section[data-testid="stSidebar"] [data-testid="stCheckbox"] input + div {
        border-color: #94a3b8 !important;
    }

    /* Slider track/thumb */
    section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
        color: #93c5fd !important;
    }

    /* Sidebar markdown bold labels */
    section[data-testid="stSidebar"] strong {
        color: #93c5fd !important;
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    /* Sidebar button */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #2563eb !important;
    }

    /* ── Header strip ── */
    .dss-header {
        background: linear-gradient(90deg, #0d1b2a 0%, #1e40af 100%);
        padding: 18px 28px;
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .dss-header h1 {
        color: #ffffff !important;
        font-size: 1.35rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.02em;
    }
    .dss-header span {
        color: #bfdbfe !important;
        font-size: 0.78rem;
        font-family: 'IBM Plex Mono', monospace;
        white-space: nowrap;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #dde3ec;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .metric-card .label {
        font-size: 0.68rem;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .metric-card .value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1e40af !important;
        line-height: 1.1;
    }
    .metric-card .sub {
        font-size: 0.72rem;
        color: #94a3b8 !important;
        margin-top: 4px;
    }

    /* ── Section titles ── */
    .section-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #374151 !important;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        border-bottom: 2px solid #dde3ec;
        padding-bottom: 6px;
        margin-bottom: 14px;
    }

    /* ── Diagnosis banner ── */
    .diagnosis-banner {
        background: #eff6ff;
        border-left: 4px solid #1e40af;
        border-radius: 0 6px 6px 0;
        padding: 14px 20px;
        margin-bottom: 16px;
    }
    .diagnosis-banner .disease-name {
        font-size: 1.45rem;
        font-weight: 700;
        color: #1e3a8a !important;
    }
    .diagnosis-banner .confidence {
        font-size: 0.83rem;
        color: #1e40af !important;
        margin-top: 4px;
    }

    /* ── Severity badges ── */
    .severity-low {
        background: #dcfce7; color: #166534 !important;
        border-radius: 12px; padding: 3px 10px;
        font-size: 0.78rem; font-weight: 600;
    }
    .severity-medium {
        background: #fef9c3; color: #854d0e !important;
        border-radius: 12px; padding: 3px 10px;
        font-size: 0.78rem; font-weight: 600;
    }
    .severity-high {
        background: #fee2e2; color: #991b1b !important;
        border-radius: 12px; padding: 3px 10px;
        font-size: 0.78rem; font-weight: 600;
    }

    /* ── Treatment cards ── */
    .treatment-card {
        background: #ffffff;
        border: 1px solid #dde3ec;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .treatment-card.best {
        border-color: #3b82f6;
        border-left: 4px solid #1e40af;
        border-radius: 0 8px 8px 0;
    }
    .treatment-card .tx-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #111827 !important;
    }
    .treatment-card .tx-notes {
        font-size: 0.82rem;
        color: #475569 !important;
        margin-top: 6px;
        line-height: 1.5;
    }

    /* ── Data table ── */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.83rem;
    }
    .styled-table th {
        background: #f1f5f9;
        color: #374151 !important;
        padding: 9px 14px;
        text-align: left;
        font-weight: 700;
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        border-bottom: 2px solid #dde3ec;
    }
    .styled-table td {
        padding: 9px 14px;
        border-bottom: 1px solid #f1f5f9;
        color: #1f2937 !important;
    }
    .styled-table tr:hover td {
        background: #f8fafc;
    }

    /* ── Placeholder / welcome ── */
    .welcome-box {
        text-align: center;
        padding: 60px 20px;
    }
    .welcome-box .icon {
        font-size: 2.5rem;
        opacity: 0.25;
        margin-bottom: 14px;
        color: #64748b;
    }
    .welcome-box p {
        color: #64748b !important;
    }

    /* ── Footer ── */
    .dss-footer {
        margin-top: 40px;
        padding: 16px;
        text-align: center;
        font-size: 0.72rem;
        color: #94a3b8 !important;
        border-top: 1px solid #dde3ec;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background-color: #1e40af !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
    }
    .stDownloadButton > button:hover {
        background-color: #2563eb !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)



if "model" not in st.session_state:
    st.session_state["model"] = None
if "result" not in st.session_state:
    st.session_state["result"] = None



def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Training Bayesian Network model (first run) ..."):
            train(DATA_PATH, MODEL_PATH)
    return load_model(MODEL_PATH)



with st.sidebar:
    st.markdown(
        "<div style='padding:10px 0 20px; font-size:1.05rem; font-weight:600;"
        "letter-spacing:0.05em; color:#ffffff;'>PATIENT INPUT</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Demographic**")
    age = st.number_input("Age (years)", min_value=1, max_value=100, value=32, step=1)

    st.markdown("<br>**Clinical Symptoms**", unsafe_allow_html=True)

    fever      = st.checkbox("Fever",       value=True)
    chills     = st.checkbox("Chills",      value=False)
    headache   = st.checkbox("Headache",    value=True)
    nausea     = st.checkbox("Nausea",      value=False)
    body_pain  = st.checkbox("Body Pain",   value=True)
    fatigue    = st.checkbox("Fatigue",     value=True)
    rash       = st.checkbox("Rash",        value=False)
    vomiting   = st.checkbox("Vomiting",    value=False)

    st.markdown("<br>**Laboratory**", unsafe_allow_html=True)
    platelets = st.number_input(
        "Platelet Count (cells/µL)",
        min_value=1_000,
        max_value=800_000,
        value=150_000,
        step=5_000,
        help="Normal range: 150,000 – 400,000 cells/µL",
    )

    st.markdown("")
    predict_btn = st.button("Run Diagnosis", use_container_width=True, type="primary")



now = datetime.now().strftime("%d %b %Y  %H:%M")
st.markdown(
    f"""
    <div class="dss-header">
        <h1>Febrile Illness Decision Support System</h1>
        <span>Bayesian Diagnostic AI &nbsp;|&nbsp; {now}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    model = get_model()
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()



if predict_btn:
    raw_inputs = {
        "fever":     int(fever),
        "chills":    int(chills),
        "headache":  int(headache),
        "nausea":    int(nausea),
        "body_pain": int(body_pain),
        "fatigue":   int(fatigue),
        "rash":      int(rash),
        "vomiting":  int(vomiting),
        "platelets": int(platelets),
        "age":       int(age),
    }

    with st.spinner("Computing probabilistic diagnosis ..."):
        try:
            evidence  = preprocess_single(raw_inputs)
            probs     = run_inference(evidence, model)
            severity  = compute_severity_score(raw_inputs)
            top_dis   = max(probs, key=probs.get)
            confidence = probs[top_dis]

            active_symptoms = {k for k in evidence if evidence[k] == 1}
            treatment_rec   = recommend_treatment(probs, active_symptoms, severity)

            contrib = top_contributing_symptoms(evidence, top_dis, model)

            st.session_state["result"] = {
                "inputs":      raw_inputs,
                "evidence":    evidence,
                "probs":       probs,
                "severity":    severity,
                "top_disease": top_dis,
                "confidence":  confidence,
                "treatment":   treatment_rec,
                "contrib":     contrib,
            }
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.session_state["result"] = None


res = st.session_state.get("result")

if res is None:
    st.markdown(
        """
        <div class="welcome-box">
            <div class="icon">✚</div>
            <p style='font-size:1.0rem; font-weight:500;'>
                Complete the patient form in the sidebar and click <strong>Run Diagnosis</strong>.
            </p>
            <p style='font-size:0.85rem;'>
                The system uses a Discrete Bayesian Network trained on febrile illness data
                to compute probabilistic differential diagnoses.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


sev = res["severity"]
sev_class = (
    "severity-low"    if sev < 35 else
    "severity-medium" if sev < 65 else
    "severity-high"
)
sev_label = (
    "Low"      if sev < 35 else
    "Moderate" if sev < 65 else
    "High"
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""<div class='metric-card'>
            <div class='label'>Top Diagnosis</div>
            <div class='value' style='font-size:1.15rem;'>{res['top_disease'].replace('_',' ')}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class='metric-card'>
            <div class='label'>Confidence</div>
            <div class='value'>{res['confidence']*100:.1f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""<div class='metric-card'>
            <div class='label'>Severity Score</div>
            <div class='value'>{sev:.0f}<span style='font-size:1rem; color:#94a3b8;'>/100</span></div>
            <div class='sub'><span class='{sev_class}'>{sev_label}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""<div class='metric-card'>
            <div class='label'>Platelet Count</div>
            <div class='value' style='font-size:1.1rem;'>{res['inputs']['platelets']:,}</div>
            <div class='sub'>cells / µL</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Diagnosis banner ──────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class='diagnosis-banner'>
        <div class='disease-name'>{res['top_disease'].replace('_',' ')}</div>
        <div class='confidence'>
            Posterior probability: {res['confidence']*100:.1f}%
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Severity: <span class='{sev_class}'>{sev_label} ({sev:.0f}/100)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("<div class='section-title'>Differential Probability</div>", unsafe_allow_html=True)
    fig_prob = disease_probability_chart(res["probs"])
    fig_prob.update_layout(
        font=dict(family="IBM Plex Sans, sans-serif", color="#111827"),
        xaxis=dict(
            tickfont=dict(color="#111827", size=12),
            title_font=dict(color="#111827", size=12),
            gridcolor="#e2e8f0",
            linecolor="#cbd5e1",
        ),
        yaxis=dict(
            tickfont=dict(color="#111827", size=12),
            title_font=dict(color="#111827", size=12),
            gridcolor="#e2e8f0",
            linecolor="#cbd5e1",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
    )
    st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})

with col_right:
    st.markdown("<div class='section-title'>Symptom Importance (Contribution)</div>", unsafe_allow_html=True)
    radar_weights = {k: v for k, v in res["contrib"].items() if k != "platelets_disc"}
    fig_radar = symptom_radar_chart(res["inputs"], feature_weights=radar_weights)
    fig_radar.update_layout(
        font=dict(family="IBM Plex Sans, sans-serif", color="#111827"),
        polar=dict(
            angularaxis=dict(tickfont=dict(color="#111827", size=12), linecolor="#cbd5e1"),
            radialaxis=dict(tickfont=dict(color="#374151", size=10), gridcolor="#e2e8f0", linecolor="#cbd5e1"),
            bgcolor="#ffffff",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)


st.markdown("<div class='section-title'>Full Probability Distribution</div>", unsafe_allow_html=True)
prob_rows = "".join(
    f"<tr><td>{d.replace('_',' ')}</td>"
    f"<td><div style='width:{v*100:.1f}%; background:#1e40af; height:6px; border-radius:3px;'></div></td>"
    f"<td style='font-family:IBM Plex Mono,monospace; color:#1f2937;'>{v*100:.2f}%</td></tr>"
    for d, v in sorted(res["probs"].items(), key=lambda x: x[1], reverse=True)
)
st.markdown(
    f"""<table class='styled-table'>
    <thead><tr><th>Disease</th><th style='width:50%'>Probability</th><th>Value</th></tr></thead>
    <tbody>{prob_rows}</tbody></table>""",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)


st.markdown(
    "<div class='section-title'>Treatment Recommendation (Hill Climbing Optimised)</div>",
    unsafe_allow_html=True,
)

tx = res["treatment"]

st.markdown(
    f"""
    <div class='treatment-card best'>
        <div class='tx-name'>{tx['treatment']}</div>
        <div class='tx-notes'>{tx['notes']}</div>
        <div style='margin-top:10px; font-size:0.75rem; color:#1e40af; font-weight:600;'>
            Optimised Score: {tx['score']*100:.1f}%
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Disease: {tx['disease'].replace('_',' ')}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if len(tx["all_options"]) > 1:
    with st.expander("View All Treatment Options"):
        fig_tx = treatment_score_chart(tx["all_options"])
        fig_tx.update_layout(
            font=dict(family="IBM Plex Sans, sans-serif", color="#111827"),
            xaxis=dict(
                tickfont=dict(color="#111827", size=12),
                title_font=dict(color="#111827", size=12),
                gridcolor="#e2e8f0",
                linecolor="#cbd5e1",
            ),
            yaxis=dict(
                tickfont=dict(color="#111827", size=12),
                title_font=dict(color="#111827", size=12),
                gridcolor="#e2e8f0",
                linecolor="#cbd5e1",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_tx, use_container_width=True, config={"displayModeBar": False})

        alt_rows = "".join(
            f"<tr><td>{'&#9733; ' if i==0 else ''}{opt['name']}</td>"
            f"<td style='font-family:IBM Plex Mono,monospace;'>{opt['score']*100:.1f}%</td>"
            f"<td>{opt['notes']}</td></tr>"
            for i, opt in enumerate(tx["all_options"])
        )
        st.markdown(
            f"""<table class='styled-table'>
            <thead><tr><th>Treatment</th><th>Score</th><th>Notes</th></tr></thead>
            <tbody>{alt_rows}</tbody></table>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)


st.markdown("<div class='section-title'>Top Contributing Symptoms</div>", unsafe_allow_html=True)

contrib_sorted = sorted(
    [(k, v) for k, v in res["contrib"].items() if k != "platelets_disc"],
    key=lambda x: x[1], reverse=True
)
contrib_rows = "".join(
    f"<tr><td>{k.replace('_',' ').title()}</td>"
    f"<td><div style='width:{v*100:.1f}%; background:{'#1e40af' if v>0.5 else '#93c5fd'}; height:8px; border-radius:4px;'></div></td>"
    f"<td style='font-family:IBM Plex Mono,monospace; color:#1f2937;'>{v:.3f}</td></tr>"
    for k, v in contrib_sorted
)
st.markdown(
    f"""<table class='styled-table'>
    <thead><tr><th>Symptom</th><th style='width:50%'>Relative Weight</th><th>Score</th></tr></thead>
    <tbody>{contrib_rows}</tbody></table>""",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)


st.markdown("<div class='section-title'>Export</div>", unsafe_allow_html=True)

report = {
    "timestamp": now,
    "patient": {
        "age": res["inputs"]["age"],
        "symptoms": {k: bool(v) for k, v in res["inputs"].items() if k not in ("age", "platelets")},
        "platelets": res["inputs"]["platelets"],
    },
    "diagnosis": {
        "top_disease": res["top_disease"],
        "confidence":  round(res["confidence"] * 100, 2),
        "severity_score": res["severity"],
        "all_probabilities": {k: round(v * 100, 2) for k, v in res["probs"].items()},
    },
    "treatment": {
        "recommended": tx["treatment"],
        "score":       round(tx["score"] * 100, 2),
        "notes":       tx["notes"],
    },
    "contributing_symptoms": {k: round(v, 3) for k, v in contrib_sorted},
}

st.download_button(
    label="Download Diagnostic Report (JSON)",
    data=json.dumps(report, indent=2),
    file_name=f"DSS_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json",
)


st.markdown(
    """
    <div class="dss-footer">
        Febrile Illness DSS &nbsp;|&nbsp; For clinical decision support only.
        Not a substitute for physician judgement. &nbsp;|&nbsp;
        Bayesian Network + Hill Climbing Inference Engine
    </div>
    """,
    unsafe_allow_html=True,
)