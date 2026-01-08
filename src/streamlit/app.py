
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from joblib import load as joblib_load


st.set_page_config(
    page_title="Projet CO2 - Presentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500&display=swap");
:root {
  --bg: #f7f5f1;
  --ink: #1c1c1c;
  --muted: #5c5c5c;
  --accent: #0f4c5c;
  --accent-2: #e07a5f;
  --panel: #ffffff;
  --panel-border: #e7e2da;
}
html, body, [class*="stApp"] {
  background: radial-gradient(1200px 600px at 10% 10%, #f2efe9 0%, var(--bg) 50%, #f0ede7 100%);
  color: var(--ink);
  font-family: "IBM Plex Sans", sans-serif;
}
h1, h2, h3, h4 {
  font-family: "Space Grotesk", sans-serif;
  letter-spacing: 0.2px;
}
.slide-title {
  font-size: 2.2rem;
  margin-bottom: 0.3rem;
}
.slide-sub {
  color: var(--muted);
  font-size: 1rem;
}
.card {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.04);
}
.pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--panel-border);
  color: var(--muted);
  font-size: 0.8rem;
  margin-right: 6px;
}
.metric-big {
  font-size: 2rem;
  font-weight: 700;
  color: var(--accent);
}
.muted {
  color: var(--muted);
}
.accent {
  color: var(--accent);
}
.accent-2 {
  color: var(--accent-2);
}
</style>
""",
    unsafe_allow_html=True,
)


PROJECT_STATS = {
    "rows_2010_2024": 91_000_000,
    "cols_total": 40,
    "rows_2022": 9_479_544,
    "dups_removed": 533_390,
    "cols_removed": 19,
    "cols_kept": 21,
}

MISSINGNESS = pd.DataFrame(
    {
        "indicator": ["Ewltp (g/km)", "Enedc (g/km)"],
        "missing_rate": [0.01, 0.82],
    }
)

REG_METRICS = pd.DataFrame(
    {
        "model": ["Random Forest", "Linear Regression"],
        "r2_test": [0.9938, 0.9358],
        "rmse": [5.41, 17.45],
        "mae": [2.44, 12.86],
    }
)

CLF_METRICS = pd.DataFrame(
    {
        "model": [
            "XGBoost (tuned)",
            "Bagging",
            "Random Forest",
            "Logistic Regression",
        ],
        "accuracy_test": [0.93, 0.9217, 0.9213, 0.7629],
        "f1_weighted": [0.9304, 0.9288, 0.9288, 0.7696],
    }
)

CLASS_THRESHOLDS = [
    ("0", "Zero emission", 0),
    ("1", "A", 100),
    ("2", "B", 120),
    ("3", "C", 140),
    ("4", "D", 160),
    ("5", "E", 200),
    ("6", "F", 250),
    ("7", "G", 9999),
]

GAIN_IMPORTANCE = pd.DataFrame(
    {
        "feature": [
            "Fuel type / energy mode",
            "Hybrid tech flags",
            "Mass (kg)",
            "Engine displacement (cm3)",
            "Power (kW)",
            "Dimensions (W, At1, At2)",
        ],
        "importance": [1.0, 0.85, 0.55, 0.48, 0.44, 0.31],
    }
)

PERM_IMPORTANCE = pd.DataFrame(
    {
        "feature": [
            "Mass (kg)",
            "Engine displacement (cm3)",
            "Power (kW)",
            "Fuel type / energy mode",
            "Hybrid tech flags",
            "Dimensions (W, At1, At2)",
        ],
        "importance": [1.0, 0.86, 0.78, 0.52, 0.41, 0.29],
    }
)


@st.cache_data
def load_data_dictionary() -> pd.DataFrame:
    path = Path("references/Dictionnaires_de_donnees.xlsx")
    df_raw = pd.read_excel(path, sheet_name="Feuil1", header=None)
    header_idx = 1
    header = df_raw.iloc[header_idx].tolist()
    df = df_raw.iloc[header_idx + 1 :].copy()
    df.columns = header
    df = df.dropna(how="all")
    col_name = df.columns[1]
    col_drop = df.columns[3]
    col_na = df.columns[5]
    df = df[[col_name, col_drop, col_na]].copy()
    df = df.rename(
        columns={
            col_name: "column",
            col_drop: "drop_reason",
            col_na: "na_pct",
        }
    )
    return df


def classify_co2(value: float) -> str:
    if value == 0:
        return "0"
    if value <= 100:
        return "1"
    if value <= 120:
        return "2"
    if value <= 140:
        return "3"
    if value <= 160:
        return "4"
    if value <= 200:
        return "5"
    if value <= 250:
        return "6"
    return "7"


def heuristic_co2_estimate(payload: dict) -> float:
    # Simple proxy model to keep the demo interactive when no trained model is available.
    mass = payload["mass_kg"]
    power = payload["power_kw"]
    disp = payload["disp_cm3"]
    width = payload["width_mm"]
    track = (payload["track_front_mm"] + payload["track_rear_mm"]) / 2
    fuel = payload["fuel_type"]

    base = 0.03 * mass + 0.09 * power + 0.015 * disp
    base += 0.01 * max(width - 1650, 0)
    base += 0.008 * max(track - 1500, 0)

    fuel_adj = {
        "EV": -base,
        "PHEV": -30,
        "HEV": -20,
        "PETROL": 30,
        "DIESEL": 40,
        "OTHER": 10,
    }
    return max(0, base + fuel_adj.get(fuel, 0))


def load_model(uploaded: io.BytesIO | None):
    model_paths = [
        Path("models/xgb_pipeline.joblib"),
        Path("models/xgb_classifier.joblib"),
        Path("models/xgb_classifier.pkl"),
    ]

    if uploaded is not None:
        return joblib_load(uploaded), "uploaded"

    for path in model_paths:
        if path.exists():
            return joblib_load(path), str(path)
    return None, None


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"<div class='slide-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='slide-sub'>{subtitle}</div>", unsafe_allow_html=True)


def chart_volumetry():
    df = pd.DataFrame(
        {
            "scope": ["2010-2024", "Focus 2022"],
            "rows": [PROJECT_STATS["rows_2010_2024"], PROJECT_STATS["rows_2022"]],
        }
    )
    fig = px.bar(
        df,
        x="scope",
        y="rows",
        text="rows",
        color="scope",
        color_discrete_sequence=["#0f4c5c", "#e07a5f"],
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(
        height=360,
        showlegend=False,
        yaxis_title="Rows",
        xaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def chart_missingness():
    fig = px.bar(
        MISSINGNESS,
        x="indicator",
        y="missing_rate",
        text="missing_rate",
        color="indicator",
        color_discrete_sequence=["#0f4c5c", "#e07a5f"],
    )
    fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
    fig.update_layout(
        height=360,
        showlegend=False,
        yaxis_tickformat=".0%",
        yaxis_title="Missing rate",
        xaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def chart_column_split():
    df = pd.DataFrame(
        {
            "category": ["Kept", "Removed"],
            "count": [PROJECT_STATS["cols_kept"], PROJECT_STATS["cols_removed"]],
        }
    )
    fig = px.pie(
        df,
        names="category",
        values="count",
        color="category",
        color_discrete_sequence=["#0f4c5c", "#e07a5f"],
        hole=0.5,
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def chart_regression():
    fig = px.bar(
        REG_METRICS,
        x="model",
        y="r2_test",
        color="model",
        color_discrete_sequence=["#0f4c5c", "#e07a5f"],
        text="r2_test",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        height=320,
        showlegend=False,
        yaxis_range=[0.8, 1.02],
        yaxis_title="R2 (test)",
        xaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def chart_regression_rmse():
    fig = px.bar(
        REG_METRICS,
        x="model",
        y="rmse",
        color="model",
        color_discrete_sequence=["#0f4c5c", "#e07a5f"],
        text="rmse",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        height=320,
        showlegend=False,
        yaxis_title="RMSE (g/km)",
        xaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def chart_classification():
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=CLF_METRICS["model"],
            y=CLF_METRICS["f1_weighted"],
            name="F1 weighted",
            marker_color="#0f4c5c",
            text=[f"{v:.3f}" for v in CLF_METRICS["f1_weighted"]],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=CLF_METRICS["model"],
            y=CLF_METRICS["accuracy_test"],
            name="Accuracy",
            marker_color="#e07a5f",
            text=[f"{v:.3f}" for v in CLF_METRICS["accuracy_test"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=360,
        barmode="group",
        yaxis_range=[0.6, 1.02],
        yaxis_title="Score (test)",
        xaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def chart_thresholds():
    df = pd.DataFrame(CLASS_THRESHOLDS, columns=["class", "label", "upper"])
    fig = px.line(
        df,
        x="class",
        y="upper",
        markers=True,
        text="upper",
        color_discrete_sequence=["#0f4c5c"],
    )
    fig.update_traces(texttemplate="%{text}", textposition="top center")
    fig.update_layout(
        height=320,
        yaxis_title="Upper threshold (g/km)",
        xaxis_title="Class (0-7)",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def chart_importance(df: pd.DataFrame, title: str):
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        color="feature",
        color_discrete_sequence=["#0f4c5c"] * len(df),
    )
    fig.update_layout(
        height=360,
        showlegend=False,
        xaxis_title="Relative importance (normalized)",
        yaxis_title="",
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


SLIDES = [
    ("Intro", "intro"),
    ("1 - Contexte & donnees", "data"),
    ("2 - Qualite & couverture", "quality"),
    ("3 - Choix metodologiques", "pipeline"),
    ("4 - Regression (Rendu 2)", "regression"),
    ("5 - Classification & seuils", "classification"),
    ("6 - XGBoost & Optuna", "xgb"),
    ("7 - Explicabilite (SHAP)", "shap"),
    ("Demo live", "demo"),
]


with st.sidebar:
    st.markdown("### Navigation")
    slide_labels = [s[0] for s in SLIDES]
    slide_key = st.radio("Slide", slide_labels, index=0, label_visibility="collapsed")
    slide_idx = slide_labels.index(slide_key)
    st.progress((slide_idx + 1) / len(SLIDES))
    st.markdown(
        f"<div class='muted'>Slide {slide_idx + 1} / {len(SLIDES)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Timing guide")
    st.markdown("- Intro: 2 min")
    st.markdown("- Slides 1-7: ~2 min chacune")
    st.markdown("- Demo live: 5 min")


if slide_key == "Intro":
    section_header("Projet CO2", "Presentation synthese en 20 minutes")
    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        st.markdown(
            """
<div class="card">
<div class="metric-big">Pourquoi ce projet ?</div>
<div class="muted">
Anticiper la trajectoire CO2 des vehicules particuliers en Europe pour
outiller la regulation, les constructeurs et les acteurs publics.
</div>
<br/>
<div class="pill">Rendu 1: exploration + preprocessing</div>
<div class="pill">Rendu 2: modelisation + explicabilite</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
<div class="card">
<div class="metric-big">Objectif</div>
<div class="muted">
1) Comprendre les drivers CO2.
<br/>2) Predire une valeur continue (regression).
<br/>3) Classer en 8 niveaux interpretable (0-7).
</div>
</div>
""",
            unsafe_allow_html=True,
        )


elif slide_key == "1 - Contexte & donnees":
    section_header("Donnees & volumetrie", "EEA CO2 cars dataset 2010-2024")
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.plotly_chart(chart_volumetry(), use_container_width=True)
    with col_b:
        st.markdown(
            """
<div class="card">
<div class="metric-big">>91M</div>
<div class="muted">Lignes sur 2010-2024, 40 colonnes</div>
<br/>
<div class="metric-big">9.48M</div>
<div class="muted">Focus 2022 (volumetrie exploitable)</div>
<br/>
<div class="metric-big">533k</div>
<div class="muted">Doublons stricts supprimes</div>
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div class='muted'>Source: Rendu 1. Le focus 2022 combine volume et meilleure couverture WLTP.</div>",
        unsafe_allow_html=True,
    )


elif slide_key == "2 - Qualite & couverture":
    section_header("Qualite des donnees", "WLTP vs NEDC")
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.plotly_chart(chart_missingness(), use_container_width=True)
    with col_b:
        st.markdown(
            """
<div class="card">
<div class="metric-big">>80%</div>
<div class="muted">De valeurs manquantes sur Enedc (ancienne norme)</div>
<br/>
<div class="metric-big">&lt;1%</div>
<div class="muted">De valeurs manquantes sur Ewltp (norme actuelle)</div>
<br/>
<div class="muted">Decision: prioriser Ewltp pour la cible.</div>
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div class='muted'>La couverture WLTP rend la prediction plus robuste et interpretable.</div>",
        unsafe_allow_html=True,
    )


elif slide_key == "3 - Choix metodologiques":
    section_header("Choix cles", "Eviter les fuites et stabiliser la base")
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.plotly_chart(chart_column_split(), use_container_width=True)
    with col_b:
        st.markdown(
            """
<div class="card">
<div class="metric-big">Pre-split</div>
<div class="muted">Split train/test AVANT le preprocessing pour eviter les fuites.</div>
<br/>
<div class="metric-big">Selection</div>
<div class="muted">19 colonnes retirees (admin, obsolete, forte NA).</div>
<br/>
<div class="metric-big">Structure</div>
<div class="muted">Focus sur les variables physiques/energetiques.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with st.expander("Voir le dictionnaire de donnees (extrait)"):
        df_dict = load_data_dictionary()
        df_drop = df_dict[df_dict["drop_reason"].notna()].copy()
        df_drop = df_drop.rename(columns={"column": "column", "drop_reason": "reason"})
        st.dataframe(df_drop.head(15), use_container_width=True, height=300)


elif slide_key == "4 - Regression (Rendu 2)":
    section_header("Regression", "Predire une valeur continue CO2 (g/km)")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_regression(), use_container_width=True)
    with col_b:
        st.plotly_chart(chart_regression_rmse(), use_container_width=True)

    st.markdown(
        """
<div class="card">
<div class="metric-big">Random Forest</div>
<div class="muted">R2 test ~0.994, RMSE ~5.41 g/km. Performance elevee grace aux variables structurelles.</div>
</div>
""",
        unsafe_allow_html=True,
    )


elif slide_key == "5 - Classification & seuils":
    section_header("Classification", "8 classes de 0 a 7 (directive 1999/94/EC)")
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.plotly_chart(chart_thresholds(), use_container_width=True)
    with col_b:
        st.markdown(
            """
<div class="card">
<div class="metric-big">Classe 0</div>
<div class="muted">Zero emission (EV)</div>
<br/>
<div class="metric-big">Classes 1-7</div>
<div class="muted">A a G selon les seuils WLTP</div>
<br/>
<div class="muted">Objectif: sortie interpretable metier.</div>
</div>
""",
            unsafe_allow_html=True,
        )


elif slide_key == "6 - XGBoost & Optuna":
    section_header("Selection du modele", "F1 pondere comme metrique principale")
    st.plotly_chart(chart_classification(), use_container_width=True)
    st.markdown(
        """
<div class="card">
<div class="metric-big">F1 test ~0.930</div>
<div class="muted">XGBoost optimise (Optuna) = meilleur compromis performance / robustesse.</div>
<br/>
<div class="muted">Les erreurs se concentrent entre classes adjacentes (2↔3, 4↔5).</div>
</div>
""",
        unsafe_allow_html=True,
    )


elif slide_key == "7 - Explicabilite (SHAP)":
    section_header("Explicabilite", "SHAP confirme la logique metier")
    tabs = st.tabs(["Importance gain", "Importance permutation"])
    with tabs[0]:
        st.plotly_chart(
            chart_importance(GAIN_IMPORTANCE, "Gain importance (XGBoost)"),
            use_container_width=True,
        )
    with tabs[1]:
        st.plotly_chart(
            chart_importance(PERM_IMPORTANCE, "Permutation importance (F1 weighted)"),
            use_container_width=True,
        )
    st.markdown(
        """
<div class="card">
<div class="metric-big">Lecture cle</div>
<div class="muted">
Carburant segmente les classes, puis les variables physiques (masse, cylindree, puissance)
affinent la decision. Les classes extremes sont quasi parfaites.
</div>
</div>
""",
        unsafe_allow_html=True,
    )


elif slide_key == "Demo live":
    section_header("Demo live", "Predire la classe CO2 (0-7)")
    st.markdown(
        "<div class='muted'>Charge un modele pipeline ou utilise le mode demo.</div>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        with st.form("demo_form"):
            mass_kg = st.number_input("Mass (kg)", 600, 4000, 1500, step=10)
            power_kw = st.number_input("Power (kW)", 40, 500, 110, step=5)
            disp_cm3 = st.number_input("Displacement (cm3)", 800, 6000, 1600, step=50)
            width_mm = st.number_input("Width (mm)", 1500, 2300, 1800, step=10)
            track_front_mm = st.number_input("Track front (mm)", 1200, 2000, 1550, step=10)
            track_rear_mm = st.number_input("Track rear (mm)", 1200, 2000, 1550, step=10)
            year = st.number_input("Model year", 2010, 2025, 2022, step=1)
            fuel_type = st.selectbox(
                "Fuel type (Ft)",
                ["PETROL", "DIESEL", "HEV", "PHEV", "EV", "OTHER"],
            )
            uploaded = st.file_uploader("Optional: upload model pipeline (.joblib/.pkl)")
            use_demo = st.checkbox("Use demo mode (heuristic)", value=True)
            submitted = st.form_submit_button("Predict class")

    with col_right:
        if submitted:
            payload = {
                "mass_kg": mass_kg,
                "power_kw": power_kw,
                "disp_cm3": disp_cm3,
                "width_mm": width_mm,
                "track_front_mm": track_front_mm,
                "track_rear_mm": track_rear_mm,
                "year": year,
                "fuel_type": fuel_type,
            }

            model, model_src = load_model(uploaded)
            pred_class = None
            pred_co2 = None

            if model is not None and not use_demo:
                df_in = pd.DataFrame([payload])
                try:
                    pred = model.predict(df_in)
                    pred_class = str(pred[0])
                except Exception:
                    pred_class = None

            if pred_class is None:
                pred_co2 = heuristic_co2_estimate(payload)
                pred_class = classify_co2(pred_co2)

            class_map = {c: label for c, label, _ in CLASS_THRESHOLDS}
            label = class_map.get(pred_class, "Unknown")

            st.markdown(
                """
<div class="card">
<div class="metric-big">Prediction</div>
<div class="muted">Classe CO2</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.metric("Class", f"{pred_class} ({label})")
            if pred_co2 is not None:
                st.metric("Estimated CO2 (demo)", f"{pred_co2:.1f} g/km")
            if model_src:
                st.caption(f"Model source: {model_src}")
            else:
                st.caption("No model file found. Demo mode used.")
