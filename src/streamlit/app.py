
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
    page_title="Projet CO2 - Présentation",
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

ASSET_DIR = Path("assets")


def asset_path(stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = ASSET_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return ASSET_DIR / f"{stem}.png"


INTRO_IMAGES = [
    asset_path("intro_green_car"),
    asset_path("intro_traffic"),
]
STRUCTURE_IMAGE = asset_path("columns_struct")
SHAP_PANELS = [
    {
        "title": "SHAP summary (dot plot)",
        "path": asset_path("shap_summary"),
        "desc": (
            "Vue globale de l'impact des variables : la couleur indique la valeur "
            "de la variable et la position l'effet sur la prédiction."
        ),
    },
    {
        "title": "SHAP bar (importance globale)",
        "path": asset_path("shap_bar"),
        "desc": "Classement global des variables les plus influentes sur la décision.",
    },
    {
        "title": "SHAP dependence",
        "path": asset_path("shap_dependence"),
        "desc": (
            "Relation locale entre une variable clé (ex. masse ou puissance) et "
            "son impact sur la sortie."
        ),
    },
    {
        "title": "SHAP force / decision",
        "path": asset_path("shap_force"),
        "desc": (
            "Lecture instance par instance : contribution positive/négative des "
            "variables à la classe prédite."
        ),
    },
]

MISSINGNESS = pd.DataFrame(
    {
        "indicator": ["Ewltp (g/km)", "Enedc (g/km)"],
        "missing_rate": [0.01, 0.82],
    }
)

REG_RESULTS_FULL = pd.DataFrame(
    [
        {
            "Modèle": "Random Forest",
            "R² CV": 0.9949226,
            "R² train": 0.998012,
            "R² test": 0.993935,
            "MAE": 2.426078,
            "RMSE": 5.362198,
        },
        {
            "Modèle": "KNN",
            "R² CV": 0.9898879,
            "R² train": 0.994041,
            "R² test": 0.989878,
            "MAE": 3.294370,
            "RMSE": 6.927429,
        },
        {
            "Modèle": "Ridge",
            "R² CV": 0.9344312,
            "R² train": 0.934468,
            "R² test": 0.935798,
            "MAE": 12.855034,
            "RMSE": 17.446780,
        },
        {
            "Modèle": "Linear Regression",
            "R² CV": -2.0018e21,
            "R² train": 0.934468,
            "R² test": 0.935797,
            "MAE": 12.855326,
            "RMSE": 17.446926,
        },
        {
            "Modèle": "Lasso",
            "R² CV": 0.9266005,
            "R² train": 0.926648,
            "R² test": 0.927446,
            "MAE": 13.687712,
            "RMSE": 18.546881,
        },
        {
            "Modèle": "Elastic Net",
            "R² CV": 0.8777564,
            "R² train": 0.877803,
            "R² test": 0.877820,
            "MAE": 17.637387,
            "RMSE": 24.068001,
        },
    ]
)

REG_RESULTS_REDUCED = pd.DataFrame(
    [
        {
            "Modèle": "Random Forest",
            "R² CV": 0.994104,
            "R² train": 0.997681,
            "R² test": 0.992326,
            "MAE": 2.524182,
            "RMSE": 6.031930,
        },
        {
            "Modèle": "KNN",
            "R² CV": 0.986014,
            "R² train": 0.992072,
            "R² test": 0.985833,
            "MAE": 3.320510,
            "RMSE": 8.195481,
        },
        {
            "Modèle": "Linear Regression",
            "R² CV": 0.792661,
            "R² train": 0.792732,
            "R² test": 0.792315,
            "MAE": 22.646579,
            "RMSE": 31.379221,
        },
        {
            "Modèle": "Ridge",
            "R² CV": 0.792661,
            "R² train": 0.792732,
            "R² test": 0.792315,
            "MAE": 22.646586,
            "RMSE": 31.379221,
        },
        {
            "Modèle": "Lasso",
            "R² CV": 0.785689,
            "R² train": 0.785729,
            "R² test": 0.785227,
            "MAE": 23.006620,
            "RMSE": 31.910176,
        },
        {
            "Modèle": "Elastic Net",
            "R² CV": 0.722661,
            "R² train": 0.722703,
            "R² test": 0.722426,
            "MAE": 27.506783,
            "RMSE": 36.276787,
        },
    ]
)

CLF_METRICS = pd.DataFrame(
    {
        "model": [
            "XGBoost",
            "Bagging",
            "Random Forest",
            "Logistic Regression",
        ],
        "accuracy_test": [0.92, 0.9217, 0.9213, 0.7629],
        "f1_weighted": [0.92, 0.9288, 0.9288, 0.7696],
    }
)

XGB_OPTUNA_SCORES = pd.DataFrame(
    {
        "Version": ["Avant optimisation", "Après Optuna + seuils"],
        "Exactitude": [0.92, 0.93],
        "F1 pondéré": [0.92, 0.93],
    }
)

CLASS_BINS = [
    {"class": "0", "label": "Zéro émission", "min": 0, "max": 0, "example": "Tesla Model 3"},
    {"class": "1", "label": "A (≤100)", "min": 1, "max": 100, "example": "Honda hybride"},
    {"class": "2", "label": "B (≤120)", "min": 101, "max": 120, "example": "Peugeot 208 essence"},
    {"class": "3", "label": "C (≤140)", "min": 121, "max": 140, "example": "Renault Clio diesel"},
    {"class": "4", "label": "D (≤160)", "min": 141, "max": 160, "example": "VW Golf GTI"},
    {"class": "5", "label": "E (≤200)", "min": 161, "max": 200, "example": "SUV essence (3008)"},
    {"class": "6", "label": "F (≤250)", "min": 201, "max": 250, "example": "Camping-car / pickup"},
    {"class": "7", "label": "G (>250)", "min": 251, "max": 9999, "example": "Voiture de sport"},
]

STRUCTURAL_COLUMNS = [
    "Country",
    "Man",
    "Mk",
    "Va",
    "Ve",
    "Cn",
    "Cr",
    "m (kg)",
    "W (mm)",
    "At2 (mm)",
    "cylindre_du_moteur_cm3",
    "puissance_du_moteur_kw",
    "IT",
]

EXCLUDED_COLUMNS = [
    "Fuel consumption (l/100km)",
    "Electric range (km)",
    "z (Wh/km)",
    "Enedc (g/km)",
    "Ernedc (g/km)",
    "Erwltp (g/km)",
]

PREPROCESS_NUMERIC = [
    "Imputation par médiane ou moyenne groupée (ex. par constructeur).",
    "Création d'indicateurs de valeurs manquantes pour certaines variables.",
    "Standardisation (StandardScaler) après split train/test.",
]

PREPROCESS_CATEGORICAL = [
    "Nettoyage des faux NA et harmonisation des libellés (constructeurs, modèles).",
    "Imputation par mode groupé (ex. Man par Mk ou Cn).",
    "Encodage One-Hot (drop='first') sur catégories à faible cardinalité.",
]

CONFUSION_MATRIX_OPT = np.array(
    [
        [11665, 0, 0, 0, 0, 0, 0, 0],
        [0, 10055, 35, 4, 0, 2, 2, 0],
        [0, 19, 5821, 1036, 14, 0, 1, 0],
        [0, 2, 934, 23382, 1219, 4, 0, 0],
        [0, 3, 2, 1568, 18130, 742, 3, 3],
        [0, 8, 1, 12, 1284, 16857, 402, 28],
        [0, 4, 2, 1, 9, 506, 9185, 104],
        [0, 6, 0, 0, 0, 5, 143, 3004],
    ]
)

GAIN_IMPORTANCE = pd.DataFrame(
    {
        "feature": [
            "Type d'énergie / carburant",
            "Hybridation",
            "Masse (kg)",
            "Cylindrée (cm3)",
            "Puissance (kW)",
            "Dimensions (W, At1, At2)",
        ],
        "importance": [1.0, 0.85, 0.55, 0.48, 0.44, 0.31],
    }
)

PERM_IMPORTANCE = pd.DataFrame(
    {
        "feature": [
            "Masse (kg)",
            "Cylindrée (cm3)",
            "Puissance (kW)",
            "Type d'énergie / carburant",
            "Hybridation",
            "Dimensions (W, At1, At2)",
        ],
        "importance": [1.0, 0.86, 0.78, 0.52, 0.41, 0.29],
    }
)


@st.cache_data
def load_data_dictionary() -> pd.DataFrame:
    path = Path("references/Dictionnaires_de_donnees.xlsx")
    if not path.exists():
        return pd.DataFrame(columns=["column", "drop_reason", "na_pct"])
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
    for row in CLASS_BINS:
        if row["min"] <= value <= row["max"]:
            return row["class"]
    return "7"


def heuristic_co2_estimate(payload: dict) -> float:
    # Simple proxy model to keep the demo interactive when no trained model is available.
    mass = payload["m (kg)"]
    power = payload["puissance_du_moteur_kw"]
    disp = payload["cylindre_du_moteur_cm3"]
    width = payload["W (mm)"]
    track = (payload["At1 (mm)"] + payload["At2 (mm)"]) / 2
    fuel = payload["Ft"]

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


def show_image(path: Path, caption: str | None = None) -> None:
    if not path.exists():
        return
    st.image(str(path), caption=caption, width=700)


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
        yaxis_title="Lignes",
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
        yaxis_title="Taux de NA",
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
    df = REG_RESULTS_FULL.sort_values("R² test", ascending=False)
    fig = px.bar(
        df,
        x="Modèle",
        y="R² test",
        color="Modèle",
        color_discrete_sequence=["#0f4c5c", "#e07a5f"],
        text="R² test",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        height=320,
        showlegend=False,
        yaxis_range=[0.8, 1.02],
        yaxis_title="R² (test)",
        xaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def chart_regression_rmse():
    df = REG_RESULTS_FULL.sort_values("RMSE", ascending=True)
    fig = px.bar(
        df,
        x="Modèle",
        y="RMSE",
        color="Modèle",
        color_discrete_sequence=["#0f4c5c", "#e07a5f"],
        text="RMSE",
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
            name="F1 pondéré",
            marker_color="#0f4c5c",
            text=[f"{v:.3f}" for v in CLF_METRICS["f1_weighted"]],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=CLF_METRICS["model"],
            y=CLF_METRICS["accuracy_test"],
            name="Exactitude",
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
    df = pd.DataFrame(CLASS_BINS)
    df["display_min"] = df["min"].clip(upper=300)
    df["display_max"] = df["max"].clip(upper=300)
    fig = go.Figure(
        go.Bar(
            x=df["display_max"],
            y=df["class"],
            base=df["display_min"],
            orientation="h",
            marker_color="#0f4c5c",
            text=[f"{row['min']}–{row['max']}" for _, row in df.iterrows()],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=320,
        yaxis_title="Classe (0–7)",
        xaxis_title="Seuils CO2 (g/km)",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def chart_xgb_scores():
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=XGB_OPTUNA_SCORES["Version"],
            y=XGB_OPTUNA_SCORES["F1 pondéré"],
            name="F1 pondéré",
            marker_color="#0f4c5c",
            text=[f"{v:.2f}" for v in XGB_OPTUNA_SCORES["F1 pondéré"]],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=XGB_OPTUNA_SCORES["Version"],
            y=XGB_OPTUNA_SCORES["Exactitude"],
            name="Exactitude",
            marker_color="#e07a5f",
            text=[f"{v:.2f}" for v in XGB_OPTUNA_SCORES["Exactitude"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=320,
        barmode="group",
        yaxis_range=[0.6, 1.02],
        yaxis_title="Score (test)",
        xaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def chart_confusion_matrix():
    labels = [row["class"] for row in CLASS_BINS]
    fig = go.Figure(
        data=go.Heatmap(
            z=CONFUSION_MATRIX_OPT,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        height=420,
        xaxis_title="Prédit",
        yaxis_title="Réel",
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
        xaxis_title="Importance relative (normalisée)",
        yaxis_title="",
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


SLIDES = [
    {"label": "Introduction", "id": "intro"},
    {"label": "Contexte & qualité des données", "id": "context"},
    {"label": "Pré-processing & sélection", "id": "preprocess"},
    {"label": "Régression", "id": "regression"},
    {"label": "Classification & optimisation", "id": "classification"},
    {"label": "Explicabilité (SHAP)", "id": "shap"},
    {"label": "Démo live", "id": "demo"},
]


with st.sidebar:
    st.markdown("### Navigation")
    slide_labels = [s["label"] for s in SLIDES]
    slide_label = st.radio("Slide", slide_labels, index=0, label_visibility="collapsed")
    slide_id = next(s["id"] for s in SLIDES if s["label"] == slide_label)
    slide_idx = slide_labels.index(slide_label)
    st.progress((slide_idx + 1) / len(SLIDES))
    st.markdown(
        f"<div class='muted'>Slide {slide_idx + 1} / {len(SLIDES)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Timing guide")
    st.markdown("- Introduction : 2 min")
    st.markdown("- Contexte & qualité : ~3 min")
    st.markdown("- Pré-processing : ~3 min")
    st.markdown("- Régression : ~2 min")
    st.markdown("- Classification & optimisation : ~3 min")
    st.markdown("- Explicabilité (SHAP) : ~2 min")
    st.markdown("- Démo live : 5 min")


if slide_id == "intro":
    section_header("Introduction", "Pourquoi suivre les émissions de CO2 des véhicules particuliers")
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown(
            """
<div class="card">
<div class="metric-big">Intérêt scientifique & métier</div>
<div class="muted">
Les véhicules particuliers représentent une part majeure des émissions de CO2 du transport.
Comprendre les déterminants techniques permet d'objectiver les trajectoires de réduction,
d'éclairer les politiques publiques et d'aider les constructeurs à concevoir des modèles plus sobres.
</div>
<br/>
<div class="metric-big">Objectifs</div>
<div class="muted">
<ul>
  <li>Identifier les véhicules les plus polluants.</li>
  <li>Comprendre les caractéristiques techniques qui influencent ces émissions.</li>
  <li>Prédire l'émission CO2 à partir des caractéristiques physiques.</li>
</ul>
</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with col_b:
        show_image(INTRO_IMAGES[0], "Contexte : transition et sobriété")
        show_image(INTRO_IMAGES[1], "Pression urbaine et pollution locale")


elif slide_id == "context":
    section_header("Contexte & qualité des données", "EEA CO2 cars dataset 2010–2024")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_volumetry(), use_container_width=True)
    with col_b:
        st.plotly_chart(chart_missingness(), use_container_width=True)
    st.markdown(
        """
<div class="card">
<div class="metric-big">>91M</div>
<div class="muted">Lignes sur 2010–2024, 40 colonnes</div>
<br/>
<div class="metric-big">9,48M</div>
<div class="muted">Focus 2022 (volume exploitable)</div>
<br/>
<div class="metric-big">Ewltp ≈ 1% NA</div>
<div class="muted">Enedc ≈ 82% NA → cible WLTP priorisée</div>
</div>
""",
        unsafe_allow_html=True,
    )


elif slide_id == "preprocess":
    section_header("Pré-processing & sélection", "Split avant traitement pour éviter toute fuite")
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown(
            """
<div class="card">
<div class="metric-big">Méthodologie</div>
<div class="muted">
<ul>
  <li>Split train/test avant tout traitement.</li>
  <li>Traitement séparé numériques vs qualitatives.</li>
  <li>Exclusion des colonnes trop corrélées à la cible (risque de fuite).</li>
</ul>
</div>
</div>
""",
            unsafe_allow_html=True,
        )
        num_col, cat_col = st.columns(2)
        with num_col:
            st.markdown(
                "<div class='card'><div class='metric-big'>Numériques</div>"
                + "<div class='muted'>" + "<br/>".join(PREPROCESS_NUMERIC) + "</div></div>",
                unsafe_allow_html=True,
            )
        with cat_col:
            st.markdown(
                "<div class='card'><div class='metric-big'>Qualitatives</div>"
                + "<div class='muted'>" + "<br/>".join(PREPROCESS_CATEGORICAL) + "</div></div>",
                unsafe_allow_html=True,
            )
        with st.expander("Dictionnaire de données (extrait)"):
            df_dict = load_data_dictionary()
            df_drop = df_dict[df_dict["drop_reason"].notna()].copy()
            df_drop = df_drop.rename(columns={"column": "column", "drop_reason": "reason"})
            st.dataframe(df_drop.head(15), use_container_width=True, height=300)
    with col_b:
        st.markdown(
            """
<div class="card">
<div class="metric-big">Colonnes structurelles conservées</div>
<div class="muted">Uniquement les variables disponibles avant la construction du véhicule.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.code("\n".join(STRUCTURAL_COLUMNS))
        st.markdown("<div class='muted'>At1 supprimée, At2 conservée.</div>", unsafe_allow_html=True)
        show_image(STRUCTURE_IMAGE, "Synthèse des colonnes structurelles")
        st.markdown(
            "<div class='card'><div class='metric-big'>Colonnes exclues (fuite)</div>"
            + "<div class='muted'>" + "<br/>".join(EXCLUDED_COLUMNS) + "</div></div>",
            unsafe_allow_html=True,
        )


elif slide_id == "regression":
    section_header("Régression", "Comparaison des modèles testés (CO2 en g/km)")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_regression(), use_container_width=True)
    with col_b:
        st.plotly_chart(chart_regression_rmse(), use_container_width=True)
    st.markdown(
        """
<div class="card">
<div class="metric-big">Random Forest</div>
<div class="muted">Meilleure performance globale, RMSE ~5–6 g/km.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.dataframe(REG_RESULTS_FULL, use_container_width=True, height=280)
    with st.expander("Résultats sur données réduites (variables structurelles uniquement)"):
        st.dataframe(REG_RESULTS_REDUCED, use_container_width=True, height=260)


elif slide_id == "classification":
    section_header("Classification & optimisation", "Classes 0–7 et optimisation XGBoost")
    tabs = st.tabs(["Catégorisation", "Comparatif modèles", "XGBoost + Optuna"])
    with tabs[0]:
        col_a, col_b = st.columns([1.1, 1])
        with col_a:
            st.plotly_chart(chart_thresholds(), use_container_width=True)
        with col_b:
            df_bins = pd.DataFrame(CLASS_BINS)
            df_bins["Seuil (g/km)"] = df_bins.apply(
                lambda row: "0" if row["class"] == "0" else f"{row['min']}–{row['max']}",
                axis=1,
            )
            df_bins = df_bins.rename(
                columns={
                    "class": "Classe",
                    "label": "Libellé",
                    "example": "Exemple",
                }
            )[["Classe", "Libellé", "Seuil (g/km)", "Exemple"]]
            st.dataframe(df_bins, use_container_width=True, height=320)
        st.markdown(
            "<div class='muted'>Ajout d'une classe 0 pour les véhicules 100% électriques.</div>",
            unsafe_allow_html=True,
        )
    with tabs[1]:
        st.plotly_chart(chart_classification(), use_container_width=True)
        st.markdown(
            "<div class='muted'>Les meilleurs modèles se concentrent sur des erreurs entre classes adjacentes.</div>",
            unsafe_allow_html=True,
        )
    with tabs[2]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(chart_xgb_scores(), use_container_width=True)
        with col_b:
            st.plotly_chart(chart_confusion_matrix(), use_container_width=True)
        st.markdown(
            "<div class='muted'>Matrice de confusion après optimisation des seuils.</div>",
            unsafe_allow_html=True,
        )


elif slide_id == "shap":
    section_header("Explicabilité (SHAP)", "Lecture globale et locale du modèle")
    cols = st.columns(2)
    for idx, panel in enumerate(SHAP_PANELS):
        with cols[idx % 2]:
            show_image(panel["path"], panel["title"])
            st.markdown(f"<div class='muted'>{panel['desc']}</div>", unsafe_allow_html=True)
    st.markdown(
        """
<div class="card">
<div class="metric-big">Lecture clé</div>
<div class="muted">
Le carburant segmente les classes, puis les variables physiques (masse, cylindrée, puissance)
affinent la décision. Les classes extrêmes sont quasi parfaites.
</div>
</div>
""",
        unsafe_allow_html=True,
    )


elif slide_id == "demo":
    section_header("Démo live", "Prédire la classe CO2 (0–7)")
    st.markdown(
        "<div class='muted'>Charge un modèle pipeline ou utilise le mode démo.</div>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        with st.form("demo_form"):
            mass_kg = st.number_input("Masse (kg)", 600, 4000, 1500, step=10)
            cyl_cm3 = st.number_input("cylindre_du_moteur_cm3", 800, 6000, 1600, step=50)
            power_kw = st.number_input("puissance_du_moteur_kw", 40, 500, 110, step=5)
            width_mm = st.number_input("W (mm)", 1500, 2300, 1800, step=10)
            at1_mm = st.number_input("At1 (mm)", 1200, 2000, 1550, step=10)
            at2_mm = st.number_input("At2 (mm)", 1200, 2000, 1550, step=10)
            it_flag = st.selectbox("IT (0/1)", [0, 1])
            year = st.number_input("Année", 2010, 2025, 2022, step=1)
            fuel_type = st.selectbox(
                "Ft (carburant)",
                ["PETROL", "DIESEL", "HEV", "PHEV", "EV", "OTHER"],
            )
            uploaded = st.file_uploader("Optionnel : charger un pipeline (.joblib/.pkl)")
            use_demo = st.checkbox("Utiliser le mode démo (heuristique)", value=True)
            submitted = st.form_submit_button("Prédire la classe")

    with col_right:
        if submitted:
            payload = {
                "m (kg)": mass_kg,
                "cylindre_du_moteur_cm3": cyl_cm3,
                "puissance_du_moteur_kw": power_kw,
                "W (mm)": width_mm,
                "At1 (mm)": at1_mm,
                "At2 (mm)": at2_mm,
                "IT": it_flag,
                "year": year,
                "Ft": fuel_type,
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

            class_map = {row["class"]: row["label"] for row in CLASS_BINS}
            label = class_map.get(pred_class, "Inconnu")

            st.markdown(
                """
<div class="card">
<div class="metric-big">Prédiction</div>
<div class="muted">Classe CO2</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.metric("Classe", f"{pred_class} ({label})")
            if pred_co2 is not None:
                st.metric("CO2 estimé (démo)", f"{pred_co2:.1f} g/km")
            if model_src:
                st.caption(f"Modèle : {model_src}")
            else:
                st.caption("Aucun modèle détecté : mode démo utilisé.")
