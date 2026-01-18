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
  color: var(--accent);
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
            "Relation locale entre une variable clé (ex. masse ou puissance) et son impact sur la sortie. "
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
            "R² CV": 0.934431,
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
            "XGB",
            "BaggingClassifier",
            "Random Forest Classifier",
            "Arbres de décision",
            "KNN",
            "Logistic Regression",
            "Ada Boost",
        ],
        "accuracy_train": [0.931047, 0.948904, 0.949954, 0.949956, 0.925815, 0.762282, 0.648459],
        "accuracy_test": [0.923282, 0.921747, 0.921333, 0.918734, 0.896947, 0.762897, 0.651021],
        "f1_weighted": [0.930407, 0.928753, 0.928805, 0.925257, 0.905331, 0.769626, 0.658861],
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
    {"class": "7", "label": "G (>250)", "min": 251, "max": 350, "example": "Voiture de sport"},
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

@st.cache_resource
def load_raw_data() -> pd.DataFrame:
    chunks= pd.read_csv("data/raw/2022_data.csv",
                       dtype={10: "string",12: "string",28: "string"},
                       usecols=[1,5,9,10,11,12,14,16,19,20,21,22,23,24,25,26,27,28,30,36,39],
                       chunksize=300_000)
    df_raw = pd.concat(chunks, ignore_index=True)
    return df_raw

def load_all_data() -> pd.DataFrame:
    df= pd.read_csv("data/raw/data_2010_2024.csv")
    return df

@st.cache_data
def load_data_dictionary() -> pd.DataFrame:
    path = Path("references/Dictionnaires_de_donnees.xlsx")
    if not path.exists():
        return pd.DataFrame(columns=["Colonne"])
    df_raw = pd.read_excel(path, sheet_name="Feuil1", header=None)
    header_idx = 1
    header = df_raw.iloc[header_idx].tolist()
    df = df_raw.iloc[header_idx + 1 :].copy()
    df.columns = header
    df = df.dropna(how="all")
    col_name = df.columns[1]
    col_desc = df.columns[2]
    col_drop = df.columns[3]
    col_na = df.columns[5]
    df = df[[col_name, col_desc, col_drop, col_na]].copy()
    df = df.rename(
        columns={
            col_name: "Colonne",
            col_desc: "Description",
            col_drop: "Raison",
            col_na: "NA_pct",
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
        "ELECTRIQUE": -base,
        "HYBRIDE RECHARGEABLE": -30,
        "HYBRIDE": -20,
        "PETROL": 30,
        "DIESEL": 40,
        "AUTRE": 10,
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
        height=380,
        showlegend=False,
        yaxis_range=[0.5, 1.02],
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


def chart_regression_reduced():
    df = REG_RESULTS_REDUCED.sort_values("R² test", ascending=False)
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
        height=380,
        showlegend=False,
        yaxis_range=[0.5, 1.02],
        yaxis_title="R² (test)",
        xaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def chart_classification():
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=CLF_METRICS["model"],
            y=CLF_METRICS["accuracy_test"],
            name="Accuracy test",
            marker_color="#e07a5f",
            text=[f"{v:.3f}" for v in CLF_METRICS["accuracy_test"]],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=CLF_METRICS["model"],
            y=CLF_METRICS["f1_weighted"],
            name="F1 Score",
            marker_color="#0f4c5c",
            text=[f"{v:.3f}" for v in CLF_METRICS["f1_weighted"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=360,
        barmode="group",
        yaxis_range=[0.6, 1.02],
        yaxis_title="Score",
        xaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def chart_thresholds():
    df = pd.DataFrame(CLASS_BINS)
    df["display_min"] = df["min"].clip(upper=500)
    df["display_max"] = df["max"].clip(upper=500)
    fig = go.Figure(
        go.Bar(
            x=(df["display_max"] - df["display_min"]),
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
    {"label": "Contexte & qualité des données", "id": "contexte"},
    {"label": "Pré-processing des données", "id": "preprocess"},
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
    


if slide_id == "intro":
    st.markdown(
        '<div class="slide-title"><h1>Introduction</h1></div>',
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown(
            """
    <div class="card">
    <div class="metric-big">Intérêt scientifique & métier</div>
    <div class="muted">
    Les véhicules particuliers représentent une part majeure des émissions de CO₂ du secteur des transports, 
    contribuant significativement au changement climatique. Dans un contexte de transition énergétique et de sobriété,
    comprendre les déterminants techniques des émissions de CO₂ est essentiel pour :
  <li><b>Éclairer les politiques publiques</b> en matière de régulation automobile</li>
  <li><b>Aider les constructeurs</b> à concevoir des modèles plus sobres et performants</li>
  <li><b>Guider les consommateurs</b> dans leurs choix de véhicules</li>
    </div>
    </div>
    <div style='height: 2.2em;'></div>
    """,
            unsafe_allow_html=True,
        )

        st.image(INTRO_IMAGES[1], width=505) 

    with col_b:
        
        st.image(INTRO_IMAGES[0], width=410)

        st.markdown(
            """
    <div style='height: 1.5em;'></div>           
    <div class="card">
    <div class="metric-big">Objectifs du projet</div>
    <div class="muted">
    Ce projet vise à analyser et prédire les émissions de CO₂ des véhicules particuliers à partir 
    de leurs caractéristiques techniques et structurelles :

    1. **Identifier les véhicules** les plus polluants 
    2. **Comprendre les caractéristiques techniques** qui influencent les émissions (masse, puissance, 
    type de carburant, hybridation, etc.)
    3. **Prédire l'émission CO₂** à partir des caractéristiques physiques disponibles avant la 
    construction du véhicule
    4. **Classer les véhicules** selon les catégories d'émissions européennes (A à G)
    </div>
    </div>
    """,
            unsafe_allow_html=True,
        )
    
    st.markdown(  
        """  
    <div class="card">
    <div class="metric-big">Résultats atteints</div>
    <div class="muted">
        Les modèles développés permettent de :<br>
        <ul>
            <li><b>Prédire les émissions CO₂</b> avec une précision élevée (R² > 0.99 pour la régression)</li>
            <li><b>Classifier les véhicules</b> avec une exactitude > 93%</li>
            <li><b>Identifier les variables les plus influentes</b> (type de carburant, masse, puissance, etc.)</li>
            <li><b>Fournir des explications locales et globales des prédictions</b></li>
        </ul>
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


if slide_id == "contexte":
    st.markdown(
        '<div class="slide-title"><h1>Contexte & qualité des données</h1></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="muted">
        Le projet s'appuie sur le dataset officiel des émissions CO₂ des véhicules particuliers 
        <b>EEA CO₂ cars</b> (European Environment Agency) couvrant la période 2010-2024, avec un focus 
        sur l'année 2022.
        </div> 
        """,
        unsafe_allow_html=True,
        )
    st.markdown('<hr style="border: none; border-top: 3px solid #F54927; margin: 1.2em 0;" />', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=" 2010-2024", value=">91M", delta='40 colonnes initiales')
        
    with col2:
        st.metric(label=" Focus 2022", value="9,48M", delta='Volume exploitable')
       
    with col3:
        st.metric(label=" Doublons supprimés", value="533K", delta='17 colonnes')
        
    with col4:
        st.metric(label=" Cible Ewltp priorisée", value="1% NA", delta='Enedc ≈ 82% NA')
        
    st.markdown('<hr style="border: none; border-top: 3px solid #F54927; margin: 1.2em 0;" />', unsafe_allow_html=True)


    st.markdown("""
        <div class="card">
        <div class="metric-big">Variable cible : Ewltp</div>
        <div class="muted">
    <b>Ewltp</b> :<br> 
    <li> ≈ 1% de valeurs manquantes → excellente complétude</li>
    <li> Cycle WLTP (Worldwide Harmonized Light Vehicles Test Procedure) plus récent</li>
    <li> Standard européen depuis 2017</li>
    <b>Enedc</b> :<br>
    <li> ≈ 82% de valeurs manquantes</li>
    <li> obsolète (ancien cycle NEDC)</li><br>

    **Décision** : Utilisation de Ewltp comme variable cible
    </div> 
    </div>
        """,
        unsafe_allow_html=True,)


    st.markdown("""
    <div style='height: 1.5em;'></div>
    <div class="card">
    <div class="metric-big">Problèmes de qualité identifiés</div>
    <div class="muted">
    <div style='height: 1.5em;'></div>
    <div style="display: flex; gap: 4em;">
      <div style="flex: 1; min-width:0;">
        <b>1. Doublons</b><br>
        - 533 390 enregistrements dupliqués détectés<br>
        - Suppression effectuée après analyse<br><br>
        <b>2. Valeurs manquantes</b><br>
        - Variables avec taux de NA élevés identifiées<br>
        - Stratégie d'imputation définie par type de variable<br><br>
        <b>3. Incohérences</b><br>
        - Libellés de constructeurs/modèles non harmonisés<br>
        - Exemples : "BMW" vs "B.M.W." vs "bmw"<br>
        - Harmonisation nécessaire pour l'encodage<br>
      </div>
      <div style="flex: 1; min-width:0;">
        <b>4. Faux NA</b><br>
        - Valeurs manquantes mais codées sous d'autres formats<br>
        - Nettoyage préalable requis<br><br>
        <b>5. Variables corrélées/dérivées</b><br>
        - Colonnes calculées à partir de la cible (risque de fuite)<br>
        - Exclusion préventive nécessaire<br>
      </div>
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='height: 1.5em;'></div>
    <div class="card">
    <div class="metric-big">Colonnes exclues</div>
    <div class="muted">
    <ul>
    <li>Redondance avec d'autres variables</li>
    <li>Taux de NA > 80%</li>
    <li>Variables non disponibles avant construction du véhicule</li>
    <li>Variables trop corrélées à la cible</li>
    </ul>
    </div>
    </div>
    """,unsafe_allow_html=True,
        )



    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)
    

    with st.expander("Colonnes exclues"):
        df_dict = load_data_dictionary()
        df_dict = df_dict[df_dict["Raison"].notna()].copy()
        df_dict_sub = df_dict.iloc[:, [0, 2, 3]]
        st.dataframe(df_dict_sub, use_container_width=True, height=300)
            

    with st.expander("Colonnes conservées"):
        df_dict = load_data_dictionary()
        df_dict = df_dict[df_dict["Raison"].isna()].copy()
        df_dict_sub = df_dict.iloc[:, [0, 1, 3]]
        st.dataframe(df_dict_sub, use_container_width=True, height=300)  

    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.image(asset_path('Emissions CO2 par année'))
        

        st.image(asset_path('co2_mean_by_fuel'))

        
        
    with col_b:
        st.image(asset_path('Distribution CO2'))
        

    st.image(asset_path('heatmap'), width=1000)
        

if slide_id == "preprocess":
    st.markdown(
        '<div class="slide-title"><h1>Pré-processing des données</h1></div>',
        unsafe_allow_html=True,
    )
    
    st.markdown(  
        """  
    <div class="card">
    <div class="metric-big">Pipeline de pré-processing</div>
    <div style='height: 1em;'></div>
    <div class="muted">
        <div style="display: flex; gap: 19em;">
            <div>
                1. Chargement des données<br>
                2. Suppression des doublons<br>
                3. Harmonisation des libellés<br>
                4. Nettoyage des faux NA<br>
                5. Split train/test (80/20)<br>
            </div>
            <div>
                6. Imputation (numériques et qualitatives)<br>
                7. Encodage<br>
                8. Standardisation<br>
                9. Discrétisation de la variable cible<br>
                10. Prêt pour la modélisation
            </div>
        </div>
    </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<div style='height: 1.5em;'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.2, 1])   # Colonne a plus large
    with col_a:
        st.markdown("""
        <div class="card">
        <div class="metric-big">Variables numériques</div>
        <div class="muted">
        
        **Imputation :**
        - Médiane ou moyenne groupée (par constructeur, modèle, etc.)
        
        **Indicateurs de manquants :**
        - Création de variables binaires pour certaines variables clés
        - Permet de distinguer "vraiment manquant" vs "valeur imputée"
        
        **Standardisation :**
        - StandardScaler
        </div>
        </div>
        """,
        unsafe_allow_html=True,)
        
    with col_b:
        st.markdown("""
        <div class="card">
        <div class="metric-big">Variables qualitatives</div>
        <div class="muted">

        **Nettoyage préalable :**
        - Harmonisation des libellés et nettoyage des faux NA
        
        **Imputation :**
        - Mode groupé (ex. Man par Mk ou Cn)
        
        **Encodage :**
        - One-Hot Encoding sur catégories à faible cardinalité
        - Hashing pour ceux à haute cardinalité
        </div>
        </div>
        """,
        unsafe_allow_html=True,)    
          

if slide_id == "regression":
    st.markdown(
        '<div class="slide-title"><h1>Régression</h1></div>',
        unsafe_allow_html=True,
    )
    
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Modèles entrainés sur l'ensemble des données**")

        st.plotly_chart(chart_regression(), use_container_width=True)

        st.dataframe(REG_RESULTS_FULL, use_container_width=True)

    with col_b:
        st.markdown("**Modèles entrainés sur les données structurelles uniquement**")

        st.plotly_chart(chart_regression_reduced(), use_container_width=True)
        st.dataframe(REG_RESULTS_REDUCED, use_container_width=True)


    st.markdown(
        """
    <div class="card" style="margin-bottom: 16px;">
    <div class="metric-big">Modèle le plus performant : Random Forest</div>
    <div class="muted">Le modèle pert très peu en efficacité malgré la réduction de variables très correlées : R² Cross Val = 99.41%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("Colonnes structurelles supprimées", expanded=True):
        
        cols_drop = [
            "z (Wh/km)",
            "Erwltp (g/km)",
            "Fuel consumption (l/100km)",
            "Electric range (km en une charge)",
        ]
        st.dataframe(pd.DataFrame({"Colonne": cols_drop}), use_container_width=True, height=180)


if slide_id == "classification":
    st.markdown(
        '<div class="slide-title"><h1>Classification & optimisation</h1></div>',
        unsafe_allow_html=True,
    )
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
        """
        <div class="card">
        <div class="metric-big">Grille de classification CO₂ (étiquette énergie + catégorie 0 émission)</div>
        <div class="muted">
        La classification suivante est adaptée de l’étiquette énergie utilisée en Europe dans le cadre de la directive 1999/94/EC. 
        Elle est enrichie d’une catégorie supplémentaire <strong>"0"</strong> destinée aux véhicules <strong>zéro émission</strong>, 
        principalement électriques ou à hydrogène. Cette grille permet de catégoriser facilement les véhicules du moins au plus 
        émetteur, à partir de la valeur cible : <code>Ewltp (g CO₂/km)</code>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
        )
    with tabs[1]:
        st.plotly_chart(chart_classification(), use_container_width=True)
        st.markdown(
            """
        <div class="card" style="margin-bottom: 16px;">
        <div class="metric-big">Modèle le plus performant : XGBoost F1 Score = 0.930407</div>
        <div class="muted">
        Le modèle de Bagging optimisé obtient des performances très proches, mais le XGBoost a été privilégié pour :
        <ul>
        <li>Sa capacité de modélisation fine</li>
        <li>Mieux adapté dans le cas des classes déséquilibrés</li>
        <li>Possibilité de tunning très fin des hyperparamètres</li>
        <li>Ses outils d’explicabilité plus riches</li>
        <li>Sa meilleure adaptabilité en contexte industriel</li>
        </ul>
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        with st.expander("Comparatif détaillé des modèles (Accuracy / Balanced Accuracy / F1 / Temps)", expanded=False):
            df_models = pd.DataFrame(
                {
                    "Model": [
                        "XGBClassifier",
                        "BaggingClassifier",
                        "RandomForestClassifier",
                        "DecisionTreeClassifier",
                        "ExtraTreesClassifier",
                        "LGBMClassifier",
                        "ExtraTreeClassifier",
                        "KNeighborsClassifier",
                        "LogisticRegression",
                        "LinearDiscriminantAnalysis",
                        "LinearSVC",
                        "SGDClassifier",
                        "NearestCentroid",
                        "Perceptron",
                        "BernoulliNB",
                        "RidgeClassifierCV",
                        "RidgeClassifier",
                        "PassiveAggressiveClassifier",
                        "AdaBoostClassifier",
                        "QuadraticDiscriminantAnalysis",
                        "GaussianNB",
                        "DummyClassifier",
                    ],
                    "Accuracy": [
                        0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.91, 0.90, 0.76, 0.70, 0.67, 0.64, 0.60, 0.55, 0.55, 0.59, 0.59, 0.51, 0.52, 0.27, 0.28, 0.24,
                    ],
                    "Balanced Accuracy": [
                        0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.92, 0.90, 0.75, 0.68, 0.65, 0.65, 0.64, 0.55, 0.53, 0.52, 0.52, 0.52, 0.43, 0.41, 0.40, 0.12,
                    ],
                    "F1 Score": [
                        0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.91, 0.90, 0.76, 0.69, 0.65, 0.60, 0.60, 0.54, 0.51, 0.55, 0.55, 0.50, 0.47, 0.28, 0.28, 0.09,
                    ],
                    "Time Taken": [
                        8.72, 17.21, 40.28, 3.19, 48.52, 6.34, 1.11, 40.20, 8.40, 1.92, 83.30, 11.48, 0.93, 5.29, 0.82, 2.09, 0.94, 5.25, 15.28, 1.78, 1.12, 0.62,
                    ],
                }
            )

            st.dataframe(df_models, use_container_width=True, height=520)
    with tabs[2]:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Matrice de confusion avant optimisation**")
st.image("assets/cm_before_Blues.png", use_container_width=True)

        with col_b:
            st.markdown("**Matrice de confusion après optimisation**")
st.image("assets/cm_after_Blues.png", use_container_width=True)


if slide_id == "shap":
    st.markdown(
        '<div class="slide-title"><h1>Explicabilité (SHAP)</h1></div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:

        st.image(SHAP_PANELS[0]["path"])
        st.markdown(f"<div class='muted'>{SHAP_PANELS[0]['desc']}</div>", unsafe_allow_html=True)
        st.markdown('<hr style="border: none; border-top: 3px solid #008bfb; margin: 1em 0;" />', unsafe_allow_html=True)
        st.image(SHAP_PANELS[3]["path"])
        st.markdown(f"<div class='muted'>{SHAP_PANELS[3]['desc']}</div>", unsafe_allow_html=True)
    with col_b:
        st.image(SHAP_PANELS[2]["path"])
        st.markdown("<div style='height: 0.5em;'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='muted'>{SHAP_PANELS[2]['desc']}</div>", unsafe_allow_html=True)
        st.markdown('<hr style="border: none; border-top: 3px solid #008bfb; margin: 1.2em 0;" />', unsafe_allow_html=True)
        st.markdown("<div style='height: 2.5em;'></div>", unsafe_allow_html=True)
        st.image(SHAP_PANELS[1]["path"])
        st.markdown(f"<div class='muted'>{SHAP_PANELS[1]['desc']}</div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="card">
        <div class="metric-big">Lecture clé</div>
        <div class="muted">
        <ul>
        <li>Les décisions du modèle sont pilotées par des <b>facteurs physiques et énergétiques plausibles</b>.</li>
        <li>Les effets sont monotones, continus et cohérents avec la réglementation.</li>
        <li>Les erreurs se concentrent logiquement aux <b>frontières de classes</b>.</li>
        <li>Le modèle est à la fois <b>performant, explicable et fiable</b>.</li>
        </ul>
    Cette vue confirme que le modèle apprend une structure globale cohérente, tout en adaptant ses décisions aux spécificités de chaque classe CO₂.
    </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


if slide_id == "demo":
    st.markdown(
        '<div class="slide-title"><h1>Démo live</h1></div>',
        unsafe_allow_html=True,
    )
    section_header("Prédire la classe CO2 (0–7)")
    
    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        with st.form("demo_form"):
            mass_kg = st.selectbox("Masse (kg)", list(range(600, 4001, 50)))
            cyl_cm3 = st.selectbox("cylindre_du_moteur_cm3", list(range(800, 6001, 100)))
            power_kw = st.selectbox("puissance_du_moteur_kw", list(range(40, 501, 10)))
            width_mm = st.selectbox("W (mm)", list(range(1500, 2301, 10)))
            at1_mm = st.selectbox("At1 (mm)", list(range(1200, 2001, 10)))
            at2_mm = st.selectbox("At2 (mm)", list(range(1200, 2001, 10)))
            it_flag = st.selectbox("IT (0/1)", [0, 1])
            fuel_type = st.selectbox(
                "Ft (carburant)",
                ["PETROL", "DIESEL", "HYBRIDE", "HYBRIDE RECHARGEABLE", "ELECTRIQUE", "AUTRE"],
            )
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
                "Ft": fuel_type,
            }

            model, model_src = load_model(ASSET_DIR / "xgb_classifier.joblib")
            pred_class = None
            pred_co2 = None

            
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
            
            st.caption(f"Modèle : XGBoost")
           



