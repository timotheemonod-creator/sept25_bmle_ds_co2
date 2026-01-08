# Projet CO2 - DataScientest 2024

## Objectif
Analyser et modeliser les emissions de CO2 des vehicules neufs en Europe afin d'identifier les facteurs d'influence et de predire les emissions futures.

## Donnees
- Source: EEA CO2 cars emission dataset (2010-2024)
- Les fichiers bruts ne sont pas versionnes (volumetrie).
- Fichiers attendus dans `data/raw/`:
  - `final_2010_2023_data.csv`
  - `previsionnal_2024_data.csv`
  - `final_2022_data.csv` (focus EDA)
- Exemples de fichiers produits: `data/processed/*.parquet`

## Structure du projet
- `data/` : donnees brutes, nettoyees et externes
- `notebooks/` : notebooks Jupyter (exploration, preprocessing, modelisation)
- `src/` : scripts Python modulaires
- `streamlit_app/` : structure pour l'application Streamlit
- `reports/` : rapports et figures
- `docs/` : documentation et metadonnees

## Installation
```bash
git clone <url_repo>
cd Projet_Co2
python -m venv venv_projet_co2
venv_projet_co2\Scripts\activate
pip install -r requirements.txt
```

Pour une reproduction stricte:
```bash
pip install -r requirements_locked.txt
```

## Notebooks
- `notebooks/1.0 Exploration des donn?es.ipynb` : EDA global + focus 2022

