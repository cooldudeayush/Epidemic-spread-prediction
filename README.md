<div align="center">

# Epidemic Spread Prediction

### Global COVID-19 outbreak forecasting and hotspot risk dashboard

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly-3f4f75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![Live App](https://img.shields.io/badge/Live-Demo-brightgreen.svg)](https://epidemic-spread-prediction.streamlit.app/)

Predicting short-term COVID-19 spread using epidemiological signals, mobility behavior, and public-health feature engineering.

[Live Streamlit App](https://epidemic-spread-prediction.streamlit.app/)

</div>

## Overview

This project is a country-level epidemic intelligence system built to forecast COVID-19 spread and identify outbreak hotspots. It combines historical case trends with mobility and mortality-related indicators, and optionally enriches the feature space with the official Our World in Data COVID dataset for population, testing, vaccination, and healthcare context.

The system is split into two practical layers:

- a forecasting layer that predicts the next 7 days of cases
- a risk layer that translates outbreak conditions into a severity-style `risk_score` and `risk_label`

The final output is visualized through an interactive Streamlit dashboard with a global risk map, case trend views, and country-level prediction summaries.

## Features

- 7-day country-level COVID case forecasting
- hotspot severity scoring with `low`, `medium`, and `high` labels
- ISO-3 standardized map rendering for reliable country visualization
- Johns Hopkins cumulative-to-daily case preprocessing
- Google mobility feature integration
- OWID excess-deaths support
- optional OWID master dataset enrichment for:
  - population
  - testing and positivity
  - vaccination coverage
  - hospitalization and ICU context where available
- Streamlit dashboard with:
  - world choropleth risk map
  - predicted country risk table
  - smoothed case trend plots
  - mobility signal snapshot for selected countries

## Tech Stack & Tools

- `Python` for data processing and modeling
- `pandas` and `numpy` for transformation and feature engineering
- `scikit-learn` for forecasting and evaluation
- `Streamlit` for the web dashboard
- `Plotly` for interactive maps and charts
- `Git` and `GitHub` for version control and deployment workflow
- `Streamlit Community Cloud` for hosting the dashboard

## Datasets Used

Primary local datasets used in this repository:

- `time_series_covid19_confirmed_global.csv`
- `estimated-cumulative-excess-deaths-per-100000-people-during-covid-19.csv`
- `Global_Mobility_Report.zip`

Additional enrichment supported:

- `owid-covid-data.csv`

Notes:

- `Global_Mobility_Report.zip` is a compressed version of the original mobility CSV because the raw file is very large.
- To run the full local pipeline, extract `Global_Mobility_Report.zip` into `Global_Mobility_Report.csv`.
- The official OWID master dataset is supported automatically when placed in the project root.

## Repository Structure

```text
.
|-- dashboard.py
|-- run_pipeline.py
|-- requirements.txt
|-- outputs/
|   |-- app/
|   |-- metrics/
|   `-- predictions/
`-- src/
    `-- epidemic_prediction/
        |-- config.py
        |-- data.py
        |-- features.py
        |-- modeling.py
        |-- pipeline.py
        `-- risk.py
```

## Installation / Setup

### 1. Clone the repository

```powershell
git clone https://github.com/cooldudeayush/Epidemic-spread-prediction.git
cd "Epidemic-spread-prediction"
```

### 2. Create and activate a virtual environment

```powershell
& 'C:\Users\ayush\AppData\Local\Programs\Python\Python313\python.exe' -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Prepare local datasets

Make sure these files are available in the project root:

- `time_series_covid19_confirmed_global.csv`
- `estimated-cumulative-excess-deaths-per-100000-people-during-covid-19.csv`
- `Global_Mobility_Report.csv`

Optional but recommended:

- `owid-covid-data.csv`

If you only have the compressed mobility file from GitHub, extract:

```powershell
Expand-Archive -LiteralPath "Global_Mobility_Report.zip" -DestinationPath "."
```

### 5. Run the pipeline

```powershell
python run_pipeline.py
```

### 6. Launch the dashboard

```powershell
streamlit run dashboard.py
```

If `streamlit` is not recognized:

```powershell
& 'C:\Users\ayush\AppData\Local\Programs\Python\Python313\Scripts\streamlit.exe' run dashboard.py
```

## Technical Workflow

### 1. Data ingestion

- load Johns Hopkins confirmed case time series
- load Google mobility reports at country level
- load OWID excess-deaths data
- optionally load the OWID master COVID dataset

### 2. Preprocessing

- convert cumulative confirmed counts into daily case series
- aggregate rows to country level
- smooth reporting noise with rolling windows
- standardize country names
- standardize map country codes to ISO-3

### 3. Feature engineering

- create lag-based case features
- build rolling trend and growth features
- add mobility lag features
- add excess-death and mortality context
- use OWID population, testing, vaccination, and positivity data when available

### 4. Forecasting

- train a regression model to predict next 7-day case totals
- evaluate forecast quality with MAE and RMSE

### 5. Risk scoring

- compute a severity-style outbreak score instead of raw classifier confidence
- combine predicted burden, recent burden, growth, positivity, and vaccination protection
- assign `low`, `medium`, and `high` labels from calibrated score bands

### 6. Dashboard serving

- save lightweight deployment files under `outputs/app/`, `outputs/predictions/`, and `outputs/metrics/`
- render the Streamlit app using only those lightweight files
- deploy the dashboard on Streamlit Community Cloud

## Generated Outputs

- `outputs/predictions/latest_country_predictions.csv`
- `outputs/app/case_trends.csv`
- `outputs/app/mobility_snapshot.csv`
- `outputs/metrics/forecast_metrics.json`
- `outputs/metrics/risk_metrics.json`
- `outputs/metrics/pipeline_summary.csv`

The large intermediate modeling table is kept local and is not required by the deployed Streamlit app.

## Current Results Snapshot

Latest local run:

- 201 countries covered
- 1,143 timestamps
- forecast MAE: `10301.64`
- forecast RMSE: `49079.15`
- risk accuracy: `0.7670`
- weighted F1: `0.7609`

## Streamlit Deployment

This repository is deployment-ready for Streamlit Community Cloud. The hosted app reads only lightweight files from:

- `outputs/app/`
- `outputs/predictions/`
- `outputs/metrics/`

That means the deployed dashboard does not need the full training artifacts to render correctly.

## Future Improvements

- improve forecasting with stronger temporal models
- add more robust cross-country calibration for risk thresholds
- include weather and healthcare capacity signals
- support region-level or India-specific hotspot prediction
- automate scheduled pipeline refreshes for production use
