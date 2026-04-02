# Epidemic Spread Prediction

Global COVID-19 outbreak forecasting and hotspot risk detection system built for hackathon-style epidemiology + AI projects.

This repository predicts short-term disease spread at the country level by combining historical case trends, mobility behavior, and mortality-related signals. It also provides an interactive dashboard for outbreak monitoring and risk visualization.

## Features

- 7-day country-level COVID case forecasting
- hotspot risk scoring and classification into `low`, `medium`, and `high`
- preprocessing pipeline for Johns Hopkins time-series data
- country-level feature enrichment from Google mobility and excess-deaths data
- optional support for `owid-covid-data.csv` for richer epidemiological features
- interactive Streamlit dashboard with:
  - global choropleth risk map
  - country risk table
  - smoothed case-trend charts
  - mobility driver view for selected countries

## Current Data Sources

Implemented around these datasets:

- Johns Hopkins global confirmed case time series
- Google COVID-19 Community Mobility Reports
- OWID excess-deaths dataset

Optional drop-in enhancement:

- `owid-covid-data.csv`

If the OWID master file is added to the project root, the pipeline automatically uses extra fields such as population, testing, vaccination, hospital, ICU, positivity, and stringency signals.

## Included In This Repository

Committed directly to the repo:

- `time_series_covid19_confirmed_global.csv`
- `estimated-cumulative-excess-deaths-per-100000-people-during-covid-19.csv`
- `Global_Mobility_Report.zip`

The original mobility CSV is about `1.14 GB`, so the repository stores a compressed archive instead.

To run the full pipeline with mobility features after cloning, extract:

- `Global_Mobility_Report.zip`

into:

- `Global_Mobility_Report.csv`

## Repository Structure

```text
.
|-- dashboard.py
|-- run_pipeline.py
|-- requirements.txt
|-- outputs/
|   |-- metrics/
|   `-- predictions/
`-- src/
    `-- epidemic_prediction/
        |-- config.py
        |-- data.py
        |-- features.py
        |-- modeling.py
        `-- pipeline.py
```

## Setup

```powershell
& 'C:\Users\ayush\AppData\Local\Programs\Python\Python313\python.exe' -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## How To Run

Run the training pipeline:

```powershell
python run_pipeline.py
```

Run the dashboard:

```powershell
streamlit run dashboard.py
```

## Generated Outputs

- `outputs/predictions/latest_country_predictions.csv`
- `outputs/app/case_trends.csv`
- `outputs/app/mobility_snapshot.csv`
- `outputs/metrics/forecast_metrics.json`
- `outputs/metrics/risk_metrics.json`
- `outputs/metrics/pipeline_summary.csv`

The large intermediate modeling table is generated locally and intentionally excluded from GitHub.

## Streamlit Deployment

This repository is now set up so the deployed Streamlit app reads only the lightweight files inside:

- `outputs/predictions/`
- `outputs/app/`
- `outputs/metrics/`

That means Streamlit Cloud does not need the full local training artifacts to render the dashboard.

## Current Pipeline Summary

Latest local run:

- 201 countries covered
- 1,143 daily timestamps
- 229,743 modeling rows
- 31 model features
- forecast MAE: `15151.18`
- forecast RMSE: `65478.08`
- risk classification accuracy: `0.7566`
- weighted F1: `0.7603`

## Important Notes

- Raw source datasets are not committed because they are too large for normal GitHub storage.
- Plotly's built-in world country geometry is used for the global risk map, so an external GeoJSON file is not required for the current dashboard.
- The current v1 is country-level by design for stable joins and faster iteration.
- Adding the OWID master dataset is the most valuable next upgrade for stronger epidemiological realism.
