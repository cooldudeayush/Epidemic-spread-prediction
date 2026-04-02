from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "outputs" / "predictions" / "latest_country_predictions.csv"
CASE_TRENDS = ROOT / "outputs" / "app" / "case_trends.csv"
MOBILITY_SNAPSHOT = ROOT / "outputs" / "app" / "mobility_snapshot.csv"


@st.cache_data
def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trends = pd.read_csv(CASE_TRENDS, parse_dates=["date"])
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    mobility = pd.read_csv(MOBILITY_SNAPSHOT)
    return trends, predictions, mobility


st.set_page_config(page_title="Global COVID Risk Dashboard", layout="wide")
st.title("Global COVID Outbreak Forecasting and Risk Dashboard")
st.caption("Country-level 7-day case forecasts and hotspot risk estimates.")

required_files = [PREDICTIONS, CASE_TRENDS, MOBILITY_SNAPSHOT]
if any(not path.exists() for path in required_files):
    st.warning("Missing deployment data files. Commit the lightweight files in `outputs/app/` and `outputs/predictions/`.")
    st.stop()

trends, predictions, mobility = load_outputs()

country_options = ["All"] + sorted(predictions["country"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Country", country_options)

col1, col2, col3 = st.columns(3)
col1.metric("Countries Covered", int(predictions["country"].nunique()))
col2.metric("High Risk Countries", int((predictions["risk_label"] == "high").sum()))
col3.metric("Latest Prediction Date", str(predictions["date"].max().date()))

map_fig = px.choropleth(
    predictions.dropna(subset=["iso_code"]),
    locations="iso_code",
    color="risk_label",
    hover_name="country",
    hover_data={
        "predicted_cases_7d": ":.0f",
        "predicted_growth_rate": ":.2f",
        "risk_score": ":.2f",
    },
    color_discrete_map={"low": "#4caf50", "medium": "#ff9800", "high": "#e53935"},
    title="Predicted Hotspot Risk by Country",
)
map_fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
st.plotly_chart(map_fig, use_container_width=True)

table_view = predictions.copy()
if selected_country != "All":
    table_view = table_view[table_view["country"] == selected_country]

st.subheader("Predicted Country Risk Table")
st.dataframe(
    table_view.sort_values(["risk_label", "predicted_cases_7d"], ascending=[True, False]),
    use_container_width=True,
)

st.subheader("Case Trends")
if selected_country == "All":
    top_countries = predictions.sort_values("predicted_cases_7d", ascending=False).head(8)["country"]
    trend_data = trends[trends["country"].isin(top_countries)]
else:
    trend_data = trends[trends["country"] == selected_country]

trend_fig = px.line(
    trend_data,
    x="date",
    y="new_cases_7d_avg",
    color="country",
    title="Smoothed Daily Cases",
)
st.plotly_chart(trend_fig, use_container_width=True)

mobility_cols = [col for col in mobility.columns if col.endswith("_lag7")]
if selected_country != "All" and mobility_cols:
    st.subheader("Mobility Drivers")
    latest_country_row = (
        mobility[mobility["country"] == selected_country]
        .melt(value_vars=mobility_cols, var_name="feature", value_name="value")
        .dropna()
    )
    if not latest_country_row.empty:
        driver_fig = px.bar(latest_country_row, x="feature", y="value", title="Lagged Mobility Signals")
        st.plotly_chart(driver_fig, use_container_width=True)
