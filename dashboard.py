from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "outputs" / "predictions" / "latest_country_predictions.csv"
CASE_TRENDS = ROOT / "outputs" / "app" / "case_trends.csv"
MOBILITY_SNAPSHOT = ROOT / "outputs" / "app" / "mobility_snapshot.csv"

RISK_COLORS = {"low": "#2fbf71", "medium": "#f4a62a", "high": "#ef5d5d"}

THEMES = {
    "Light": {
        "page_bg": "#f6f8fb",
        "panel_bg": "#ffffff",
        "panel_alt": "#f0f5fb",
        "text": "#1a2433",
        "muted": "#6e7d92",
        "border": "rgba(22, 34, 56, 0.08)",
        "accent": "#1f6feb",
        "accent_soft": "rgba(31, 111, 235, 0.08)",
        "shadow": "0 14px 40px rgba(15, 23, 42, 0.08)",
        "template": "plotly_white",
        "land": "#edf2f8",
        "ocean": "#f6f8fb",
        "grid": "rgba(22, 34, 56, 0.08)",
    },
    "Dark": {
        "page_bg": "#09121d",
        "panel_bg": "#101c2b",
        "panel_alt": "#0c1725",
        "text": "#eef4fb",
        "muted": "#8da0bc",
        "border": "rgba(164, 182, 209, 0.12)",
        "accent": "#7dd3fc",
        "accent_soft": "rgba(125, 211, 252, 0.10)",
        "shadow": "0 18px 50px rgba(0, 0, 0, 0.34)",
        "template": "plotly_dark",
        "land": "#142335",
        "ocean": "#0a1220",
        "grid": "rgba(164, 182, 209, 0.12)",
    },
}


@st.cache_data
def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trends = pd.read_csv(CASE_TRENDS, parse_dates=["date"])
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    mobility = pd.read_csv(MOBILITY_SNAPSHOT)
    return trends, predictions, mobility


def apply_theme(theme_name: str) -> dict[str, str]:
    theme = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(31, 111, 235, 0.05), transparent 22%),
                linear-gradient(180deg, {theme["page_bg"]} 0%, {theme["page_bg"]} 100%);
            color: {theme["text"]};
        }}
        .block-container {{
            max-width: 1380px;
            padding-top: 1.35rem;
            padding-bottom: 2rem;
        }}
        [data-testid="stSidebar"] {{
            background: {theme["panel_alt"]};
            border-right: 1px solid {theme["border"]};
        }}
        [data-testid="stSidebar"] * {{
            color: {theme["text"]};
        }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {{
            color: {theme["text"]};
        }}
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stToggle label {{
            color: {theme["text"]} !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] input {{
            color: {theme["text"]} !important;
            background-color: {theme["panel_bg"]} !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] {{
            background-color: {theme["panel_bg"]} !important;
            border-radius: 12px;
        }}
        [data-testid="stSidebar"] [data-baseweb="radio"] > div {{
            color: {theme["text"]} !important;
        }}
        .top-shell {{
            background: {theme["panel_bg"]};
            border: 1px solid {theme["border"]};
            box-shadow: {theme["shadow"]};
            border-radius: 28px;
            padding: 1.15rem 1.45rem;
            margin-bottom: 1rem;
        }}
        .top-line {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }}
        .title-block {{
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
        }}
        .dash-title {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: {theme["text"]};
            margin: 0;
        }}
        .dash-subtitle {{
            color: {theme["muted"]};
            font-size: 0.98rem;
            margin: 0;
        }}
        .refresh-pill {{
            padding: 0.7rem 1rem;
            border-radius: 999px;
            background: {theme["accent_soft"]};
            color: {theme["text"]};
            font-size: 0.9rem;
            border: 1px solid {theme["border"]};
            white-space: nowrap;
        }}
        .kpi-card {{
            background: {theme["panel_bg"]};
            border: 1px solid {theme["border"]};
            box-shadow: {theme["shadow"]};
            border-radius: 22px;
            padding: 1rem 1.1rem;
            min-height: 126px;
            margin-bottom: 0.6rem;
        }}
        .kpi-label {{
            color: {theme["muted"]};
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }}
        .kpi-value {{
            color: {theme["text"]};
            font-size: 2.05rem;
            font-weight: 800;
            line-height: 1.02;
            margin: 0;
        }}
        .kpi-foot {{
            color: {theme["muted"]};
            font-size: 0.86rem;
            margin-top: 0.65rem;
        }}
        .panel-card {{
            background: {theme["panel_bg"]};
            border: 1px solid {theme["border"]};
            box-shadow: {theme["shadow"]};
            border-radius: 24px;
            padding: 0.95rem 1rem 0.6rem 1rem;
            margin-bottom: 1rem;
        }}
        .panel-title {{
            color: {theme["text"]};
            font-size: 1.1rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }}
        .panel-note {{
            color: {theme["muted"]};
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }}
        div[data-testid="stDataFrame"] {{
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid {theme["border"]};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {theme["text"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return theme


def kpi(label: str, value: str, foot: str) -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-foot">{foot}</div>
    </div>
    """


def format_big_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def panel_header(title: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="panel-title">{title}</div>
        <div class="panel-note">{note}</div>
        """,
        unsafe_allow_html=True,
    )


def make_map(frame: pd.DataFrame, theme: dict[str, str]) -> go.Figure:
    fig = px.choropleth(
        frame.dropna(subset=["iso_code"]),
        locations="iso_code",
        locationmode="ISO-3",
        color="risk_label",
        color_discrete_map=RISK_COLORS,
        hover_name="country",
        hover_data={
            "predicted_cases_7d": ":.0f",
            "predicted_cases_per_100k_7d": ":.2f",
            "predicted_growth_rate": ":.2f",
            "risk_score": ":.2f",
        },
        template=theme["template"],
    )
    fig.update_traces(marker_line_color=theme["page_bg"], marker_line_width=0.45)
    fig.update_geos(
        projection_type="natural earth",
        bgcolor="rgba(0,0,0,0)",
        showframe=False,
        showcoastlines=False,
        showocean=True,
        oceancolor=theme["ocean"],
        showland=True,
        landcolor=theme["land"],
        showcountries=True,
        countrycolor=theme["grid"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=430,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom"),
        legend_title_text="Risk",
    )
    return fig


def make_trend_chart(trend_data: pd.DataFrame, theme: dict[str, str], selected_country: str) -> go.Figure:
    title = "Case Trend" if selected_country == "All" else f"{selected_country} Case Trend"
    fig = px.line(
        trend_data,
        x="date",
        y="new_cases_7d_avg",
        color="country",
        template=theme["template"],
    )
    fig.update_traces(line=dict(width=3))
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=48, b=0),
        height=360,
        legend_title_text="Country",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title="7-day avg cases", gridcolor=theme["grid"])
    return fig


def make_distribution_chart(predictions: pd.DataFrame, theme: dict[str, str]) -> go.Figure:
    counts = predictions["risk_label"].value_counts().reindex(["high", "medium", "low"], fill_value=0)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.66,
                marker=dict(colors=[RISK_COLORS[label] for label in counts.index]),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template=theme["template"],
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=290,
        showlegend=False,
    )
    return fig


def make_top_burden_chart(predictions: pd.DataFrame, theme: dict[str, str]) -> go.Figure:
    frame = predictions.sort_values("predicted_cases_7d", ascending=False).head(8).sort_values("predicted_cases_7d")
    fig = px.bar(
        frame,
        x="predicted_cases_7d",
        y="country",
        color="risk_label",
        color_discrete_map=RISK_COLORS,
        orientation="h",
        template=theme["template"],
        text="predicted_cases_7d",
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=8, t=0, b=0),
        height=360,
        showlegend=False,
        xaxis_title="Predicted cases in next 7 days",
        yaxis_title="",
    )
    return fig


def make_risk_signal_chart(predictions: pd.DataFrame, theme: dict[str, str]) -> go.Figure:
    frame = predictions.sort_values(["risk_score", "predicted_cases_7d"], ascending=[False, False]).head(8).sort_values("risk_score")
    fig = px.bar(
        frame,
        x="risk_score",
        y="country",
        color="risk_label",
        color_discrete_map=RISK_COLORS,
        orientation="h",
        template=theme["template"],
        text="risk_score",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=8, t=0, b=0),
        height=360,
        showlegend=False,
        xaxis_title="Risk score",
        yaxis_title="",
    )
    return fig


def make_mobility_chart(mobility_row: pd.DataFrame, theme: dict[str, str]) -> go.Figure:
    cleaned = mobility_row.copy()
    cleaned["feature"] = (
        cleaned["feature"]
        .str.replace("_percent_change_from_baseline_lag7", "", regex=False)
        .str.replace("_", " ")
        .str.title()
    )
    cleaned["direction"] = cleaned["value"].apply(lambda v: "Increase" if v >= 0 else "Decrease")
    fig = px.bar(
        cleaned.sort_values("value"),
        x="value",
        y="feature",
        color="direction",
        color_discrete_map={"Increase": "#2fbf71", "Decrease": "#ef5d5d"},
        orientation="h",
        template=theme["template"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=320,
        xaxis_title="Change from baseline",
        yaxis_title="",
    )
    return fig


st.set_page_config(page_title="Global COVID Risk Dashboard", page_icon=":microbe:", layout="wide")

required_files = [PREDICTIONS, CASE_TRENDS, MOBILITY_SNAPSHOT]
if any(not path.exists() for path in required_files):
    st.warning("Missing deployment data files. Commit the lightweight files in `outputs/app/` and `outputs/predictions/`.")
    st.stop()

trends, predictions, mobility = load_outputs()

st.sidebar.markdown("## Dashboard Settings")
theme_name = st.sidebar.radio("Appearance", ["Light", "Dark"], index=0, horizontal=True)
theme = apply_theme(theme_name)

country_options = ["All"] + sorted(predictions["country"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Country focus", country_options)
show_non_low_only = st.sidebar.toggle("Hide low-risk countries", value=False)
sort_mode = st.sidebar.selectbox("Table sort", ["Highest risk", "Highest predicted cases", "Alphabetical"])

display_predictions = predictions.copy()
if show_non_low_only:
    display_predictions = display_predictions[display_predictions["risk_label"] != "low"]

latest_date = predictions["date"].max().strftime("%b %d, %Y")
global_cases = float(predictions["predicted_cases_7d"].sum())
top_risk_country = predictions.sort_values("risk_score", ascending=False).iloc[0]
highest_burden_country = predictions.sort_values("predicted_cases_7d", ascending=False).iloc[0]

selected_prediction = predictions[predictions["country"] == selected_country].copy() if selected_country != "All" else pd.DataFrame()

st.markdown(
    f"""
    <div class="top-shell">
      <div class="top-line">
        <div class="title-block">
          <div class="dash-title">Coronavirus Disease Situation Dashboard</div>
          <div class="dash-subtitle">Country-level 7-day forecasts, outbreak severity scoring, and mobility-linked surveillance insights.</div>
        </div>
        <div class="refresh-pill">Data refreshed at {latest_date}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_row = st.columns([1.15, 1, 1, 1], gap="medium")
with top_row[0]:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    panel_header("Select Country", "Switch between a global dashboard view and a country drilldown.")
    selected_country = st.selectbox("Country", country_options, index=country_options.index(selected_country), label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
with top_row[1]:
    st.markdown(kpi("Forecasted Cases", format_big_number(global_cases), f"Highest burden: {highest_burden_country['country']}"), unsafe_allow_html=True)
with top_row[2]:
    st.markdown(kpi("High-Risk Countries", str(int((predictions['risk_label'] == 'high').sum())), f"Strongest severity: {top_risk_country['country']}"), unsafe_allow_html=True)
with top_row[3]:
    st.markdown(kpi("Average Risk Score", f"{predictions['risk_score'].mean():.1f}", "Average severity across modeled countries"), unsafe_allow_html=True)

if selected_country != "All":
    selected_prediction = predictions[predictions["country"] == selected_country].copy()

map_col, trend_col = st.columns([1.55, 1], gap="medium")
with map_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    panel_header("Global Outbreak Surface", "Countries are colored by outbreak severity bands using the latest calibrated risk score.")
    st.plotly_chart(make_map(display_predictions, theme), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with trend_col:
    if selected_country == "All":
        trend_focus = predictions.sort_values(["risk_score", "predicted_cases_7d"], ascending=[False, False]).head(6)["country"]
        trend_data = trends[trends["country"].isin(trend_focus)]
        note = "Trend view for the most operationally relevant countries in the current snapshot."
    else:
        trend_data = trends[trends["country"] == selected_country]
        note = "Historical smoothed case trend for the selected country."
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    panel_header("Case Trend", note)
    st.plotly_chart(make_trend_chart(trend_data, theme, selected_country), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

mid_left, mid_mid, mid_right = st.columns([1.05, 1.05, 1.2], gap="medium")
with mid_left:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    panel_header("Risk Distribution", "Share of countries currently grouped into each severity band.")
    st.plotly_chart(make_distribution_chart(predictions, theme), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with mid_mid:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    panel_header("Top Severity Signals", "Countries with the strongest near-term outbreak risk score.")
    st.plotly_chart(make_risk_signal_chart(predictions, theme), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with mid_right:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    if selected_country == "All":
        panel_header("Predicted Burden Leaders", "Countries with the largest predicted 7-day case totals.")
        st.plotly_chart(make_top_burden_chart(predictions, theme), use_container_width=True)
    else:
        panel_header(f"{selected_country} Mobility Snapshot", "Lagged movement signals that may influence transmission dynamics.")
        mobility_cols = [col for col in mobility.columns if col.endswith("_lag7")]
        mobility_row = (
            mobility[mobility["country"] == selected_country]
            .melt(value_vars=mobility_cols, var_name="feature", value_name="value")
            .dropna()
        )
        if mobility_row.empty:
            st.info("No lagged mobility snapshot is available for this country.")
        else:
            st.plotly_chart(make_mobility_chart(mobility_row, theme), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if selected_country != "All" and not selected_prediction.empty:
    row = selected_prediction.iloc[0]
    drill_cols = st.columns(4, gap="medium")
    drill_cols[0].markdown(
        kpi("Risk Label", str(row["risk_label"]).upper(), f"Severity score {row['risk_score']:.2f}"),
        unsafe_allow_html=True,
    )
    drill_cols[1].markdown(
        kpi("Predicted Cases (7d)", format_big_number(float(row["predicted_cases_7d"])), f"{row['predicted_cases_per_100k_7d']:.2f} per 100k"),
        unsafe_allow_html=True,
    )
    drill_cols[2].markdown(
        kpi("Growth Rate", f"{float(row['predicted_growth_rate']):.2f}", "Forecasted burden relative to recent 7-day average"),
        unsafe_allow_html=True,
    )
    drill_cols[3].markdown(
        kpi("Recent Cases /100k", f"{float(row['recent_cases_per_100k_7d']):.2f}", "Recent observed incidence level"),
        unsafe_allow_html=True,
    )

table_view = display_predictions.copy()
if selected_country != "All":
    table_view = table_view[table_view["country"] == selected_country]
if sort_mode == "Highest risk":
    table_view = table_view.sort_values(["risk_score", "predicted_cases_7d"], ascending=[False, False])
elif sort_mode == "Highest predicted cases":
    table_view = table_view.sort_values("predicted_cases_7d", ascending=False)
else:
    table_view = table_view.sort_values("country")

st.markdown('<div class="panel-card">', unsafe_allow_html=True)
panel_header("Prediction Summary by Country", "Tabular view of the latest forecast, burden, and severity signals.")
st.dataframe(
    table_view,
    use_container_width=True,
    hide_index=True,
    column_config={
        "date": st.column_config.DateColumn("Date"),
        "country": st.column_config.TextColumn("Country"),
        "iso_code": st.column_config.TextColumn("ISO-3"),
        "predicted_cases_7d": st.column_config.NumberColumn("Predicted Cases (7d)", format="%.0f"),
        "predicted_cases_per_100k_7d": st.column_config.NumberColumn("Predicted Cases /100k", format="%.2f"),
        "predicted_growth_rate": st.column_config.NumberColumn("Growth Rate", format="%.2f"),
        "risk_score": st.column_config.NumberColumn("Risk Score", format="%.2f"),
        "risk_label": st.column_config.TextColumn("Risk Label"),
        "recent_cases_7d_avg": st.column_config.NumberColumn("Recent 7d Avg", format="%.1f"),
        "recent_cases_per_100k_7d": st.column_config.NumberColumn("Recent Cases /100k", format="%.2f"),
    },
)
st.markdown("</div>", unsafe_allow_html=True)
