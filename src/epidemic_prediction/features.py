from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data import DatasetBundle, build_iso_lookup
from .risk import attach_risk_context


def _add_case_features(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("country", group_keys=False)
    frame["lag_cases_1"] = grouped["new_cases"].shift(1)
    frame["lag_cases_7"] = grouped["new_cases"].shift(7)
    frame["lag_cases_14"] = grouped["new_cases"].shift(14)
    frame["rolling_cases_mean_7"] = grouped["new_cases"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    frame["rolling_cases_mean_14"] = grouped["new_cases"].transform(lambda s: s.rolling(14, min_periods=1).mean())
    frame["rolling_cases_std_14"] = grouped["new_cases"].transform(lambda s: s.rolling(14, min_periods=2).std())
    frame["cases_growth_7"] = frame["rolling_cases_mean_7"] / frame["lag_cases_7"].replace(0, np.nan)
    frame["cases_growth_14"] = frame["rolling_cases_mean_7"] / frame["lag_cases_14"].replace(0, np.nan)
    frame["cases_per_million_proxy"] = frame["rolling_cases_mean_7"]
    return frame


def _add_mobility_features(frame: pd.DataFrame) -> pd.DataFrame:
    mobility_columns = [
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]
    grouped = frame.groupby("country", group_keys=False)
    for col in mobility_columns:
        if col not in frame.columns:
            continue
        frame[f"{col}_lag7"] = grouped[col].shift(7)
        frame[f"{col}_lag14"] = grouped[col].shift(14)
    return frame


def _add_targets(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    grouped = frame.groupby("country", group_keys=False)
    forward_sum = grouped["new_cases"].transform(
        lambda s: s.iloc[::-1].shift(1).rolling(horizon, min_periods=horizon).sum().iloc[::-1]
    )
    frame["target_cases_7d"] = forward_sum
    frame["target_growth_rate"] = frame["target_cases_7d"] / (
        frame["rolling_cases_mean_7"].replace(0, np.nan) * horizon
    )
    return frame


def build_modeling_table(bundle: DatasetBundle, config: ProjectConfig) -> pd.DataFrame:
    cases = bundle.cases.copy()
    iso_lookup = build_iso_lookup(cases, bundle.mobility, bundle.excess_deaths, bundle.owid)
    table = cases.merge(iso_lookup, on="country", how="left")

    table = table.merge(
        bundle.mobility,
        on=["country", "date"],
        how="left",
        suffixes=("", "_mob"),
    )
    if "iso_code_mob" in table.columns:
        table["iso_code"] = table["iso_code"].fillna(table["iso_code_mob"])
        table = table.drop(columns=["iso_code_mob"])

    table = table.merge(
        bundle.excess_deaths,
        on=["country", "date"],
        how="left",
        suffixes=("", "_excess"),
    )
    if "iso_code_excess" in table.columns:
        table["iso_code"] = table["iso_code"].fillna(table["iso_code_excess"])
        table = table.drop(columns=["iso_code_excess"])

    if not bundle.owid.empty:
        owid_columns = [
            "country",
            "date",
            "iso_code",
            "population",
            "new_deaths",
            "new_tests",
            "people_vaccinated_per_hundred",
            "people_fully_vaccinated_per_hundred",
            "total_boosters_per_hundred",
            "hosp_patients",
            "icu_patients",
            "stringency_index",
            "positive_rate",
            "reproduction_rate",
        ]
        available_cols = [col for col in owid_columns if col in bundle.owid.columns]
        table = table.merge(
            bundle.owid[available_cols],
            on=["country", "date"],
            how="left",
            suffixes=("", "_owid"),
        )
        if "iso_code_owid" in table.columns:
            table["iso_code"] = table["iso_code"].fillna(table["iso_code_owid"])
            table = table.drop(columns=["iso_code_owid"])

    table["population"] = table.get("population", pd.Series(index=table.index, dtype=float))
    table["population"] = table.groupby("country")["population"].transform(lambda s: s.ffill().bfill())
    if "people_fully_vaccinated_per_hundred" in table.columns:
        table["people_fully_vaccinated_per_hundred"] = (
            table.groupby("country")["people_fully_vaccinated_per_hundred"].transform(lambda s: s.ffill().fillna(0))
        )
    if "positive_rate" in table.columns:
        table["positive_rate"] = table.groupby("country")["positive_rate"].transform(lambda s: s.ffill().fillna(0))

    table = table.sort_values(["country", "date"]).reset_index(drop=True)
    table = _add_case_features(table)
    table = _add_mobility_features(table)
    table = _add_targets(table, config.forecast_horizon_days)
    table = attach_risk_context(table, cases_col="target_cases_7d", growth_col="target_growth_rate", prefix="target_")
    table["risk_score"] = table["target_risk_score"]
    table["risk_label"] = table["target_risk_label"]
    table["target_cases_per_million_7d"] = table["target_cases_per_100k_7d"] * 10
    table["risk_score_future"] = table["target_risk_score"]

    feature_fill_zero = [
        "excess_deaths_per_100k",
        "confirmed_deaths_per_100k",
        "new_deaths",
        "new_tests",
        "people_vaccinated_per_hundred",
        "people_fully_vaccinated_per_hundred",
        "total_boosters_per_hundred",
        "hosp_patients",
        "icu_patients",
        "stringency_index",
        "positive_rate",
        "reproduction_rate",
    ]
    for col in feature_fill_zero:
        if col in table.columns:
            table[col] = table[col].fillna(0)

    table["country"] = table["country"].astype(str)
    return table
