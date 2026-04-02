from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data import DatasetBundle, build_iso_lookup


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


def _add_risk_labels(frame: pd.DataFrame) -> pd.DataFrame:
    cases_per_million = np.where(
        frame["population"].fillna(0) > 0,
        frame["target_cases_7d"] / frame["population"] * 1_000_000,
        np.nan,
    )
    frame["target_cases_per_million_7d"] = cases_per_million
    risk_score = (
        np.log1p(frame["target_cases_7d"].clip(lower=0))
        + np.log1p(np.nan_to_num(cases_per_million, nan=0.0).clip(min=0))
    )
    frame["risk_score_future"] = risk_score
    risk_band = pd.cut(
        risk_score,
        bins=[-np.inf, 3.5, 6.0, np.inf],
        labels=["low", "medium", "high"],
    )
    frame["risk_label"] = risk_band.astype("object")
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
    if table["population"].isna().all():
        # Without OWID we can still train, but per-capita metrics fall back to raw counts.
        table["population"] = np.nan

    table = table.sort_values(["country", "date"]).reset_index(drop=True)
    table = _add_case_features(table)
    table = _add_mobility_features(table)
    table = _add_targets(table, config.forecast_horizon_days)
    table = _add_risk_labels(table)

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
