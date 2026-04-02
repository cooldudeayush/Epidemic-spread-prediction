from __future__ import annotations

import numpy as np
import pandas as pd


COUNTRY_ISO3_OVERRIDES = {
    "Kosovo": "XKX",
    "Timor": "TLS",
    "Micronesia": "FSM",
    "North Korea": "PRK",
    "Laos": "LAO",
    "South Korea": "KOR",
    "Russia": "RUS",
    "United States": "USA",
    "United Kingdom": "GBR",
    "Iran": "IRN",
    "Syria": "SYR",
    "Venezuela": "VEN",
    "Bolivia": "BOL",
    "Moldova": "MDA",
    "Tanzania": "TZA",
    "Vietnam": "VNM",
    "Brunei": "BRN",
    "Congo": "COG",
    "Democratic Republic of Congo": "COD",
    "Palestine": "PSE",
    "Taiwan": "TWN",
    "Myanmar": "MMR",
    "Cape Verde": "CPV",
    "Vatican": "VAT",
}


NON_MAP_ENTITIES = {
    "Diamond Princess",
    "MS Zaandam",
    "Summer Olympics 2020",
    "Winter Olympics 2022",
    "Antarctica",
}


def standardize_iso3(code: object, country: object) -> str | None:
    country_name = str(country).strip() if pd.notna(country) else None
    if country_name in NON_MAP_ENTITIES:
        return None
    if country_name in COUNTRY_ISO3_OVERRIDES:
        return COUNTRY_ISO3_OVERRIDES[country_name]

    if pd.isna(code):
        return None

    value = str(code).strip().upper()
    if not value:
        return None
    if value.startswith("OWID_"):
        return COUNTRY_ISO3_OVERRIDES.get(country_name)
    if len(value) == 3 and value.isalpha():
        return value
    return COUNTRY_ISO3_OVERRIDES.get(country_name)


def _piecewise_score(values: pd.Series, xp: list[float], fp: list[float]) -> pd.Series:
    filled = values.fillna(0).astype(float).clip(lower=min(xp), upper=max(xp))
    return pd.Series(np.interp(filled, xp, fp), index=values.index)


def _per_100k(cases: pd.Series, population: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=cases.index, dtype=float)
    valid = population.fillna(0) > 0
    result.loc[valid] = cases.loc[valid] / population.loc[valid] * 100_000
    return result


def attach_risk_context(
    frame: pd.DataFrame,
    cases_col: str,
    growth_col: str,
    prefix: str = "",
) -> pd.DataFrame:
    enriched = frame.copy()
    future_cases = enriched[cases_col].fillna(0).clip(lower=0)
    population = enriched.get("population", pd.Series(np.nan, index=enriched.index, dtype=float)).astype(float)
    recent_weekly_cases = enriched["rolling_cases_mean_7"].fillna(0).clip(lower=0) * 7

    future_per_100k = _per_100k(future_cases, population)
    recent_per_100k = _per_100k(recent_weekly_cases, population)

    if future_per_100k.isna().all():
        future_scale = _piecewise_score(future_cases, [0, 50, 500, 5_000, 20_000], [0, 20, 45, 75, 100])
    else:
        future_scale = _piecewise_score(future_per_100k.fillna(0), [0, 5, 20, 50, 100], [0, 20, 45, 75, 100])

    if recent_per_100k.isna().all():
        recent_scale = _piecewise_score(recent_weekly_cases, [0, 50, 500, 5_000, 20_000], [0, 20, 45, 75, 100])
    else:
        recent_scale = _piecewise_score(recent_per_100k.fillna(0), [0, 5, 20, 50, 100], [0, 20, 45, 75, 100])

    growth = enriched[growth_col].replace([np.inf, -np.inf], np.nan).fillna(0)
    growth_scale = _piecewise_score(growth, [0.5, 0.8, 1.0, 1.2, 1.6], [0, 15, 40, 70, 100])

    positive_rate = enriched.get("positive_rate", pd.Series(0, index=enriched.index, dtype=float)).fillna(0)
    positive_scale = _piecewise_score(positive_rate, [0.0, 0.02, 0.05, 0.1, 0.2], [0, 10, 35, 75, 100])

    fully_vaccinated = enriched.get(
        "people_fully_vaccinated_per_hundred",
        pd.Series(0, index=enriched.index, dtype=float),
    ).fillna(0)
    vaccination_protection = _piecewise_score(fully_vaccinated, [0, 20, 40, 60, 80], [0, 0.05, 0.10, 0.18, 0.28])

    base_severity = (
        0.30 * future_scale
        + 0.40 * recent_scale
        + 0.20 * growth_scale
        + 0.10 * positive_scale
    )
    severity = (base_severity * (1 - vaccination_protection)).clip(lower=0, upper=100)

    labels = pd.cut(
        severity,
        bins=[-np.inf, 20, 45, np.inf],
        labels=["low", "medium", "high"],
    ).astype("object")

    enriched[f"{prefix}cases_per_100k_7d"] = future_per_100k
    enriched[f"{prefix}recent_cases_per_100k_7d"] = recent_per_100k
    enriched[f"{prefix}risk_score"] = severity.round(2)
    enriched[f"{prefix}risk_label"] = labels
    return enriched
