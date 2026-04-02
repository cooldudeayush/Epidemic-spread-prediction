from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ProjectConfig


JHU_RENAMES = {
    "US": "United States",
    "Korea, South": "South Korea",
    "Taiwan*": "Taiwan",
    "West Bank and Gaza": "Palestine",
    "Congo (Kinshasa)": "Democratic Republic of Congo",
    "Congo (Brazzaville)": "Congo",
    "Burma": "Myanmar",
    "Cabo Verde": "Cape Verde",
    "Eswatini": "Swaziland",
    "Holy See": "Vatican",
    "Timor-Leste": "Timor",
}


@dataclass
class DatasetBundle:
    cases: pd.DataFrame
    mobility: pd.DataFrame
    excess_deaths: pd.DataFrame
    owid: pd.DataFrame


def _normalize_country_name(name: str) -> str:
    if pd.isna(name):
        return name
    return JHU_RENAMES.get(str(name).strip(), str(name).strip())


def load_jhu_confirmed(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    date_columns = [col for col in raw.columns if col not in {"Province/State", "Country/Region", "Lat", "Long"}]
    melted = raw.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        value_vars=date_columns,
        var_name="date",
        value_name="confirmed_cumulative",
    )
    melted["date"] = pd.to_datetime(melted["date"], format="%m/%d/%y")
    melted["country"] = melted["Country/Region"].map(_normalize_country_name)
    country_daily = (
        melted.groupby(["country", "date"], as_index=False)["confirmed_cumulative"]
        .sum()
        .sort_values(["country", "date"])
    )
    country_daily["new_cases"] = (
        country_daily.groupby("country")["confirmed_cumulative"].diff().fillna(country_daily["confirmed_cumulative"])
    )
    # Reporting corrections can create negative daily values. Clip to zero for stable modeling.
    country_daily["new_cases"] = country_daily["new_cases"].clip(lower=0)
    country_daily["new_cases_7d_avg"] = (
        country_daily.groupby("country")["new_cases"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
        .astype(float)
    )
    return country_daily


def load_mobility(path: Path) -> pd.DataFrame:
    cols = [
        "country_region_code",
        "country_region",
        "sub_region_1",
        "sub_region_2",
        "date",
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]
    chunks = []
    read_kwargs = {
        "usecols": cols,
        "low_memory": False,
        "chunksize": 500_000,
        "dtype": {
            "country_region_code": "string",
            "country_region": "string",
            "sub_region_1": "string",
            "sub_region_2": "string",
        },
    }
    for chunk in pd.read_csv(path, **read_kwargs):
        top_level = chunk[chunk["sub_region_1"].isna() & chunk["sub_region_2"].isna()].copy()
        if top_level.empty:
            top_level = chunk.copy()
        top_level["date"] = pd.to_datetime(top_level["date"])
        top_level["country"] = top_level["country_region"].map(_normalize_country_name)
        chunks.append(top_level)

    mobility = pd.concat(chunks, ignore_index=True)

    metrics = [
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]
    grouped = (
        mobility.groupby(["country", "country_region_code", "date"], as_index=False)[metrics]
        .mean()
        .sort_values(["country", "date"])
    )
    grouped = grouped.rename(columns={"country_region_code": "iso_code"})
    return grouped


def load_excess_deaths(path: Path) -> pd.DataFrame:
    cols = [
        "Entity",
        "Code",
        "Day",
        "Central estimate",
        "Confirmed COVID-19 deaths (per 100,000)",
    ]
    excess = pd.read_csv(path, usecols=cols)
    excess["date"] = pd.to_datetime(excess["Day"])
    excess["country"] = excess["Entity"].map(_normalize_country_name)
    excess = excess.rename(
        columns={
            "Code": "iso_code",
            "Central estimate": "excess_deaths_per_100k",
            "Confirmed COVID-19 deaths (per 100,000)": "confirmed_deaths_per_100k",
        }
    )
    return excess[["country", "iso_code", "date", "excess_deaths_per_100k", "confirmed_deaths_per_100k"]]


def load_owid(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    candidate_columns = [
        "iso_code",
        "location",
        "date",
        "population",
        "new_cases",
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
    owid = pd.read_csv(path, usecols=lambda c: c in candidate_columns)
    owid["date"] = pd.to_datetime(owid["date"])
    owid["country"] = owid["location"].map(_normalize_country_name)
    return owid


def build_iso_lookup(cases: pd.DataFrame, mobility: pd.DataFrame, excess: pd.DataFrame, owid: pd.DataFrame) -> pd.DataFrame:
    mobility_lookup = mobility[["country", "iso_code"]].dropna().drop_duplicates()
    excess_lookup = excess[["country", "iso_code"]].dropna().drop_duplicates()

    lookups = [mobility_lookup, excess_lookup]
    if not owid.empty and "iso_code" in owid.columns:
        owid_lookup = owid[["country", "iso_code"]].dropna().drop_duplicates()
        lookups.append(owid_lookup)

    combined = pd.concat(lookups, ignore_index=True).drop_duplicates()
    combined = combined.sort_values(["country", "iso_code"]).drop_duplicates("country", keep="first")
    countries = cases[["country"]].drop_duplicates()
    return countries.merge(combined, on="country", how="left")


def load_all_datasets(config: ProjectConfig) -> DatasetBundle:
    return DatasetBundle(
        cases=load_jhu_confirmed(config.jhu_confirmed_path),
        mobility=load_mobility(config.mobility_path),
        excess_deaths=load_excess_deaths(config.excess_deaths_path),
        owid=load_owid(config.owid_path),
    )
