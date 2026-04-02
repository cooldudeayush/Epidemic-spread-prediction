from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score

from .config import ProjectConfig
from .risk import attach_risk_context


@dataclass
class ModelingArtifacts:
    forecast_model: Ridge
    imputer: SimpleImputer
    feature_columns: list[str]
    forecast_metrics: dict[str, float]
    risk_metrics: dict[str, float]
    latest_predictions: pd.DataFrame


def _feature_columns(table: pd.DataFrame) -> list[str]:
    blocked = {
        "country",
        "date",
        "risk_label",
        "risk_score_future",
        "target_cases_7d",
        "target_growth_rate",
        "target_cases_per_million_7d",
        "target_cases_per_100k_7d",
        "target_recent_cases_per_100k_7d",
        "target_risk_score",
        "target_risk_label",
    }
    numeric_cols = [col for col in table.columns if col not in blocked and pd.api.types.is_numeric_dtype(table[col])]
    cols = [col for col in numeric_cols if col != "confirmed_cumulative"]
    return [col for col in cols if not table[col].isna().all()]


def _time_split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = table["date"].quantile(0.85)
    train = table[table["date"] <= cutoff].copy()
    test = table[table["date"] > cutoff].copy()
    return train, test


def _prepare_xy(
    table: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    imputer: SimpleImputer | None = None,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer]:
    frame = table[feature_columns].copy()
    if imputer is None:
        imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        x = imputer.fit_transform(frame)
    else:
        x = imputer.transform(frame)
    y = table[target_column].to_numpy()
    return x, y, imputer


def train_models(table: pd.DataFrame, config: ProjectConfig) -> ModelingArtifacts:
    usable = table.dropna(subset=["target_cases_7d", "risk_label"]).copy()
    feature_columns = _feature_columns(usable)

    train, test = _time_split(usable)
    if len(train) < config.min_training_rows or test.empty:
        raise ValueError("Not enough training data after preprocessing to fit the models.")

    x_train, y_train, imputer = _prepare_xy(train, feature_columns, "target_cases_7d")
    x_test, y_test, _ = _prepare_xy(test, feature_columns, "target_cases_7d", imputer=imputer)

    forecast_model = Ridge(alpha=1.0)
    forecast_model.fit(x_train, np.log1p(y_train))
    forecast_preds = np.expm1(forecast_model.predict(x_test)).clip(min=0)

    forecast_metrics = {
        "mae": float(mean_absolute_error(y_test, forecast_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, forecast_preds))),
    }

    risk_eval = test.copy()
    risk_eval["predicted_cases_7d"] = forecast_preds
    risk_eval["predicted_growth_rate"] = risk_eval["predicted_cases_7d"] / (
        risk_eval["rolling_cases_mean_7"].replace(0, np.nan) * config.forecast_horizon_days
    )
    risk_eval = attach_risk_context(
        risk_eval,
        cases_col="predicted_cases_7d",
        growth_col="predicted_growth_rate",
        prefix="predicted_",
    )
    actual_labels = test["target_risk_label"].fillna("low")
    predicted_labels = risk_eval["predicted_risk_label"].fillna("low")
    risk_metrics = {
        "accuracy": float(accuracy_score(actual_labels, predicted_labels)),
        "precision_weighted": float(precision_score(actual_labels, predicted_labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(actual_labels, predicted_labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(actual_labels, predicted_labels, average="weighted", zero_division=0)),
    }

    latest_rows = (
        table.sort_values(["country", "date"])
        .groupby("country", as_index=False)
        .tail(1)
        .copy()
    )
    latest_rows["target_cases_7d"] = latest_rows["target_cases_7d"].fillna(0)
    latest_x, _, _ = _prepare_xy(latest_rows, feature_columns, "target_cases_7d", imputer=imputer)
    latest_rows["predicted_cases_7d"] = np.expm1(forecast_model.predict(latest_x)).clip(min=0)
    latest_rows["predicted_growth_rate"] = latest_rows["predicted_cases_7d"] / (
        latest_rows["rolling_cases_mean_7"].replace(0, np.nan) * config.forecast_horizon_days
    )
    latest_rows = attach_risk_context(
        latest_rows,
        cases_col="predicted_cases_7d",
        growth_col="predicted_growth_rate",
        prefix="predicted_",
    )
    latest_rows["risk_score"] = latest_rows["predicted_risk_score"]
    latest_rows["risk_label"] = latest_rows["predicted_risk_label"]
    latest_predictions = latest_rows[
        [
            "date",
            "country",
            "iso_code",
            "predicted_cases_7d",
            "predicted_cases_per_100k_7d",
            "predicted_growth_rate",
            "risk_score",
            "risk_label",
            "rolling_cases_mean_7",
            "predicted_recent_cases_per_100k_7d",
        ]
    ].rename(
        columns={
            "rolling_cases_mean_7": "recent_cases_7d_avg",
            "predicted_recent_cases_per_100k_7d": "recent_cases_per_100k_7d",
        }
    )

    return ModelingArtifacts(
        forecast_model=forecast_model,
        imputer=imputer,
        feature_columns=feature_columns,
        forecast_metrics=forecast_metrics,
        risk_metrics=risk_metrics,
        latest_predictions=latest_predictions.sort_values("predicted_cases_7d", ascending=False),
    )


def write_metrics(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
