from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .data import load_all_datasets
from .features import build_modeling_table
from .modeling import train_models, write_metrics


def run_pipeline(root_dir: Path) -> dict[str, str]:
    config = ProjectConfig.from_root(root_dir)
    config.ensure_output_dirs()

    bundle = load_all_datasets(config)
    modeling_table = build_modeling_table(bundle, config)
    artifacts = train_models(modeling_table, config)

    modeling_table.to_csv(config.processed_dir / "modeling_table.csv", index=False)
    artifacts.latest_predictions.to_csv(
        config.predictions_dir / "latest_country_predictions.csv",
        index=False,
    )
    write_metrics(config.metrics_dir / "forecast_metrics.json", artifacts.forecast_metrics)
    write_metrics(config.metrics_dir / "risk_metrics.json", artifacts.risk_metrics)

    summary = pd.DataFrame(
        [
            {"metric": "countries", "value": modeling_table["country"].nunique()},
            {"metric": "dates", "value": modeling_table["date"].nunique()},
            {"metric": "rows", "value": len(modeling_table)},
            {"metric": "features_used", "value": len(artifacts.feature_columns)},
        ]
    )
    summary.to_csv(config.metrics_dir / "pipeline_summary.csv", index=False)

    return {
        "modeling_table": str(config.processed_dir / "modeling_table.csv"),
        "predictions": str(config.predictions_dir / "latest_country_predictions.csv"),
        "forecast_metrics": str(config.metrics_dir / "forecast_metrics.json"),
        "risk_metrics": str(config.metrics_dir / "risk_metrics.json"),
    }
