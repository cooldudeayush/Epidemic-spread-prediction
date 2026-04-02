from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    root_dir: Path
    jhu_confirmed_path: Path
    mobility_path: Path
    excess_deaths_path: Path
    owid_path: Path
    outputs_dir: Path
    processed_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    forecast_horizon_days: int = 7
    min_training_rows: int = 500

    @classmethod
    def from_root(cls, root_dir: Path) -> "ProjectConfig":
        outputs_dir = root_dir / "outputs"
        return cls(
            root_dir=root_dir,
            jhu_confirmed_path=root_dir / "time_series_covid19_confirmed_global.csv",
            mobility_path=root_dir / "Global_Mobility_Report.csv",
            excess_deaths_path=root_dir
            / "estimated-cumulative-excess-deaths-per-100000-people-during-covid-19.csv",
            owid_path=root_dir / "owid-covid-data.csv",
            outputs_dir=outputs_dir,
            processed_dir=outputs_dir / "processed",
            predictions_dir=outputs_dir / "predictions",
            metrics_dir=outputs_dir / "metrics",
        )

    def ensure_output_dirs(self) -> None:
        for path in (
            self.outputs_dir,
            self.processed_dir,
            self.predictions_dir,
            self.metrics_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
