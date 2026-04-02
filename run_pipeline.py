from __future__ import annotations

from pathlib import Path

from src.epidemic_prediction.pipeline import run_pipeline


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    outputs = run_pipeline(root)
    print("Pipeline completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")
