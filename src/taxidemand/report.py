"""Generate report figures.  Usage: python -m taxidemand.report"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C
from . import panel as P
from . import viz


def main() -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    F = C.FIGURES_DIR
    metrics = json.loads((C.ARTIFACTS_DIR / "metrics.json").read_text())
    panel = P.panel_from_artifacts()
    pred = pd.read_parquet(C.PROCESSED_DIR / "test_predictions.parquet")
    viz.plot_patterns(panel, F / "01_demand_patterns.png")
    viz.plot_zone_map(panel, C.ARTIFACTS_DIR / "zones.geojson", F / "02_zone_map.png")
    viz.plot_error_by_horizon_tier(metrics["test"], F / "03_error_horizon_tier.png")
    viz.plot_holiday_week(pred, F / "04_holiday_week.png")
    viz.plot_shap(metrics["shap_top"], F / "05_shap.png")
    print("figures written to", F)


if __name__ == "__main__":
    main()
