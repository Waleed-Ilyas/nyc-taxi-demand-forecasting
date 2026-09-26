"""Forecast metrics: error by zone tier / horizon / hour, and the operational top-k hit rate."""
from __future__ import annotations

import numpy as np
import pandas as pd

MAPE_MIN_COUNT = 5  # MAPE is only defined on cells with at least this many actual pickups
HORIZON_BUCKETS = {"1-3h": (1, 3), "4-6h": (4, 6), "7-12h": (7, 12), "13-24h": (13, 24)}


def point_metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    mask = y >= MAPE_MIN_COUNT
    return {
        "n": int(len(y)),
        "mape_pct": float(np.mean(np.abs(err[mask]) / y[mask]) * 100) if mask.any() else float("nan"),
        "mape_coverage_pct": float(mask.mean() * 100),
        "wape_pct": float(np.abs(err).sum() / max(y.sum(), 1e-9) * 100),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
    }


def grouped(df: pd.DataFrame, pred_col: str, by: str) -> dict:
    return {str(k): point_metrics(g["y"], g[pred_col]) for k, g in df.groupby(by, observed=True)}


def horizon_buckets(df: pd.DataFrame, pred_col: str) -> dict:
    out = {}
    for name, (lo, hi) in HORIZON_BUCKETS.items():
        g = df[(df["horizon"] >= lo) & (df["horizon"] <= hi)]
        out[name] = point_metrics(g["y"], g[pred_col])
    return out


def topk_hit_rate(df: pd.DataFrame, pred_col: str, k: int = 10, window: int = 3) -> float:
    """Operational KPI: at each origin, of the k zones with the highest *predicted* pickups over the
    next `window` hours, what share are among the k zones with the highest *actual* pickups?"""
    d = df[df["horizon"] <= window]
    agg = d.groupby(["origin", "zone_id"], observed=True).agg(y=("y", "sum"), p=(pred_col, "sum"))
    hits = []
    for _, g in agg.groupby(level=0):
        top_p = set(g.nlargest(k, "p").index.get_level_values(1))
        top_y = set(g.nlargest(k, "y").index.get_level_values(1))
        hits.append(len(top_p & top_y) / k)
    return float(np.mean(hits))
