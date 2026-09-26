"""Forecast the next 24 hours for every zone from a chosen 'as-of' hour."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import models as M
from .panel import Panel


def load_model() -> M.Forecaster:
    return M.Forecaster.load(C.ARTIFACTS_DIR / "model")


def forecast(panel: Panel, model: M.Forecaster, as_of) -> pd.DataFrame:
    """Rows: zone_id, horizon 1..24, target_time, pred (LightGBM for the modelled zones, the
    4-week historical average for the sparse tail), ha4 baseline, and the actual where known."""
    pos = panel.pos(as_of)
    X, meta = F.build_dataset(panel, [pos], require_target=False)
    meta["pred"] = model.predict(X)
    meta["ha4"] = meta["b_ha4"]
    parts = [meta[["zone_id", "horizon", "target_time", "pred", "ha4", "y"]]]
    if panel.tail_D.shape[1]:
        t = pos + np.arange(1, C.HORIZON + 1)
        t = t[t < len(panel.hours)]
        ha = F.same_slot_mean(panel.tail_D, t)
        actual = np.where((t < panel.n_real)[:, None], panel.tail_D[t], np.nan)
        tail = pd.DataFrame({
            "zone_id": np.tile(panel.tail_zones, len(t)),
            "horizon": np.repeat(np.arange(1, len(t) + 1), len(panel.tail_zones)),
            "target_time": np.repeat(panel.hours[t].to_numpy(), len(panel.tail_zones)),
            "pred": ha.ravel(), "ha4": ha.ravel(), "y": actual.ravel()})
        parts.append(tail)
    out = pd.concat(parts, ignore_index=True)
    out["zone_id"] = out["zone_id"].astype(int)
    return out


def recent_history(panel: Panel, zone_id: int, as_of, hours: int = 48) -> pd.Series:
    pos = panel.pos(as_of)
    if zone_id in set(panel.zones.tolist()):
        col = list(panel.zones).index(zone_id)
        series = panel.D[max(pos - hours + 1, 0): pos + 1, col]
    else:
        col = list(panel.tail_zones).index(zone_id)
        series = panel.tail_D[max(pos - hours + 1, 0): pos + 1, col]
    return pd.Series(series, index=panel.hours[max(pos - hours + 1, 0): pos + 1])
