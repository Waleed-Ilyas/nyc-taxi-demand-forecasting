"""Features for direct multi-horizon zone-hour demand forecasting.

One row = (forecast origin t, zone z, horizon h in 1..24). Known at the origin: recent demand of the
zone and of the whole city (lags / rolling means) and how unusual it is versus the typical demand
for that hour of the week. Known in advance for the target hour t+h: the clock, holidays and (in
production) the weather forecast; in training the observed weather at t+h stands in for the
forecast - a mild optimistic bias documented in the README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .panel import Panel, holiday_flags

LAGS = (0, 1, 2, 3, 6, 12, 24, 48, 168)
ROLLS = (3, 6, 24, 168)
MIN_ORIGIN = 336  # two weeks of history before the first usable origin
WEATHER_TARGET = ["temperature_2m", "apparent_temperature", "precipitation", "snowfall",
                  "wind_speed_10m", "cloud_cover", "relative_humidity_2m"]
WEEK, N_WEEKS = 168, 4
_PAD = WEEK * N_WEEKS


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """a[t - k] with NaN where t - k < 0 (works on (T,) and (T, Z) arrays)."""
    a = a.astype(np.float32)
    if k == 0:
        return a
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def _roll_mean(a: np.ndarray, w: int) -> np.ndarray:
    """Mean of a[t-w+1 .. t] over the past only; NaN for the first w-1 rows."""
    c = np.cumsum(np.nan_to_num(a, nan=0.0).astype(np.float64), axis=0)
    out = np.full(a.shape, np.nan, dtype=np.float32)
    lead = np.zeros((1,) + a.shape[1:])
    out[w - 1:] = (c[w - 1:] - np.concatenate([lead, c[:-w]])) / w
    return out


def _roll_max(a: np.ndarray, w: int) -> np.ndarray:
    out = a.astype(np.float32).copy()
    for k in range(1, w):
        out = np.fmax(out, _shift(a, k))
    return out


def same_slot_mean(D: np.ndarray, t: np.ndarray, weeks: int = N_WEEKS) -> np.ndarray:
    """Historical-average baseline: mean demand at the same hour-of-week over the last `weeks`
    weeks *before* hour t (t - 168, t - 336, ...). Uses only hours < t."""
    padded = np.concatenate([np.full((_PAD,) + D.shape[1:], np.nan, dtype=np.float32), D])
    stack = np.stack([padded[t + _PAD - WEEK * m] for m in range(1, weeks + 1)])
    with np.errstate(all="ignore"):
        return np.nanmean(stack, axis=0)


def origin_block(panel: Panel) -> dict[str, np.ndarray]:
    """Everything known at each origin hour t, as arrays over the full time axis."""
    D, city = panel.D, panel.city
    b: dict[str, np.ndarray] = {f"y_lag{lag}": _shift(D, lag) for lag in LAGS}
    for w in ROLLS:
        b[f"y_mean{w}"] = _roll_mean(D, w)
    b["y_trend"] = (b["y_mean3"] + 1) / (b["y_mean24"] + 1)
    b["y_std24"] = np.sqrt(np.maximum(_roll_mean(np.nan_to_num(D) ** 2, 24) - b["y_mean24"] ** 2, 0))
    b["y_max24"] = _roll_max(D, 24)
    ha_all = same_slot_mean(D, np.arange(D.shape[0]))
    b["now_vs_ha4"] = (D + 1) / (ha_all + 1)  # >1: busier than usual for this hour of the week
    b["last3h_vs_ha4"] = (b["y_mean3"] + 1) / (_roll_mean(ha_all, 3) + 1)
    for lag in (0, 24, 168):
        b[f"city_lag{lag}"] = _shift(city, lag)
    b["city_mean3"] = _roll_mean(city, 3)
    b["city_mean24"] = _roll_mean(city, 24)
    b["city_trend"] = (b["city_mean3"] + 1) / (b["city_mean24"] + 1)
    city_ha = same_slot_mean(city, np.arange(len(city)))
    b["city_now_vs_ha4"] = (city + 1) / (city_ha + 1)
    b["zone_share_now"] = D / (city[:, None] + 1)
    return b


def target_features(panel: Panel) -> pd.DataFrame:
    """Clock / holiday / weather columns per hour (used at the target time)."""
    h = panel.hours
    f = pd.DataFrame(index=h)
    f["hour"] = h.hour
    f["dow"] = h.dayofweek
    f["is_weekend"] = (h.dayofweek >= 5).astype(int)
    f["hour_sin"] = np.sin(2 * np.pi * h.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * h.hour / 24)
    f = f.join(holiday_flags(h))
    for c in WEATHER_TARGET:
        f[f"{c}_tgt"] = panel.weather[c].to_numpy()
    rain = panel.weather["precipitation"].to_numpy()
    f["precip_next3h"] = pd.Series(rain[::-1]).rolling(3, min_periods=1).sum().to_numpy()[::-1]
    return f


def train_hours(panel: Panel) -> int:
    return int((panel.hours <= pd.Timestamp(C.TRAIN_END)).sum())


def build_dataset(
    panel: Panel, origins: np.ndarray, horizons=range(1, C.HORIZON + 1), require_target: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(X, meta). `origins` are integer positions on panel.hours. meta carries the baselines."""
    origins = np.asarray(origins)
    base = origin_block(panel)
    tf = target_features(panel)
    T, Z = panel.D.shape
    zones = panel.zones
    zm = panel.zone_meta.loc[zones]
    static = {
        "zone_id": zones.astype(np.float32),
        "borough": zm["borough_code"].to_numpy(np.float32),
        "service_zone": zm["service_zone_code"].to_numpy(np.float32),
        "is_airport": zm["is_airport"].to_numpy(np.float32),
        "zone_log_volume": np.log1p(zm["train_trips"].to_numpy(np.float32) / train_hours(panel)),
    }
    D_pad = np.concatenate([np.full((_PAD, Z), np.nan, dtype=np.float32), panel.D])
    parts_x, parts_m = [], []
    for h in horizons:
        t = origins + h
        keep = t < T
        o, t = origins[keep], t[keep]
        cols: dict[str, np.ndarray] = {}
        for name, arr in base.items():
            a = arr[o]
            cols[name] = (np.repeat(a[:, None], Z, axis=1) if a.ndim == 1 else a).ravel()
        for name, v in static.items():
            cols[name] = np.tile(v, len(o))
        cols["horizon"] = np.full(len(o) * Z, h, dtype=np.float32)
        for name in tf.columns:
            cols[name] = np.repeat(tf[name].to_numpy(np.float32)[t], Z)
        # references for the target hour - all observable at the origin because h <= 24 < 168
        y_day, y_week = D_pad[t + _PAD - 24], D_pad[t + _PAD - WEEK]
        ha4 = same_slot_mean(panel.D, t)
        weeks = np.stack([D_pad[t + _PAD - WEEK * m] for m in range(1, N_WEEKS + 1)])
        with np.errstate(all="ignore"):
            med4, sd4 = np.nanmedian(weeks, axis=0), np.nanstd(weeks, axis=0)
        cols["ref_day"], cols["ref_week"], cols["ref_ha4"] = y_day.ravel(), y_week.ravel(), ha4.ravel()
        for m in range(2, N_WEEKS + 1):
            cols[f"ref_week{m}"] = weeks[m - 1].ravel()
        # the median is robust to a holiday / storm week contaminating the 4-week average
        cols["ref_med4"], cols["ref_cv4"] = med4.ravel(), (sd4 / (ha4 + 1)).ravel()
        y = np.where((t < panel.n_real)[:, None], panel.D[t], np.nan).ravel()
        meta = pd.DataFrame({
            "origin": np.repeat(panel.hours[o].to_numpy(), Z),
            "target_time": np.repeat(panel.hours[t].to_numpy(), Z),
            "zone_id": np.tile(zones, len(o)), "horizon": np.int16(h), "y": y,
            "b_day": y_day.ravel(), "b_week": y_week.ravel(), "b_ha4": ha4.ravel(),
        })
        parts_x.append(pd.DataFrame(cols).astype("float32"))
        parts_m.append(meta)
    X = pd.concat(parts_x, ignore_index=True)
    meta = pd.concat(parts_m, ignore_index=True)
    if require_target:
        ok = meta["y"].notna().to_numpy()
        X, meta = X[ok].reset_index(drop=True), meta[ok].reset_index(drop=True)
    return X, meta
