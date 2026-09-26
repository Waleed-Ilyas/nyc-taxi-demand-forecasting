"""The demand panel: a (hours x zones) matrix of pickups plus weather, holidays and zone metadata."""
from __future__ import annotations

from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Panel:
    hours: pd.DatetimeIndex  # local NYC time, tz-naive; includes `n_future` hours after the data
    zones: np.ndarray  # zone ids of the *modelled* zones (columns of D)
    D: np.ndarray  # (T, Z) float32 pickups per hour; NaN in the future rows
    city: np.ndarray  # (T,) total pickups over ALL zones (incl. the sparse tail); NaN in the future
    weather: pd.DataFrame  # aligned to `hours`
    zone_meta: pd.DataFrame  # index = zone id; borough, zone name, tier, ...
    tail_D: np.ndarray  # (T, Ztail) pickups of the non-modelled tail zones
    tail_zones: np.ndarray
    n_real: int  # number of hours with real trip data

    def pos(self, ts) -> int:
        return int(self.hours.get_loc(pd.Timestamp(ts)))


def local_hours(start: str, end: str) -> pd.DatetimeIndex:
    """Hourly NYC wall-clock timestamps, dropping the hour that does not exist at DST start."""
    idx = pd.date_range(start, end, freq="h")
    ok = pd.Series(idx).dt.tz_localize(C.TIMEZONE, nonexistent="NaT", ambiguous="NaT").notna()
    return idx[ok.to_numpy()]


def holiday_flags(hours: pd.DatetimeIndex) -> pd.DataFrame:
    cal = holidays.country_holidays("US", subdiv="NY", years=sorted({h.year for h in hours}))
    days = hours.normalize()
    is_h = days.isin(pd.to_datetime(list(cal.keys())))
    tomorrow = (days + pd.Timedelta(days=1)).isin(pd.to_datetime(list(cal.keys())))
    yesterday = (days - pd.Timedelta(days=1)).isin(pd.to_datetime(list(cal.keys())))
    return pd.DataFrame({"is_holiday": is_h.astype(int), "is_holiday_eve": tomorrow.astype(int),
                         "is_day_after_holiday": yesterday.astype(int)}, index=hours)


def build_panel(n_future: int = C.HORIZON, n_model_zones: int = C.N_MODEL_ZONES) -> Panel:
    zh = pd.read_parquet(C.PROCESSED_DIR / "zone_hour.parquet")
    weather = pd.read_parquet(C.PROCESSED_DIR / "weather.parquet")
    lookup = pd.read_csv(C.RAW_DIR / "taxi_zone_lookup.csv").set_index("LocationID")

    real = local_hours(C.DATA_START, C.DATA_END)
    future = local_hours(pd.Timestamp(C.DATA_END) + pd.Timedelta(hours=1),
                         pd.Timestamp(C.DATA_END) + pd.Timedelta(hours=n_future + 2))[:n_future]
    hours = real.append(future)

    wide = zh.pivot(index="hour", columns="zone_id", values="trips").reindex(real).fillna(0.0)
    train_mask = real <= pd.Timestamp(C.TRAIN_END)
    ranking = wide[train_mask].sum().sort_values(ascending=False)  # rank on TRAIN volume only
    zones = ranking.index[:n_model_zones].to_numpy()
    tail = ranking.index[n_model_zones:].to_numpy()

    def pad(a: np.ndarray) -> np.ndarray:
        return np.vstack([a, np.full((len(future), a.shape[1]), np.nan, dtype=np.float32)])

    D = pad(wide[zones].to_numpy(np.float32))
    tail_D = pad(wide[tail].to_numpy(np.float32)) if len(tail) else np.zeros((len(hours), 0))
    city = np.concatenate([wide.sum(axis=1).to_numpy(np.float32), np.full(len(future), np.nan)])

    w = weather.reindex(hours)
    w = w.ffill().bfill()  # a couple of edge hours around DST; weather has no gaps otherwise

    meta = lookup.loc[lookup.index.isin(ranking.index)].copy()
    meta["train_trips"] = ranking.reindex(meta.index)
    order = ranking.index.tolist()
    meta["rank"] = [order.index(z) + 1 for z in meta.index]
    meta["tier"] = np.select([meta["rank"] <= 20, meta["rank"] <= 50, meta["rank"] <= n_model_zones],
                             ["A (top 20)", "B (21-50)", "C (51-150)"], "Tail (unmodelled)")
    meta["is_airport"] = meta["Zone"].fillna("").str.contains("Airport").astype(int)
    meta["borough_code"] = meta["Borough"].astype("category").cat.codes
    meta["service_zone_code"] = meta["service_zone"].astype("category").cat.codes
    return Panel(hours=hours, zones=zones, D=D, city=city, weather=w, zone_meta=meta, tail_D=tail_D,
                 tail_zones=tail, n_real=len(real))


def panel_from_artifacts(n_future: int = C.HORIZON) -> Panel:
    """Rebuild the panel from the small committed artifacts only (no raw TLC files needed).

    Used by the app / inference: history_hourly.parquet (all zones), weather.parquet (covers the
    next `n_future` hours after the data as well) and zone_meta.parquet.
    """
    hist = pd.read_parquet(C.ARTIFACTS_DIR / "history_hourly.parquet")
    weather = pd.read_parquet(C.ARTIFACTS_DIR / "weather.parquet")
    meta = pd.read_parquet(C.ARTIFACTS_DIR / "zone_meta.parquet").set_index("LocationID")
    hist.columns = hist.columns.astype(int)
    real = pd.DatetimeIndex(hist.index)
    future = weather.index[weather.index > real[-1]][:n_future]
    hours = real.append(future)
    model_zones = meta.index[~meta["tier"].str.startswith("Tail")]
    model_zones = meta.loc[model_zones].sort_values("rank").index.to_numpy()
    tail = meta.index[meta["tier"].str.startswith("Tail")].to_numpy()

    def pad(a: np.ndarray) -> np.ndarray:
        return np.vstack([a, np.full((len(future), a.shape[1]), np.nan, dtype=np.float32)])

    return Panel(
        hours=hours, zones=model_zones, D=pad(hist[model_zones].to_numpy(np.float32)),
        city=np.concatenate([hist.sum(axis=1).to_numpy(np.float32), np.full(len(future), np.nan)]),
        weather=weather.reindex(hours).ffill().bfill(), zone_meta=meta,
        tail_D=pad(hist[tail].to_numpy(np.float32)), tail_zones=tail, n_real=len(real))
