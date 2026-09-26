import numpy as np
import pandas as pd
import pytest

from taxidemand import config as C
from taxidemand.panel import Panel

N_WEEKS_DATA = 4  # 4 weeks of hourly history + 24 future hours


@pytest.fixture(scope="session")
def panel() -> Panel:
    """Synthetic panel: 5 modelled zones with a strong daily/weekly pattern, 2 tail zones."""
    rng = np.random.default_rng(0)
    real = pd.date_range("2026-03-02", periods=24 * 7 * N_WEEKS_DATA, freq="h")  # Mon 00:00
    hours = real.append(pd.date_range(real[-1] + pd.Timedelta(hours=1), periods=24, freq="h"))
    hod, dow = real.hour.to_numpy(), real.dayofweek.to_numpy()
    base = 50 + 40 * np.sin(2 * np.pi * (hod - 8) / 24) + 15 * (dow >= 5)
    scale = np.array([4.0, 2.5, 1.5, 1.0, 0.6])
    D = np.clip(base[:, None] * scale[None, :] + rng.normal(0, 3, (len(real), 5)), 0, None)
    D = np.rint(D).astype(np.float32)
    tail = np.rint(np.clip(rng.normal(0.4, 0.5, (len(real), 2)), 0, None)).astype(np.float32)
    pad = lambda a: np.vstack([a, np.full((24, a.shape[1]), np.nan, dtype=np.float32)])  # noqa: E731
    weather = pd.DataFrame({"temperature_2m": 10.0, "apparent_temperature": 8.0, "precipitation": 0.0,
                            "rain": 0.0, "snowfall": 0.0, "wind_speed_10m": 12.0, "cloud_cover": 50.0,
                            "relative_humidity_2m": 60.0}, index=hours)
    zones = np.array([161, 237, 132, 79, 5])
    meta = pd.DataFrame({
        "Borough": ["Manhattan", "Manhattan", "Queens", "Manhattan", "Staten Island", "Bronx", "Bronx"],
        "Zone": ["Midtown", "UES", "JFK Airport", "East Village", "Arden", "T1", "T2"],
        "service_zone": ["Yellow Zone"] * 7, "train_trips": [4e4, 2.5e4, 1.5e4, 1e4, 6e3, 50, 40],
        "rank": [1, 2, 3, 4, 5, 6, 7],
        "tier": ["A (top 20)"] * 3 + ["B (21-50)", "C (51-150)"] + ["Tail (unmodelled)"] * 2,
        "is_airport": [0, 0, 1, 0, 0, 0, 0], "borough_code": [2, 2, 3, 2, 4, 0, 0],
        "service_zone_code": [1] * 7}, index=pd.Index([161, 237, 132, 79, 5, 900, 901], name="LocationID"))
    return Panel(hours=hours, zones=zones, D=pad(D), city=np.concatenate([D.sum(1) + tail.sum(1),
                 np.full(24, np.nan)]).astype(np.float32), weather=weather, zone_meta=meta,
                 tail_D=pad(tail), tail_zones=np.array([900, 901]), n_real=len(real))


@pytest.fixture(autouse=True)
def _short_train_window(monkeypatch):
    """Tests use a tiny calendar; make 'training hours' cover the whole synthetic history."""
    monkeypatch.setattr(C, "TRAIN_END", "2026-03-29 23:00")
