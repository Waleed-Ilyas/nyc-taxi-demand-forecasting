"""Download TLC trip files (parquet), aggregate to zone-hour demand with DuckDB SQL, fetch weather.

Usage:  python -m taxidemand.ingest

Source: NYC Taxi & Limousine Commission Trip Record Data (public domain, NYC Open Data terms);
weather: Open-Meteo Historical Weather API (CC BY 4.0).
"""
from __future__ import annotations

import json
import time

import duckdb
import pandas as pd
import requests

from . import config as C

VALID_TRIPS_SQL = """
    tpep_pickup_datetime >= TIMESTAMP '{start}' AND tpep_pickup_datetime < TIMESTAMP '{end_excl}'
    AND PULocationID IS NOT NULL AND PULocationID NOT IN {unknown}
    AND tpep_dropoff_datetime > tpep_pickup_datetime
    AND total_amount >= 0
"""

# First failing rule wins, so the counts add up to the raw total.
REASON_SQL = """
    CASE
      WHEN tpep_pickup_datetime < TIMESTAMP '{start}' OR tpep_pickup_datetime >= TIMESTAMP '{end_excl}'
        THEN '1 pickup time outside the file period (bad timestamp)'
      WHEN PULocationID IS NULL OR PULocationID IN {unknown} THEN '2 unknown pickup zone (264/265)'
      WHEN tpep_dropoff_datetime <= tpep_pickup_datetime THEN '3 dropoff not after pickup'
      WHEN total_amount < 0 THEN '4 negative total amount (refund / dispute)'
      ELSE '5 valid trip'
    END
"""


def _params() -> dict:
    end_excl = (pd.Timestamp(C.DATA_END) + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    return {"start": C.DATA_START, "end_excl": end_excl, "unknown": str(C.UNKNOWN_ZONES)}


def download(force: bool = False) -> None:
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = [(f"trip-data/yellow_tripdata_{m}.parquet", f"yellow_tripdata_{m}.parquet")
             for m in C.MONTHS]
    files += [("misc/taxi_zone_lookup.csv", "taxi_zone_lookup.csv"),
              ("misc/taxi_zones.zip", "taxi_zones.zip")]
    for remote, local in files:
        path = C.RAW_DIR / local
        if path.exists() and not force:
            continue
        with requests.get(f"{C.TLC_BASE}/{remote}", stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)


def trips_glob() -> str:
    return str(C.RAW_DIR / "yellow_tripdata_2026-*.parquet").replace("\\", "/")


def quality_report(con: duckdb.DuckDBPyConnection | None = None, glob: str | None = None) -> dict:
    """One SQL pass over all raw trips: why each row is (not) usable + a few sanity stats."""
    con = con or duckdb.connect()
    glob = glob or trips_glob()
    p = _params()
    reasons = con.execute(f"""
        SELECT {REASON_SQL.format(**p)} AS reason, count(*) AS trips
        FROM read_parquet('{glob}') GROUP BY 1 ORDER BY 1""").df()
    raw = int(reasons["trips"].sum())
    extra = con.execute(f"""
        SELECT count(*) AS n, sum((trip_distance <= 0)::int) AS zero_distance,
               sum((trip_distance > 100)::int) AS distance_over_100mi
        FROM read_parquet('{glob}') WHERE {VALID_TRIPS_SQL.format(**p)}""").df().iloc[0]
    return {
        "raw_trips": raw,
        "reasons": {r.reason: int(r.trips) for r in reasons.itertuples()},
        "valid_trips": int(reasons[reasons.reason.str.startswith("5")]["trips"].iloc[0]),
        "valid_zero_distance_trips_kept": int(extra["zero_distance"]),
        "valid_trips_over_100mi_kept": int(extra["distance_over_100mi"]),
    }


def build_zone_hour(con: duckdb.DuckDBPyConnection | None = None, glob: str | None = None) -> pd.DataFrame:
    """Pickups per (zone, hour) - the demand series. Only non-empty cells; the grid is completed later."""
    con = con or duckdb.connect()
    glob = glob or trips_glob()
    p = _params()
    df = con.execute(f"""
        SELECT PULocationID AS zone_id, date_trunc('hour', tpep_pickup_datetime) AS hour,
               count(*)::INTEGER AS trips
        FROM read_parquet('{glob}')
        WHERE {VALID_TRIPS_SQL.format(**p)}
        GROUP BY 1, 2 ORDER BY 1, 2""").df()
    df["zone_id"] = df["zone_id"].astype("int16")
    return df


def fetch_weather() -> pd.DataFrame:
    params = {"latitude": C.LAT, "longitude": C.LON, "hourly": ",".join(C.WEATHER_VARS),
              "timezone": C.TIMEZONE, "start_date": C.DATA_START,
              # one extra day: the "weather forecast" for the day after the last trip data
              "end_date": (pd.Timestamp(C.DATA_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")}
    last = None
    for attempt in range(4):
        try:
            r = requests.get(C.WEATHER_URL, params=params, timeout=60)
            r.raise_for_status()
            df = pd.DataFrame(r.json()["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            return df.set_index("time").sort_index()
        except requests.RequestException as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError("weather download failed") from last


def run() -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.ARTIFACTS_DIR.mkdir(exist_ok=True)
    download()
    con = duckdb.connect()
    report = quality_report(con)
    (C.ARTIFACTS_DIR / "quality_report.json").write_text(json.dumps(report, indent=1))
    zh = build_zone_hour(con)
    zh.to_parquet(C.PROCESSED_DIR / "zone_hour.parquet")
    fetch_weather().to_parquet(C.PROCESSED_DIR / "weather.parquet")
    print(json.dumps(report, indent=1))
    print(f"zone-hour cells: {len(zh):,}, zones: {zh.zone_id.nunique()}, trips: {zh.trips.sum():,}")


if __name__ == "__main__":
    run()
