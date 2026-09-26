import duckdb
import pandas as pd

from taxidemand import ingest
from taxidemand import panel as P


def _write_trips(tmp_path):
    rows = [
        # (pickup, dropoff, zone, total, distance)
        ("2026-01-05 10:05:00", "2026-01-05 10:20:00", 161, 12.5, 2.0),   # valid
        ("2026-01-05 10:40:00", "2026-01-05 10:55:00", 161, 9.0, 0.0),    # valid, zero distance kept
        ("2026-01-05 11:10:00", "2026-01-05 11:30:00", 237, 15.0, 3.0),   # valid
        ("2026-01-05 11:15:00", "2026-01-05 11:10:00", 237, 15.0, 3.0),   # dropoff before pickup
        ("2026-01-05 11:20:00", "2026-01-05 11:40:00", 264, 15.0, 3.0),   # unknown zone
        ("2026-01-05 11:25:00", "2026-01-05 11:45:00", 237, -7.0, 1.0),   # refund
        ("2008-12-31 23:00:00", "2008-12-31 23:20:00", 237, 5.0, 1.0),    # bad timestamp
    ]
    df = pd.DataFrame(rows, columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID",
                                     "total_amount", "trip_distance"])
    for c in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
        df[c] = pd.to_datetime(df[c])
    path = tmp_path / "yellow_tripdata_2026-01.parquet"
    df.to_parquet(path)
    return str(path).replace("\\", "/")


def test_quality_report_counts_first_failing_rule(tmp_path):
    glob = _write_trips(tmp_path)
    rep = ingest.quality_report(duckdb.connect(), glob)
    reasons = {k[0]: v for k, v in rep["reasons"].items()}
    assert rep["raw_trips"] == 7 and rep["valid_trips"] == 3
    assert reasons == {"1": 1, "2": 1, "3": 1, "4": 1, "5": 3}
    assert sum(rep["reasons"].values()) == rep["raw_trips"]
    assert rep["valid_zero_distance_trips_kept"] == 1


def test_zone_hour_aggregation(tmp_path):
    zh = ingest.build_zone_hour(duckdb.connect(), _write_trips(tmp_path))
    got = {(int(r.zone_id), r.hour.hour): int(r.trips) for r in zh.itertuples()}
    assert got == {(161, 10): 2, (237, 11): 1}


def test_local_hours_drop_the_nonexistent_dst_hour():
    idx = P.local_hours("2026-03-08 00:00", "2026-03-08 05:00")
    assert pd.Timestamp("2026-03-08 02:00") not in idx and len(idx) == 5


def test_holiday_flags_cover_observed_independence_day():
    hours = pd.date_range("2026-07-02", "2026-07-06", freq="h")
    f = P.holiday_flags(hours)
    assert f.loc["2026-07-03", "is_holiday"].all()  # July 4 is a Saturday: observed on Friday
    assert f.loc["2026-07-02", "is_holiday_eve"].all() and f.loc["2026-07-06", "is_holiday"].sum() == 0
