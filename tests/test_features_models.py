import numpy as np
import pandas as pd

from taxidemand import evaluate as E
from taxidemand import features as F
from taxidemand import inference
from taxidemand import models as M


def _origins(panel, lo=336, step=6):
    return np.arange(lo, panel.n_real - 24, step)


def test_targets_and_baselines_are_aligned(panel):
    X, meta = F.build_dataset(panel, _origins(panel), horizons=[1, 12, 24])
    all_nan = X.columns[X.isna().all()].tolist()
    assert len(X) == len(meta) and all_nan == ["ref_week4"]  # only 4 weeks of synthetic history
    zi = {z: i for i, z in enumerate(panel.zones)}
    for k in np.random.default_rng(1).integers(0, len(meta), 40):
        r = meta.iloc[k]
        ti = panel.pos(r.target_time)
        assert r.y == panel.D[ti, zi[r.zone_id]]
        assert r.b_week == panel.D[ti - 168, zi[r.zone_id]] and r.b_day == panel.D[ti - 24, zi[r.zone_id]]
        assert r.target_time - r.origin == pd.Timedelta(hours=int(r.horizon))


def test_features_do_not_see_the_future(panel):
    """Changing demand *after* the origin must not change any feature of that origin."""
    origin = np.array([24 * 7 * 3])
    base, _ = F.build_dataset(panel, origin, require_target=False)
    tampered = type(panel)(**{**panel.__dict__, "D": panel.D.copy(), "city": panel.city.copy()})
    tampered.D[origin[0] + 1:] += 500
    tampered.city[origin[0] + 1:] += 5000
    changed, _ = F.build_dataset(tampered, origin, require_target=False)
    ref_cols = [c for c in base.columns if c.startswith(("ref_", "y_", "city_", "now_", "last3h", "zone_share"))]
    # the horizon-specific reference for hours <= origin never changes; y_lag/means only use the past
    pd.testing.assert_frame_equal(base[[c for c in ref_cols if not c.startswith("ref_")]],
                                  changed[[c for c in ref_cols if not c.startswith("ref_")]])


def test_same_slot_mean_uses_only_earlier_weeks(panel):
    t = np.array([24 * 7 * 3 + 5])
    got = F.same_slot_mean(panel.D, t)
    expect = np.mean([panel.D[t[0] - 168 * m] for m in (1, 2, 3)], axis=0)  # only 3 full weeks before
    # 4th week is before the start of the data -> ignored (NaN-mean over the available weeks)
    assert np.allclose(got[0], expect)


def test_dataset_beyond_the_data_has_no_target(panel):
    origin = np.array([panel.n_real - 1])
    X, meta = F.build_dataset(panel, origin, require_target=False)
    assert len(meta) == 24 * len(panel.zones) and meta["y"].isna().all()


def test_forecaster_learns_the_daily_pattern_and_roundtrips(panel, tmp_path):
    X, meta = F.build_dataset(panel, _origins(panel, step=3), horizons=range(1, 25))
    f = M.Forecaster("logratio", "l1", {"n_estimators": 60, "min_child_samples": 20}).fit(X, meta["y"].to_numpy())
    pred = f.predict(X)
    assert (pred >= 0).all()
    assert E.point_metrics(meta["y"], pred)["wape_pct"] < E.point_metrics(meta["y"], meta["b_day"])["wape_pct"]
    f.save(tmp_path)
    np.testing.assert_allclose(M.Forecaster.load(tmp_path).predict(X), pred)


def test_inference_returns_every_zone_and_horizon(panel, tmp_path):
    X, meta = F.build_dataset(panel, _origins(panel, step=6), horizons=range(1, 25))
    f = M.Forecaster("logratio", "regression", {"n_estimators": 20, "min_child_samples": 20}).fit(
        X, meta["y"].to_numpy())
    out = inference.forecast(panel, f, panel.hours[24 * 7 * 3])
    assert set(out["zone_id"]) == set(panel.zones.tolist()) | set(panel.tail_zones.tolist())
    assert out.groupby("zone_id")["horizon"].nunique().eq(24).all()
    assert out["pred"].notna().all() and out["y"].notna().all()  # backtest: actuals are known


def test_metrics_and_topk():
    y = np.array([10, 20, 0, 40.0])
    assert E.point_metrics(y, y)["mape_pct"] == 0
    m = E.point_metrics(y, y * 1.5)
    assert abs(m["wape_pct"] - 50) < 1e-9 and m["mape_coverage_pct"] == 75
    df = pd.DataFrame({"origin": [0] * 4, "zone_id": [1, 2, 3, 4], "horizon": [1] * 4,
                       "y": [50, 40, 30, 5], "p": [45, 42, 10, 35]})
    assert E.topk_hit_rate(df, "p", k=2, window=3) == 1.0
    assert E.topk_hit_rate(df, "p", k=3, window=3) == 2 / 3
