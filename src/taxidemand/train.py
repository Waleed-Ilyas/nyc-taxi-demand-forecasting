"""End-to-end training / evaluation pipeline with MLflow tracking.

Usage:  python -m taxidemand.train      (needs data from `python -m taxidemand.ingest`)

Protocol (time-ordered, no leakage)
  * train      2026-01-01 .. 2026-05-31   (targets must lie in this window)
  * validation 2026-06                    (random search over hyper-parameters; MAPE)
  * test       2026-07                    (scored once; includes the July 3-4 holiday and a 39 mm
                                           rain day)
  * stress test: the blizzard days (Jan 25-26, Feb 22-23) are held out of training and the model is
    scored on them against the historical-average baseline
  * production model: refit on all data through 2026-07-31 with the chosen hyper-parameters
"""
from __future__ import annotations

import json
import time
import warnings

import mlflow
import numpy as np
import pandas as pd
import shap

from . import config as C
from . import evaluate as E
from . import features as F
from . import models as M
from . import panel as P

N_TRIALS = 6
TRAIN_STEP, TUNE_STEP, TEST_STEP = 4, 8, 2
STORM_DAYS = ["2026-01-25", "2026-01-26", "2026-02-22", "2026-02-23"]


def search_space(rng: np.random.Generator) -> dict:
    n_est, lr = [(300, 0.05), (500, 0.05), (700, 0.03)][rng.integers(3)]
    return {
        "n_estimators": n_est, "learning_rate": lr,
        "num_leaves": int(rng.choice([31, 63, 127])),
        "min_child_samples": int(rng.choice([50, 100, 300])),
        "feature_fraction": float(rng.choice([0.6, 0.8, 1.0])),
        "lambda_l2": float(rng.choice([1.0, 5.0, 20.0])),
        "_objective": str(rng.choice(["regression", "l1"])),
    }


def fit_with(params: dict, X: pd.DataFrame, y: np.ndarray) -> M.Forecaster:
    p = dict(params)
    objective = p.pop("_objective", "regression")
    return M.Forecaster("logratio", objective, p).fit(X, y)


def sets(panel: P.Panel, lo: str, hi: str, step: int, min_origin: int = F.MIN_ORIGIN):
    """Rows whose *target* lies in [lo, hi]; origins every `step` hours."""
    start = max(panel.pos(lo) - C.HORIZON, min_origin)
    origins = np.arange(start, panel.pos(hi) + 1, step)
    X, meta = F.build_dataset(panel, origins)
    keep = ((meta["target_time"] >= pd.Timestamp(lo)) & (meta["target_time"] <= pd.Timestamp(hi))).to_numpy()
    return X[keep].reset_index(drop=True), meta[keep].reset_index(drop=True)


def evaluation_tables(meta: pd.DataFrame, panel: P.Panel, pred_cols: dict[str, str]) -> dict:
    """All breakdowns for the models / baselines named in `pred_cols` (label -> column)."""
    tf = F.target_features(panel)
    m = meta.copy()
    m["tier"] = m["zone_id"].map(panel.zone_meta["tier"])
    m["hour"] = pd.DatetimeIndex(m["target_time"]).hour
    m["is_holiday"] = tf["is_holiday"].reindex(m["target_time"]).to_numpy()
    m["rain"] = (tf["precipitation_tgt"].reindex(m["target_time"]).to_numpy() >= 0.5)
    m["weekend"] = tf["is_weekend"].reindex(m["target_time"]).to_numpy()
    out: dict = {"overall": {}, "by_tier": {}, "by_horizon": {}, "by_hour": {}, "by_condition": {}}
    for label, col in pred_cols.items():
        out["overall"][label] = E.point_metrics(m["y"], m[col])
        out["by_tier"][label] = E.grouped(m, col, "tier")
        out["by_horizon"][label] = E.horizon_buckets(m, col)
        out["by_hour"][label] = {int(h): v["wape_pct"] for h, v in E.grouped(m, col, "hour").items()}
        cond = {}
        for name, mask in {"holiday": m["is_holiday"] == 1, "non_holiday": m["is_holiday"] == 0,
                           "rain_(>=0.5mm/h)": m["rain"], "dry": ~m["rain"],
                           "weekend": m["weekend"] == 1, "weekday": m["weekend"] == 0}.items():
            g = m[mask]
            cond[name] = E.point_metrics(g["y"], g[col]) if len(g) else None
        out["by_condition"][label] = cond
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    C.ARTIFACTS_DIR.mkdir(exist_ok=True)
    (C.ARTIFACTS_DIR / "model").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{C.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("nyc-taxi-demand")
    panel = P.build_panel()
    print(f"panel: {panel.D.shape}, modelled-zone share of trips "
          f"{np.nansum(panel.D) / np.nansum(panel.city) * 100:.2f}%")

    Xtr, mtr = sets(panel, C.DATA_START, C.TRAIN_END, TRAIN_STEP)
    Xva, mva = sets(panel, C.VALID_START, C.VALID_END, TUNE_STEP)
    print(f"train {Xtr.shape}, valid {Xva.shape}  ({time.time() - t0:.0f}s)")

    with mlflow.start_run(run_name="pipeline"):
        mlflow.log_params({"train_end": C.TRAIN_END, "n_model_zones": C.N_MODEL_ZONES,
                           "n_features": Xtr.shape[1], "train_rows": len(Xtr)})

        # ---- 1. hyper-parameter search on the validation month (MAPE) ----
        rng = np.random.default_rng(C.SEED)
        trials = [dict(M.DEFAULT_PARAMS, _objective="regression")] + [search_space(rng) for _ in range(N_TRIALS)]
        sub = np.arange(len(Xtr)) % 2 == 0  # speed: search on half the training rows
        log, best, best_mape = [], None, np.inf
        for i, params in enumerate(trials):
            f = fit_with(params, Xtr[sub], mtr["y"].to_numpy()[sub])
            r = E.point_metrics(mva["y"], f.predict(Xva))
            with mlflow.start_run(run_name=f"trial-{i}", nested=True):
                mlflow.log_params({k: v for k, v in params.items()})
                mlflow.log_metrics({"valid_mape": r["mape_pct"], "valid_wape": r["wape_pct"],
                                    "valid_rmse": r["rmse"]})
            log.append({"trial": i, **params, **{f"valid_{k}": r[k] for k in ("mape_pct", "wape_pct", "rmse")}})
            print(f"trial {i}: MAPE {r['mape_pct']:.2f} WAPE {r['wape_pct']:.2f} RMSE {r['rmse']:.2f}")
            if r["mape_pct"] < best_mape:
                best, best_mape = params, r["mape_pct"]
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})

        # ---- 2. fit on all training rows, score the held-out test month once ----
        model = fit_with(best, Xtr, mtr["y"].to_numpy())
        model.save(C.ARTIFACTS_DIR / "model_holdout")  # out-of-sample model for backtest replays
        Xte, mte = sets(panel, C.TEST_START, C.TEST_END, TEST_STEP)
        mte["lgbm"] = model.predict(Xte)
        pred_cols = {"lightgbm": "lgbm", "historical_avg_4wk": "b_ha4", "same_hour_last_week": "b_week",
                     "same_hour_yesterday": "b_day"}
        tables = evaluation_tables(mte, panel, pred_cols)

        # operational KPI: are the busiest zones the ones we say they are?
        topk = {label: {f"top10_next{w}h": E.topk_hit_rate(mte, col, 10, w) for w in (3, 6)}
                for label, col in pred_cols.items()}
        # city-wide total demand per hour (sum over modelled zones), 1-24h ahead
        city = mte.groupby(["origin", "horizon", "target_time"], observed=True)[
            ["y", "lgbm", "b_ha4", "b_week", "b_day"]].sum().reset_index()
        city_metrics = {label: E.point_metrics(city["y"], city[col]) for label, col in pred_cols.items()}
        # sparse long-tail zones (unmodelled): historical-average baseline only
        tail_rows = []
        tt = np.arange(panel.pos(C.TEST_START), panel.pos(C.TEST_END) + 1)
        if panel.tail_D.shape[1]:
            ha_tail = F.same_slot_mean(panel.tail_D, tt)
            tail_rows = E.point_metrics(panel.tail_D[tt].ravel(), ha_tail.ravel())
        mte.to_parquet(C.PROCESSED_DIR / "test_predictions.parquet")

        # ---- 3. stress test: blizzards held out of training ----
        def on_storm(meta: pd.DataFrame) -> np.ndarray:
            days = [pd.Timestamp(d) for d in STORM_DAYS]
            return (pd.DatetimeIndex(meta["origin"]).normalize().isin(days)
                    | pd.DatetimeIndex(meta["target_time"]).normalize().isin(days))
        storm_tr = on_storm(mtr)
        model_ns = fit_with(best, Xtr[~storm_tr], mtr["y"].to_numpy()[~storm_tr])
        Xs, ms = sets(panel, C.DATA_START, C.TRAIN_END, 1)
        keep = on_storm(ms) & (pd.DatetimeIndex(ms["origin"]).normalize().isin([pd.Timestamp(d) for d in STORM_DAYS]))
        Xs, ms = Xs[keep].reset_index(drop=True), ms[keep].reset_index(drop=True)
        ms["lgbm"] = model_ns.predict(Xs)
        storm = {label: E.point_metrics(ms["y"], ms[col]) for label, col in pred_cols.items()}
        storm["days"] = STORM_DAYS
        storm["rows_evaluated"] = len(ms)

        # ---- 4. explainability ----
        rows = np.random.default_rng(C.SEED).choice(len(Xte), size=min(3000, len(Xte)), replace=False)
        shap_out = {}
        for b in (0, 3):
            sel = rows[M.bucket_of(Xte["horizon"].to_numpy()[rows]) == b]
            sample = Xte.iloc[sel][model.feature_names]
            sv = shap.TreeExplainer(model.models[b]).shap_values(sample)
            shap_out[f"bucket_{M.BUCKETS[b][0]}-{M.BUCKETS[b][1]}h"] = pd.Series(
                np.abs(sv).mean(0), index=sample.columns).sort_values(ascending=False).head(12).round(4).to_dict()

        # ---- 5. production refit on ALL real data ----
        Xall, mall = sets(panel, C.DATA_START, C.DATA_END, TRAIN_STEP)
        final = fit_with(best, Xall, mall["y"].to_numpy())
        final.save(C.ARTIFACTS_DIR / "model")

        # ---- 6. artifacts for the app ----
        wide = np.concatenate([panel.D[:panel.n_real], panel.tail_D[:panel.n_real]], axis=1)
        ids = np.concatenate([panel.zones, panel.tail_zones]).astype(int)
        hist = pd.DataFrame(wide.astype(np.int32), index=panel.hours[:panel.n_real], columns=ids)
        hist.index.name = "hour"
        hist.columns = hist.columns.astype(str)
        hist.to_parquet(C.ARTIFACTS_DIR / "history_hourly.parquet")
        panel.weather.to_parquet(C.ARTIFACTS_DIR / "weather.parquet")
        panel.zone_meta.reset_index().rename(columns={"index": "LocationID"}).to_parquet(
            C.ARTIFACTS_DIR / "zone_meta.parquet")

        n_te_origins = int(mte["origin"].nunique())
        metrics = {
            "data": {"hours": int(panel.n_real), "modelled_zones": int(len(panel.zones)),
                     "tail_zones": int(len(panel.tail_zones)),
                     "modelled_trip_share_pct": float(np.nansum(panel.D) / np.nansum(panel.city) * 100),
                     "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
                     "test_origins": n_te_origins, "test_window": [C.TEST_START, C.TEST_END],
                     "n_features": int(Xtr.shape[1])},
            "search": {"best_params": best, "best_valid_mape": best_mape, "trials": log},
            "test": {**tables, "topk_hit_rate": topk, "city_total": city_metrics,
                     "tail_zones_ha_baseline": tail_rows},
            "storm_stress_test": storm,
            "shap_top": shap_out,
            "runtime_min": (time.time() - t0) / 60,
        }
        (C.ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
        mlflow.log_metrics({
            "test_mape": tables["overall"]["lightgbm"]["mape_pct"],
            "test_wape": tables["overall"]["lightgbm"]["wape_pct"],
            "test_rmse": tables["overall"]["lightgbm"]["rmse"],
            "test_ha4_mape": tables["overall"]["historical_avg_4wk"]["mape_pct"],
            "test_top10_next3h": topk["lightgbm"]["top10_next3h"],
        })
        mlflow.log_artifacts(str(C.ARTIFACTS_DIR / "model"), artifact_path="model")
        mlflow.log_artifact(str(C.ARTIFACTS_DIR / "metrics.json"))
        print(f"done in {(time.time() - t0) / 60:.1f} min")
        for label in pred_cols:
            o = tables["overall"][label]
            print(f"TEST {label:22s} MAPE {o['mape_pct']:.2f}  WAPE {o['wape_pct']:.2f}  RMSE {o['rmse']:.2f}"
                  f"  top10@3h {topk[label]['top10_next3h']:.3f}")
        print("STORM", {k: (round(v["wape_pct"], 1) if isinstance(v, dict) else v) for k, v in storm.items()})


def refit_holdout() -> None:
    """Re-create artifacts/model_holdout from the best parameters stored in metrics.json."""
    warnings.filterwarnings("ignore")
    best = json.loads((C.ARTIFACTS_DIR / "metrics.json").read_text())["search"]["best_params"]
    panel = P.build_panel()
    Xtr, mtr = sets(panel, C.DATA_START, C.TRAIN_END, TRAIN_STEP)
    fit_with(best, Xtr, mtr["y"].to_numpy()).save(C.ARTIFACTS_DIR / "model_holdout")


if __name__ == "__main__":
    import sys

    refit_holdout() if sys.argv[1:] == ["holdout"] else main()
