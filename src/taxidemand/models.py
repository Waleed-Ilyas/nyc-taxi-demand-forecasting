"""LightGBM global demand model (one model per horizon bucket, shared by all zones)."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C

BUCKETS = [(1, 3), (4, 6), (7, 12), (13, 24)]

DEFAULT_PARAMS = {
    "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
}


def bucket_of(h: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array([hi for _, hi in BUCKETS]), h, side="left")


class Forecaster:
    """Predicts log(1+demand) relative to the historical-average baseline for that hour-of-week.

    `mode="logratio"`: target = log1p(y) - log1p(ha4)  (scale-free, so busy and quiet zones train
    together); prediction = expm1(model + log1p(ha4)).
    `mode="poisson"`: raw counts with a Poisson objective (used only as a comparison).
    """

    def __init__(self, mode: str = "logratio", objective: str = "regression",
                 params: dict | None = None):
        self.mode, self.objective = mode, objective
        self.params = DEFAULT_PARAMS | (params or {})
        self.models: dict[int, lgb.Booster] = {}
        self.feature_names: list[str] = []

    def _target(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        if self.mode == "logratio":
            return np.log1p(y) - np.log1p(X["ref_ha4"].to_numpy(float))
        return y

    def fit(self, X: pd.DataFrame, y: np.ndarray, verbose: bool = False) -> Forecaster:
        self.feature_names = list(X.columns)
        target = self._target(X, np.asarray(y, float))
        bucket = bucket_of(X["horizon"].to_numpy())
        objective = "poisson" if self.mode == "poisson" else self.objective
        for b in range(len(BUCKETS)):
            rows = bucket == b
            reg = lgb.LGBMRegressor(**self.params, objective=objective, random_state=C.SEED,
                                    verbose=-1, n_jobs=-1)
            reg.fit(X[rows], target[rows])
            self.models[b] = reg.booster_
            if verbose:
                print(f"  bucket {BUCKETS[b]} fitted on {rows.sum():,} rows")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names]
        bucket = bucket_of(X["horizon"].to_numpy())
        out = np.zeros(len(X))
        for b, booster in self.models.items():
            rows = bucket == b
            if rows.any():
                out[rows] = booster.predict(X[rows])
        if self.mode == "logratio":
            out = np.expm1(out + np.log1p(X["ref_ha4"].to_numpy(float)))
        return np.clip(out, 0, None)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for b, booster in self.models.items():
            booster.save_model(str(directory / f"lgbm_bucket{b}.txt"))
        meta = {"mode": self.mode, "features": self.feature_names, "params": self.params,
                "objective": self.objective}
        (directory / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, directory: Path) -> Forecaster:
        meta = json.loads((directory / "meta.json").read_text())
        obj = cls(meta["mode"], meta["objective"], meta["params"])
        obj.feature_names = meta["features"]
        for b in range(len(BUCKETS)):
            obj.models[b] = lgb.Booster(model_file=str(directory / f"lgbm_bucket{b}.txt"))
        return obj
