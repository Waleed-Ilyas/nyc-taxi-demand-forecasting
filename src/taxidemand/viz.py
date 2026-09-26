"""Report figures (matplotlib, dark/gold theme matching the portfolio site)."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

from . import config as C
from .panel import Panel, holiday_flags

BG, PANEL, IVORY, GOLD, BRONZE, CYAN, RED = (
    "#0A0A0C", "#15110F", "#F2EDE4", "#D9B26A", "#8B5E3C", "#4FD1C5", "#ff5a4f",
)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": "#3a332d", "axes.labelcolor": IVORY, "text.color": IVORY,
        "xtick.color": IVORY, "ytick.color": IVORY, "grid.color": "#2a2521", "grid.linewidth": 0.8,
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "legend.frameon": False,
        "figure.dpi": 110, "font.family": "DejaVu Sans",
    })


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_patterns(panel: Panel, path) -> None:
    """Hour-of-week heatmap + daily demand ratio with anomalies annotated."""
    apply_style()
    real = panel.hours[:panel.n_real]
    city = pd.Series(panel.city[:panel.n_real], index=real)
    fig, ax = plt.subplots(2, 1, figsize=(13, 8.2), gridspec_kw={"height_ratios": [1, 1.1]})
    how = city.groupby([real.dayofweek, real.hour]).mean().unstack()
    im = ax[0].imshow(how.values, aspect="auto", cmap=LinearSegmentedColormap.from_list(
        "g", [PANEL, BRONZE, GOLD, "#F8E7B9"]))
    ax[0].set_yticks(range(7), DAYS)
    ax[0].set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax[0].set_title("City-wide pickups per hour, by hour of week (evening peak, quiet 3-5 am)")
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], label="pickups / hour")
    daily = city.resample("D").sum()
    hol = holiday_flags(panel.hours).is_holiday.resample("D").max().reindex(daily.index)
    typical = daily[hol == 0].groupby(daily[hol == 0].index.dayofweek).median()
    ratio = daily / daily.index.dayofweek.map(typical).to_numpy()
    snow = panel.weather["snowfall"].resample("D").sum().reindex(daily.index)
    ax[1].bar(ratio.index, ratio.values, color=np.where(hol == 1, CYAN, GOLD), width=0.9)
    ax[1].bar(snow.index, snow.values / 20, color="white", alpha=0.5, width=0.9, label="snowfall (cm / 20)")
    ax[1].axhline(1, color=IVORY, lw=1, ls=":")
    ax[1].grid(False, axis="x")
    ax[1].set_ylabel("daily pickups vs typical for that weekday")
    ax[1].set_title("Anomalies: blizzards (Jan 25, Feb 23) and holidays (cyan) cut demand by 25-75%")
    for day in ratio[(ratio < 0.5)].index:
        ax[1].annotate(f"{day:%b %d}: {ratio[day]:.2f}x", (day, ratio[day]), xytext=(4, 14),
                       textcoords="offset points", fontsize=9, color=RED)
    ax[1].legend(loc="upper right")
    _save(fig, path)


def plot_zone_map(panel: Panel, geojson_path, path) -> None:
    """Static choropleth of average hourly pickups per zone (log colour)."""
    apply_style()
    gj = json.loads(open(geojson_path).read())
    hist = pd.read_parquet(C.ARTIFACTS_DIR / "history_hourly.parquet")
    mean = hist.mean()
    polys, vals = [], []
    for f in gj["features"]:
        g = f["geometry"]
        rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]
        for ring in rings:
            polys.append(np.array(ring))
            vals.append(mean.get(str(f["properties"]["LocationID"]), 0.0))
    fig, ax = plt.subplots(figsize=(8, 9.5))
    pc = PolyCollection(polys, array=np.log10(np.array(vals) + 0.05), edgecolors="#2a2521",
                        clim=(np.log10(0.06), np.log10(400)),
                        linewidths=0.3, cmap=LinearSegmentedColormap.from_list(
                            "g", [PANEL, BRONZE, GOLD, "#F8E7B9"]))
    ax.add_collection(pc)
    ax.set_xlim(-74.28, -73.68)
    ax.set_ylim(40.48, 40.93)
    ax.set_aspect(1 / np.cos(np.deg2rad(40.7)))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(pc, ax=ax, shrink=0.6)
    cb.set_ticks([np.log10(x + 0.05) for x in (0.1, 1, 10, 100, 300)])
    cb.set_ticklabels(["0.1", "1", "10", "100", "300"])
    cb.set_label("average pickups per hour (log)")
    ax.set_title("Demand is extremely concentrated: 20 zones = 57% of pickups")
    _save(fig, path)


def plot_error_by_horizon_tier(test: dict, path) -> None:
    apply_style()
    names = {"lightgbm": ("LightGBM", GOLD), "historical_avg_4wk": ("4-week same-hour avg", BRONZE),
             "same_hour_last_week": ("Same hour last week", CYAN),
             "same_hour_yesterday": ("Same hour yesterday", "#7d7368")}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for axis, key, title in ((ax[0], "by_horizon", "Error by forecast horizon"),
                             (ax[1], "by_tier", "Error by zone tier (volume)")):
        groups = list(test[key]["lightgbm"])
        w = 0.2
        for i, (m, (label, color)) in enumerate(names.items()):
            vals = [test[key][m][g]["wape_pct"] for g in groups]
            bars = axis.bar(np.arange(len(groups)) + (i - 1.5) * w, vals, w, color=color, label=label)
            axis.bar_label(bars, fmt="%.0f", fontsize=7, color=IVORY, padding=2)
        axis.set_xticks(range(len(groups)), groups)
        axis.set_ylabel("WAPE % (lower is better)")
        axis.set_title(title + " - July 2026 hold-out")
    ax[0].legend(ncol=2, loc="upper left")
    _save(fig, path)


def plot_holiday_week(pred: pd.DataFrame, path) -> None:
    """City total: 12-hour-ahead forecast vs actual around July 4th (a holiday the 4-week average misses)."""
    apply_style()
    d = pred[pred["horizon"] == 12]
    city = d.groupby("target_time")[["y", "lgbm", "b_ha4"]].sum()
    w = city.loc["2026-06-30":"2026-07-08"]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(w.index, w["y"], color=CYAN, lw=1.8, label="Actual")
    ax.plot(w.index, w["lgbm"], color=GOLD, lw=2.2, label="LightGBM (12 h ahead)")
    ax.plot(w.index, w["b_ha4"], color=BRONZE, lw=1.5, ls="--", label="4-week same-hour average")
    for day, label in (("2026-07-03", "Fri Jul 3\n(holiday observed)"), ("2026-07-04", "Sat Jul 4"),
                       ("2026-07-05", "Sun Jul 5\n(39 mm rain)")):
        ax.axvspan(pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1), color=RED, alpha=0.07)
        ax.text(pd.Timestamp(day) + pd.Timedelta(hours=12), ax.get_ylim()[1] * 0.97, label, ha="center",
                va="top", fontsize=9, color=RED)
    ax.set_ylabel("city-wide pickups per hour")
    ax.set_title("Holiday week: the model learns to expect a quieter city, the 4-week average does not")
    ax.legend(loc="lower left", ncol=3)
    _save(fig, path)


def plot_shap(shap_top: dict, path) -> None:
    apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for axis, (name, vals) in zip(ax, shap_top.items(), strict=True):
        s = pd.Series(vals).sort_values()
        axis.barh(s.index, s.values, color=GOLD)
        axis.set_title(f"Drivers, horizon {name.replace('bucket_', '')}")
        axis.set_xlabel("mean |SHAP| (log-ratio vs 4-week average)")
    _save(fig, path)
