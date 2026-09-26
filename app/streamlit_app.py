"""NYC Taxi Demand Forecast - operations dashboard (Streamlit)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taxidemand import config as C  # noqa: E402
from taxidemand import inference, panel  # noqa: E402
from taxidemand import models as M  # noqa: E402

ART = C.ARTIFACTS_DIR
GOLD, IVORY, PANEL_BG, CYAN, BRONZE = "#D9B26A", "#F2EDE4", "#15110F", "#4FD1C5", "#8B5E3C"

st.set_page_config(page_title="NYC Taxi Demand", page_icon="🚕", layout="wide")
st.markdown(
    f"""
    <style>
    .block-container {{padding-top: 2rem; max-width: 1300px;}}
    h1, h2, h3 {{font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.01em;}}
    h1 {{font-weight: 400; font-size: 2.5rem;}}
    .kpi {{background: {PANEL_BG}; border: 1px solid rgba(242,237,228,.08); border-radius: 12px;
          padding: 1rem 1.2rem; height: 100%;}}
    .kpi .label {{font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; opacity: .6;}}
    .kpi .value {{font-size: 1.9rem; font-family: Georgia, serif; line-height: 1.25;}}
    .kpi .sub {{font-size: .85rem; opacity: .75;}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_static():
    return {
        "panel": panel.panel_from_artifacts(),
        "holdout": M.Forecaster.load(ART / "model_holdout"),
        "prod": inference.load_model(),
        "geo": json.loads((ART / "zones.geojson").read_text()),
        "metrics": json.loads((ART / "metrics.json").read_text()),
        "quality": json.loads((ART / "quality_report.json").read_text()),
    }


S = load_static()
P, META = S["panel"], S["panel"].zone_meta
NAME = META["Zone"].to_dict()
BOROUGH = META["Borough"].to_dict()
LAST = P.hours[P.n_real - 1]
HOLDOUT_FIRST = pd.Timestamp(C.VALID_START)


@st.cache_data(show_spinner="Building features and forecasting all zones...")
def get_forecast(as_of: str, use_prod: bool) -> pd.DataFrame:
    model = S["prod"] if use_prod else S["holdout"]
    fc = inference.forecast(P, model, pd.Timestamp(as_of))
    fc["zone"] = fc["zone_id"].map(NAME)
    fc["borough"] = fc["zone_id"].map(BOROUGH)
    return fc


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=IVORY), margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.14, x=0))
    fig.update_xaxes(gridcolor="rgba(242,237,228,.06)")
    fig.update_yaxes(gridcolor="rgba(242,237,228,.06)")
    return fig


# ------------------------------------------------------------------ sidebar: choose the moment
PRESETS = {
    "Typical Friday evening (Fri 17 Jul, 17:00)": "2026-07-17 17:00",
    "Independence Day holiday (Fri 3 Jul, 08:00)": "2026-07-03 08:00",
    "Heavy-rain Sunday (Sun 5 Jul, 12:00)": "2026-07-05 12:00",
    "Latest data - forecast for Sat 1 Aug (Fri 31 Jul, 23:00)": str(LAST),
    "Custom...": None,
}
st.sidebar.markdown("### Forecast moment")
choice = st.sidebar.selectbox("Replay", list(PRESETS))
if PRESETS[choice] is None:
    d = st.sidebar.date_input("Date", pd.Timestamp("2026-07-10"), min_value=HOLDOUT_FIRST.date(),
                              max_value=LAST.date())
    hh = st.sidebar.slider("Hour", 0, 23, 17)
    as_of = pd.Timestamp(d) + pd.Timedelta(hours=hh)
    as_of = min(as_of, LAST)
else:
    as_of = pd.Timestamp(PRESETS[choice])
use_prod = as_of >= LAST - pd.Timedelta(hours=23)
st.sidebar.caption(
    "Production model (trained on all data through 31 Jul). The next 24 h have no actuals yet."
     if use_prod else
     "Out-of-sample replay: model trained only on data through 31 May; June was used for tuning "
     "and July was never seen in training."
)
st.sidebar.markdown("---")
top_n = st.sidebar.slider("Zones in the driver table", 5, 30, 15)

fc = get_forecast(str(as_of), use_prod)
has_actual = fc["y"].notna().any()

st.title("NYC Taxi Demand Forecast")
st.caption("Hourly pickups per taxi zone, next 24 hours · NYC TLC yellow-taxi trips (Jan-Jul 2026) + "
           "weather · LightGBM global model")

city_pred = fc.groupby("horizon")["pred"].sum()
city_ha = fc.groupby("horizon")["ha4"].sum()
peak_h = int(city_pred.idxmax())
c = st.columns(4)
c[0].markdown(kpi("As of", f"{as_of:%a %d %b %H:%M}", "local NYC time"), unsafe_allow_html=True)
c[1].markdown(kpi("Expected pickups, next 24 h", f"{city_pred.sum():,.0f}",
                  f"typical for this slot: {city_ha.sum():,.0f} ({(city_pred.sum() / city_ha.sum() - 1) * 100:+.0f}%)"),
              unsafe_allow_html=True)
c[2].markdown(kpi("City peak", f"{as_of + pd.Timedelta(hours=peak_h):%a %H:%M}",
                  f"{city_pred.max():,.0f} pickups in that hour"), unsafe_allow_html=True)
if has_actual:
    ok = fc[fc["y"].notna()]
    wape = lambda col: np.abs(ok[col] - ok["y"]).sum() / ok["y"].sum() * 100  # noqa: E731
    c[3].markdown(kpi("Error on this replay (WAPE)", f"{wape('pred'):.1f}%",
                      f"4-week average baseline: {wape('ha4'):.1f}%"), unsafe_allow_html=True)
else:
    c[3].markdown(kpi("Error on this replay", "n/a", "future hours: no actuals yet"), unsafe_allow_html=True)
st.markdown("")

tab_map, tab_drv, tab_zone, tab_back, tab_about = st.tabs(
    ["Zone map", "Position drivers", "Zone forecast", "Backtest accuracy", "About"])

# ------------------------------------------------------------------ map
with tab_map:
    ctl, mapcol = st.columns([1, 3])
    win = ctl.radio("Show pickups for", ["Next 1 hour", "Next 3 hours", "Next 6 hours", "Next 24 hours",
                                       "A specific hour"], index=1)
    if win == "A specific hour":
        hz = ctl.slider("Hours ahead", 1, 24, 6)
        d = fc[fc["horizon"] == hz]
    else:
        n = {"Next 1 hour": 1, "Next 3 hours": 3, "Next 6 hours": 6, "Next 24 hours": 24}[win]
        d = fc[fc["horizon"] <= n].groupby(["zone_id", "zone", "borough"], as_index=False)[
            ["pred", "ha4"]].sum()
    metric = ctl.radio("Colour by", ["Expected pickups", "Versus typical (%)"])
    d = d.assign(vs_typical=(d["pred"] + 1) / (d["ha4"] + 1) * 100 - 100)
    color, scale, rng_ = (("pred", [[0, "#15110F"], [0.35, "#8B5E3C"], [1, "#F2D796"]], None)
                          if metric == "Expected pickups" else
                          ("vs_typical", [[0, CYAN], [0.5, "#26221E"], [1, "#ff9f5a"]], (-60, 60)))
    fig = px.choropleth_map(d, geojson=S["geo"], locations="zone_id", featureidkey="properties.LocationID",
                            color=color, color_continuous_scale=scale, range_color=rng_,
                            hover_name="zone", hover_data={"zone_id": False, "pred": ":.0f", "ha4": ":.0f",
                                                           "vs_typical": ":+.0f"},
                            map_style="carto-darkmatter", center={"lat": 40.74, "lon": -73.95}, zoom=10.2,
                            opacity=0.85, labels={"pred": "expected", "ha4": "typical", "vs_typical": "vs typical %"})
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=IVORY))
    mapcol.plotly_chart(fig, width="stretch")
    st.caption("'Typical' = mean pickups in the same hour-of-week over the previous 4 weeks. "
               "Sparse outer-borough zones use that typical value as their forecast.")

# ------------------------------------------------------------------ drivers table
with tab_drv:
    st.markdown("#### Where to position drivers")
    win = st.radio("Horizon", ["Next 3 hours", "Next 6 hours"], horizontal=True)
    n = 3 if win == "Next 3 hours" else 6
    g = fc[fc["horizon"] <= n].groupby(["zone_id", "zone", "borough"], as_index=False).agg(
        expected=("pred", "sum"), typical=("ha4", "sum"), actual=("y", "sum"))
    if not has_actual:
        g["actual"] = np.nan
    g["vs typical"] = (g["expected"] + 1) / (g["typical"] + 1) * 100 - 100
    g = g.sort_values("expected", ascending=False).reset_index(drop=True)
    g.index += 1
    view = g.head(top_n)[["zone", "borough", "expected", "typical", "vs typical", "actual"]]
    st.dataframe(view, width="stretch", column_config={
        "expected": st.column_config.NumberColumn("expected pickups", format="%.0f"),
        "typical": st.column_config.NumberColumn("typical", format="%.0f"),
        "vs typical": st.column_config.NumberColumn("vs typical", format="%+.0f%%"),
        "actual": st.column_config.NumberColumn("actual (replay)", format="%.0f")})
    if has_actual:
        ok = g.dropna(subset=["actual"])
        pred_top = set(ok.nlargest(10, "expected")["zone_id"])
        true_top = set(ok.nlargest(10, "actual")["zone_id"])
        base_top = set(ok.nlargest(10, "typical")["zone_id"])
        st.success(f"Hit rate on this replay: the model's top-10 zones contain {len(pred_top & true_top)} of the "
                   f"10 truly busiest zones (4-week-average baseline: {len(base_top & true_top)}).")
    st.download_button("Download all zones (CSV)", g.to_csv(index=False).encode(),
                       file_name=f"zone_forecast_{as_of:%Y%m%d_%H}.csv", mime="text/csv")

# ------------------------------------------------------------------ zone forecast
with tab_zone:
    order = fc.groupby("zone_id")["pred"].sum().sort_values(ascending=False).index.tolist()
    zid = st.selectbox("Zone", order, format_func=lambda z: f"{NAME.get(z, z)} ({BOROUGH.get(z, '')})")
    hist = inference.recent_history(P, zid, as_of, 48)
    z = fc[fc["zone_id"] == zid].sort_values("horizon")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual (history)", line=dict(color=CYAN, width=2)))
    fig.add_trace(go.Scatter(x=z["target_time"], y=z["ha4"], name="4-week average", line=dict(color=BRONZE, dash="dot")))
    fig.add_trace(go.Scatter(x=z["target_time"], y=z["pred"], name="Forecast", line=dict(color=GOLD, width=3)))
    if has_actual:
        fig.add_trace(go.Scatter(x=z["target_time"], y=z["y"], name="Actual (replay)",
                                 line=dict(color=CYAN, width=2, dash="dash")))
    fig.add_vline(x=as_of, line_dash="dot", line_color=IVORY, opacity=0.5)
    fig.update_yaxes(title="pickups per hour")
    st.plotly_chart(style(fig, 440), width="stretch")

# ------------------------------------------------------------------ backtest
with tab_back:
    m = S["metrics"]
    t = m["test"]
    lgb, ha = t["overall"]["lightgbm"], t["overall"]["historical_avg_4wk"]
    st.markdown(f"#### Held-out test month: {m['data']['test_window'][0][:10]} → {m['data']['test_window'][1][:10]} "
                f"({m['data']['test_origins']} forecast origins × {m['data']['modelled_zones']} zones × 24 h)")
    a, b, c2, d2 = st.columns(4)
    a.markdown(kpi("MAPE (cells ≥ 5 pickups)", f"{lgb['mape_pct']:.1f}%", f"4-week avg {ha['mape_pct']:.1f}%"),
               unsafe_allow_html=True)
    b.markdown(kpi("WAPE (all cells)", f"{lgb['wape_pct']:.1f}%", f"4-week avg {ha['wape_pct']:.1f}%"),
               unsafe_allow_html=True)
    c2.markdown(kpi("RMSE", f"{lgb['rmse']:.1f}", f"4-week avg {ha['rmse']:.1f} pickups/zone-hour"),
                unsafe_allow_html=True)
    tk = t["topk_hit_rate"]
    d2.markdown(kpi("Top-10 zones found (next 3 h)", f"{tk['lightgbm']['top10_next3h'] * 100:.0f}%",
                    f"4-week avg {tk['historical_avg_4wk']['top10_next3h'] * 100:.0f}%"), unsafe_allow_html=True)
    names = {"lightgbm": ("LightGBM", GOLD), "historical_avg_4wk": ("4-week average", BRONZE),
             "same_hour_last_week": ("Same hour last week", CYAN)}
    fig = go.Figure()
    for k, (lab, col) in names.items():
        bh = t["by_horizon"][k]
        fig.add_trace(go.Bar(x=list(bh), y=[v["wape_pct"] for v in bh.values()], name=lab, marker_color=col))
    fig.update_layout(barmode="group", title="Error by forecast horizon (WAPE %, lower is better)")
    left, right = st.columns(2)
    left.plotly_chart(style(fig, 360), width="stretch")
    fig = go.Figure()
    tiers = list(t["by_tier"]["lightgbm"])
    for k, (lab, col) in names.items():
        fig.add_trace(go.Bar(x=tiers, y=[t["by_tier"][k][tt]["wape_pct"] for tt in tiers], name=lab, marker_color=col))
    fig.update_layout(barmode="group", title="Error by zone tier (WAPE %)")
    right.plotly_chart(style(fig, 360), width="stretch")
    st.markdown("**Blizzard stress test** - Jan 25-26 and Feb 22-23 were held out of training:")
    sst = m["storm_stress_test"]
    st.write(f"WAPE on those {sst['rows_evaluated']:,} zone-hours: LightGBM **{sst['lightgbm']['wape_pct']:.1f}%** vs "
             f"4-week average {sst['historical_avg_4wk']['wape_pct']:.1f}% and same-hour-last-week "
             f"{sst['same_hour_last_week']['wape_pct']:.1f}%.")

# ------------------------------------------------------------------ about
with tab_about:
    q = S["quality"]
    st.markdown(
        f"""
**What this is.** A demand forecast a ride-hailing or fleet operator could use to position drivers: pickups per
hour for every NYC taxi zone over the next 24 hours, refreshed hourly.

**Data.** {q['raw_trips']:,} raw yellow-taxi trips (NYC TLC, Jan-Jul 2026) → {q['valid_trips']:,} valid trips after
removing bad timestamps, unknown zones, non-positive durations and refunds. Weather from Open-Meteo. Zone shapes from the TLC.

**Model.** One LightGBM model per horizon bucket, shared by the top {S['metrics']['data']['modelled_zones']} zones
({S['metrics']['data']['modelled_trip_share_pct']:.1f}% of trips); predicts demand relative to the same hour-of-week
average, from recent demand, city-wide demand, calendar/holiday flags and the weather at the target hour.

**Honest limitations.** Beyond ~6 hours the model is no better than the 4-week same-hour average - hourly zone demand
is noisy. Training uses observed weather as the "forecast", so real-world accuracy in storms will be somewhat lower.
The dashboard replays history (the TLC publishes with a lag); it is not connected to a live trip feed.
"""
    )
