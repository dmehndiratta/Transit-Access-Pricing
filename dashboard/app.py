"""Transit-access pricing dashboard (bullet one).

Self-serve answers without rebuilding a spreadsheet: filter listings, read the
estimated transit premium, inspect the model. Reads artifacts written by the
pipeline (run `python run_pipeline.py` first).

Run:  streamlit run dashboard/app.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DATA = PROC / "listings_features.parquet"
RESULTS = ROOT / "models" / "results.json"
PREDS = PROC / "model_predictions.parquet"
SHAP_RAIL = PROC / "shap_rail.parquet"

st.set_page_config(page_title="Transit Access & Price", layout="wide")
st.title("What is transit access worth?")
st.caption("Hedonic pricing on open Montreal listings + GTFS transit features. "
           "Data: Inside Airbnb (CC BY 4.0) and STM GTFS.")

if not DATA.exists():
    st.warning("Run `python run_pipeline.py` first to build the dataset.")
    st.stop()

df = pd.read_parquet(DATA)
results = json.loads(RESULTS.read_text()) if RESULTS.exists() else None

with st.sidebar:
    st.header("Filters")
    hoods = sorted(df["neighbourhood_cleansed"].dropna().unique())
    pick_hood = st.multiselect("Neighbourhood", hoods, default=hoods)
    rooms = sorted(df["room_type"].dropna().unique())
    pick_room = st.multiselect("Room type", rooms, default=rooms)

view = df[df["neighbourhood_cleansed"].isin(pick_hood)
          & df["room_type"].isin(pick_room)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings", f"{len(view):,}")
c2.metric("Median price", f"${view['price'].median():,.0f}" if len(view) else "-")
c3.metric("Median walk to stop",
          f"{view['dist_nearest_stop_m'].median():,.0f} m" if len(view) else "-")
c4.metric("Median walk to rail",
          f"{view['dist_nearest_rail_m'].median():,.0f} m" if len(view) else "-")

tab_map, tab_prem, tab_model, tab_method = st.tabs(
    ["Map", "Transit premium", "Model", "Methodology"])

with tab_map:
    color_by = st.radio("Shade points by", ["price", "dist_nearest_rail_m"],
                        horizontal=True)
    pts = view.dropna(subset=["latitude", "longitude"])
    if len(pts):
        fig = px.scatter_map(
            pts, lat="latitude", lon="longitude", color=color_by,
            color_continuous_scale="Viridis", zoom=10, height=560,
            hover_data=["price", "dist_nearest_rail_m", "neighbourhood_cleansed"])
        fig.update_layout(map_style="carto-positron",
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No listings match the current filters.")

with tab_prem:
    st.subheader("Implicit price of transit access (OLS hedonic)")
    if not results:
        st.info("Run the pipeline to populate model results.")
    else:
        ols = pd.DataFrame(results["ols"])
        show = ols.rename(columns={
            "spec": "Specification",
            "pct_per_km_farther": "Price change per +1km from rail (%)",
            "transit_pval": "p-value", "r2": "R²", "n": "N"})
        st.table(show[["Specification",
                       "Price change per +1km from rail (%)",
                       "p-value", "R²", "N"]])
        naive = ols.iloc[0]["pct_per_km_farther"]
        ctrl = ols.iloc[1]["pct_per_km_farther"]
        a, b = st.columns(2)
        a.metric("Naive transit effect", f"{naive:.1f}% / km")
        b.metric("After controlling for centrality", f"{ctrl:.1f}% / km",
                 delta=f"{ctrl - naive:.1f} pts", delta_color="off")
        st.markdown(
            "**Finding.** A naive model (transit only) shows a small "
            "apparent premium for rail proximity (about 3.7% per km). "
            "But central areas are both transit-rich and expensive for "
            "many unrelated reasons, so the naive estimate conflates "
            "two things.\n\n"
            "Adding distance-to-core as a control **reverses the sign**: "
            "holding centrality constant, listings *farther* from rail "
            "are slightly pricier (+9.7%/km). The within-neighbourhood "
            "spec shrinks the gap to +4.8%/km but the reversal survives, "
            "ruling out neighbourhood selection.\n\n"
            "A plausible reading: for short-term tourist rentals, "
            "centrality and metro access are substitutes; a central "
            "listing doesn't *need* a metro. Conditional on centrality, "
            "station-adjacent micro-disamenities (noise, foot traffic) "
            "appear to dominate the residual transit value.")
        st.divider()
        st.subheader("What drives price? (mean |SHAP|, LightGBM)")
        rank = pd.DataFrame(results["shap_ranking"])
        figr = px.bar(rank.sort_values("mean_abs_shap"), x="mean_abs_shap",
                      y="feature", orientation="h", height=380)
        figr.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title="mean |SHAP| (log-price)", yaxis_title="")
        st.plotly_chart(figr, use_container_width=True)

with tab_model:
    if results:
        m = results["gbm"]
        a, b, c = st.columns(3)
        a.metric("Test R²", m["r2"])
        b.metric("RMSE (log price)", m["rmse_log"])
        c.metric("Test listings", f"{m['n_test']:,}")
    if PREDS.exists():
        pred = pd.read_parquet(PREDS)
        fig = px.scatter(pred, x="actual_log_price", y="predicted_log_price",
                         opacity=0.4, height=440,
                         labels={"actual_log_price": "Actual (log price)",
                                 "predicted_log_price": "Predicted (log price)"})
        lo = float(min(pred.min())); hi = float(max(pred.max()))
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                      line=dict(dash="dash"))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    if SHAP_RAIL.exists():
        st.subheader("How rail proximity moves price (SHAP dependence)")
        sr = pd.read_parquet(SHAP_RAIL)
        fig2 = px.scatter(sr, x="dist_nearest_rail_m", y="shap_log_price",
                          opacity=0.4, height=380,
                          labels={"dist_nearest_rail_m": "Distance to nearest rail (m)",
                                  "shap_log_price": "Contribution to log price"})
        fig2.add_hline(y=0, line_dash="dash")
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

with tab_method:
    st.markdown(
        "**Identification.** Central areas are both transit-rich and "
        "amenity-rich; without a centrality control the transit effect "
        "is confounded. The headline result here is the *reversal* "
        "once `dist_to_core_m` is added, indicating that the naive premium was "
        "centrality, not transit.\n\n"
        "**Interpretation.** Airbnb nightly prices reflect short-term "
        "tourist willingness to pay, not commuter rent. For tourists, "
        "centrality and metro access are partial substitutes, which "
        "may explain why station proximity carries a *discount* once "
        "centrality is held fixed.\n\n"
        "**Vintage.** Built on the Inside Airbnb 2025-06-15 Montreal "
        "snapshot. This is the earliest snapshot with intact pricing data "
        "(later vintages either ship prices stripped out or reflect "
        "the post-crackdown market).\n\n"
        "**Data.** Inside Airbnb (CC BY 4.0); STM GTFS (attribution "
        "required).")