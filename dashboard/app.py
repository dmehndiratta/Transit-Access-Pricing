"""Transit Access & Price - dashboard.

Plain-English self-serve view onto the hedonic-pricing analysis. Reads the
artifacts written by run_pipeline.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DATA = PROC / "listings_features.parquet"
RESULTS = ROOT / "models" / "results.json"
PREDS = PROC / "model_predictions.parquet"
SHAP_RAIL = PROC / "shap_rail.parquet"

# Plain-English labels used everywhere a column name would otherwise show.
LABELS = {
    "price": "Price ($/night)",
    "log_price": "Log price",
    "dist_nearest_stop_m": "Distance to nearest transit stop (m)",
    "dist_nearest_rail_m": "Distance to nearest rail station (m)",
    "n_stops_within_buffer": "Transit stops within 800m walk",
    "freq_weighted_access": "Frequency-weighted transit access (AM-peak)",
    "dist_to_core_m": "Distance to downtown (m)",
    "accommodates": "Accommodates (guests)",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "room_type": "Room type",
    "neighbourhood_cleansed": "Neighbourhood",
    "review_scores_rating": "Average review score",
    "number_of_reviews": "Number of reviews",
}
def L(c): return LABELS.get(c, c)

st.set_page_config(page_title="Transit Access & Price", layout="wide")
st.title("Does being near transit make a place more expensive?")
st.markdown(
    "A hedonic-pricing study of Montreal short-term rentals. We test the "
    "intuition that proximity to transit raises listing prices — and find "
    "that **the obvious answer flips once you account for centrality.** "
    "Built end-to-end from open data with a reproducible pipeline.")

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
c1.metric("Listings shown", f"{len(view):,}")
c2.metric("Median price/night",
          f"${view['price'].median():,.0f}" if len(view) else "-")
c3.metric("Median walk to any stop",
          f"{view['dist_nearest_stop_m'].median():,.0f} m" if len(view) else "-")
c4.metric("Median walk to nearest rail",
          f"{view['dist_nearest_rail_m'].median():,.0f} m" if len(view) else "-")

tab_map, tab_prem, tab_model, tab_method = st.tabs(
    ["Map", "Transit premium", "Model", "Methodology"])

# ---------------- MAP ----------------
with tab_map:
    st.caption(
        "Each dot is one listing in your filtered selection. Use the toggle "
        "to colour by price or by walking distance to the nearest rail "
        "station. Colour scales are clipped to the 5th–95th percentile so "
        "the bulk of variation is visible — a few extreme outliers don't "
        "wash everything else out.")
    map_choices = {
        "Price": "price",
        "Distance to rail": "dist_nearest_rail_m",
        "Distance to any stop": "dist_nearest_stop_m",
        "Stops within 800m walk": "n_stops_within_buffer",
    }
    pick = st.radio("Shade points by", list(map_choices.keys()), horizontal=True)
    color_col = map_choices[pick]

    pts = view.dropna(subset=["latitude", "longitude", color_col])
    if len(pts):
        lo = float(pts[color_col].quantile(0.05))
        hi = float(pts[color_col].quantile(0.95))
        if lo == hi:
            hi = lo + 1e-9
        scale = "Plasma" if color_col == "price" else "Viridis"
        fig = px.scatter_map(
            pts, lat="latitude", lon="longitude", color=color_col,
            range_color=[lo, hi], color_continuous_scale=scale,
            zoom=10.5, height=580,
            labels={color_col: L(color_col),
                    "price": L("price"),
                    "dist_nearest_rail_m": L("dist_nearest_rail_m"),
                    "neighbourhood_cleansed": L("neighbourhood_cleansed")},
            hover_data={"price": ":$.0f",
                        "dist_nearest_rail_m": ":,.0f",
                        "neighbourhood_cleansed": True,
                        "latitude": False, "longitude": False})
        fig.update_layout(map_style="carto-positron",
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No listings match the current filters.")

# ---------------- TRANSIT PREMIUM ----------------
with tab_prem:
    st.subheader("Implicit price of rail proximity")
    if not results:
        st.info("Run the pipeline to populate model results.")
    else:
        ols = pd.DataFrame(results["ols"])
        show = ols.rename(columns={
            "spec": "Specification",
            "pct_per_km_farther": "Price change per +1km from rail (%)",
            "transit_pval": "p-value", "r2": "R²", "n": "Listings"})
        st.table(show[["Specification",
                       "Price change per +1km from rail (%)",
                       "p-value", "R²", "Listings"]])

        naive = ols.iloc[0]["pct_per_km_farther"]
        ctrl = ols.iloc[1]["pct_per_km_farther"]
        a, b = st.columns(2)
        a.metric("Naive transit effect (no centrality control)",
                 f"{naive:+.1f}% per km farther from rail")
        b.metric("After controlling for centrality",
                 f"{ctrl:+.1f}% per km farther from rail",
                 delta=f"{ctrl - naive:+.1f} pts", delta_color="off")

        st.markdown(
            "**How to read this.** The first row is the *naive* model — "
            "structural features (size, bathrooms, room type, reviews) plus "
            "the rail-distance variable, but **no** control for how central "
            "a listing is. It suggests being closer to rail commands a small "
            "price premium.\n\n"
            "Row 2 adds one variable: distance to downtown. The sign **flips**. "
            "Conditional on centrality, listings *farther* from rail are "
            "slightly pricier. Row 3 adds neighbourhood fixed effects — i.e. "
            "we compare listings *within the same neighbourhood* — and the "
            "reversal survives, ruling out the explanation that it's just "
            "that rail tends to run through cheaper neighbourhoods.\n\n"
            "**A plausible reading.** Short-term rentals are priced for "
            "tourists. For a tourist, *centrality* and *metro access* are "
            "partial substitutes — if you're already downtown, you don't "
            "need the metro. So the residual marginal value of rail proximity "
            "is small, and station-adjacent micro-disamenities (noise, "
            "foot traffic) appear to dominate.")

        st.divider()
        st.subheader("What drives price overall? (LightGBM model)")
        st.caption(
            "We also fit a flexible machine-learning model (gradient-boosted "
            "trees) as an independent cross-check. The bars below show each "
            "feature's average contribution to the price prediction — bigger "
            "bars mean the model leans more on that feature. Notice that "
            "distance to downtown matters far more than distance to rail, "
            "consistent with the OLS reversal above.")
        rank = pd.DataFrame(results["shap_ranking"]).copy()
        rank["feature_label"] = rank["feature"].map(L)
        figr = px.bar(rank.sort_values("mean_abs_shap"),
                      x="mean_abs_shap", y="feature_label",
                      orientation="h", height=420,
                      labels={"mean_abs_shap": "Average impact on price prediction",
                              "feature_label": ""})
        figr.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figr, use_container_width=True)

# ---------------- MODEL ----------------
with tab_model:
    st.subheader("How well does the model predict prices?")
    st.caption(
        "We split listings 80/20 into a training set (used to fit the model) "
        "and a test set (held out). The chart below plots, for each test "
        "listing, the price the model predicted against the price the "
        "listing actually had. Dots near the dashed diagonal are accurate "
        "predictions; dots far from it are misses.")
    if results:
        m = results["gbm"]
        a, b, c = st.columns(3)
        a.metric("R² on held-out listings", m["r2"],
                 help="Fraction of price variation the model captures. "
                      "0.5–0.7 is typical for hedonic models on listing data — "
                      "there's a lot the data simply doesn't observe "
                      "(interior quality, exact square footage, etc.).")
        b.metric("Typical error (log price)", m["rmse_log"],
                 help="Root mean squared error in log price. ~0.4 means the "
                      "model is typically within a factor of e^0.4 ≈ 1.5x of "
                      "the actual price.")
        c.metric("Test listings", f"{m['n_test']:,}")
    if PREDS.exists():
        pred = pd.read_parquet(PREDS)
        fig = px.scatter(pred, x="actual_log_price", y="predicted_log_price",
                         opacity=0.35, height=440,
                         labels={"actual_log_price": "Actual price (log)",
                                 "predicted_log_price": "Predicted price (log)"})
        lo = float(min(pred.min())); hi = float(max(pred.max()))
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                      line=dict(dash="dash"))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    if SHAP_RAIL.exists():
        st.divider()
        st.subheader("How does rail proximity move the model's prediction?")
        st.caption(
            "Each dot is one test listing. The x-axis is its walking "
            "distance to the nearest rail station; the y-axis is how much "
            "that single feature pushed the model's price prediction up "
            "(above zero) or down (below zero). If rail proximity were a "
            "strong price driver, we'd expect a clear downward trend "
            "(closer to rail → higher predicted price). The fact that the "
            "cloud is fairly flat is exactly the picture you'd expect "
            "given the OLS finding: once other features absorb the "
            "centrality story, rail distance has little independent pull.")
        sr = pd.read_parquet(SHAP_RAIL)
        fig2 = px.scatter(sr, x="dist_nearest_rail_m", y="shap_log_price",
                          opacity=0.35, height=380,
                          labels={"dist_nearest_rail_m": L("dist_nearest_rail_m"),
                                  "shap_log_price": "Contribution to predicted price (log)"})
        fig2.add_hline(y=0, line_dash="dash")
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- METHODOLOGY ----------------
with tab_method:
    st.markdown("### The question")
    st.markdown(
        "How much of a listing's price is explained by being near transit, "
        "once everything else is held constant? This is a hedonic-pricing "
        "question — the standard economics framework for measuring how an "
        "unpriced amenity (transit access) is capitalised into the price of "
        "something that *is* priced (here, accommodation).")

    st.markdown("### What this study accounts for")
    st.markdown(
        "Each listing's price is modelled as a function of:\n\n"
        "- **Structural features** — accommodates, bedrooms, bathrooms, "
        "room type\n"
        "- **Quality and popularity proxies** — average review score, "
        "number of reviews\n"
        "- **Location** — distance to downtown (continuous), plus "
        "neighbourhood fixed effects in the strict spec (so we compare "
        "listings *within* the same neighbourhood)\n"
        "- **Transit access** — distance to nearest stop, distance to "
        "nearest rail station, number of stops within an 800m walk, and a "
        "frequency-weighted access score (stops nearby, weighted by "
        "AM-peak departures)")

    st.markdown("### What this study can't see")
    st.markdown(
        "Inside Airbnb publishes a generous schema, but some price-relevant "
        "factors are unobservable in this data:\n\n"
        "- **Square footage** — not reported by Airbnb\n"
        "- **Renovation status, building age, interior quality** — not in "
        "the data\n"
        "- **Amenity quality** — listings carry an amenities list, but "
        "verifying or scoring quality is out of scope\n"
        "- **Host-specific effects** beyond what room type captures\n"
        "- **Seasonality** — we use one snapshot, so seasonal demand "
        "swings aren't separated from listing-level effects\n\n"
        "These limitations apply equally to the naive and the controlled "
        "models, so they can't explain the sign reversal — that comes "
        "purely from adding centrality. But they do mean the model's "
        "absolute predictive accuracy has a ceiling: a typical R² of "
        "~0.6 reflects how much listing variation simply isn't observable "
        "in open data.")

    st.markdown("### Why two methods")
    st.markdown(
        "We report **OLS hedonic regression** for interpretable implicit "
        "prices (the standard econ approach) and **LightGBM with SHAP** "
        "as an independent flexible-form check. If a linear model and a "
        "gradient-boosted tree model agree on the centrality story, the "
        "finding isn't an artefact of either's functional form. They do.")

    st.markdown("### Interpretation caveats")
    st.markdown(
        "- Airbnb nightly prices reflect short-term **tourist** "
        "willingness to pay, not commuter rent. The substitution between "
        "centrality and transit access that drives the finding is "
        "specific to the tourist context — commuter-rent capitalisation "
        "of transit may look different.\n"
        "- The estimated coefficient on rail distance is conditional on "
        "the centrality measure used. Distance to a single downtown "
        "reference is a reasonable proxy for tourist-relevant centrality "
        "in Montreal; a more granular measure (e.g. distance to specific "
        "attractions) might shift the magnitude.\n"
        "- Results are conditioned on the 2025-06-15 snapshot — the "
        "earliest free Inside Airbnb release for Montreal that still "
        "carries listing prices.")

    st.markdown("### Data and attribution")
    st.markdown(
        "- **Listings**: Inside Airbnb (insideairbnb.com), Montreal "
        "snapshot dated 2025-06-15, licensed CC BY 4.0.\n"
        "- **Transit**: Société de transport de Montréal (STM) static "
        "GTFS feed, used under STM's open data licence. The current STM "
        "feed includes the REM rail line as well as the metro and buses, "
        "so a single feed covers all transit modes relevant here.")