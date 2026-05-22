"""Transit-access pricing dashboard (bullet one).

Lets a user answer ad-hoc questions without rebuilding a spreadsheet:
filter by neighbourhood / room type and read off the transit picture.

Run:  streamlit run dashboard/app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "listings_features.parquet"

st.set_page_config(page_title="Transit Access & Price", layout="wide")
st.title("What is transit access worth?")

if not DATA.exists():
    st.warning("Run `python run_pipeline.py` first to build the dataset.")
    st.stop()

df = pd.read_parquet(DATA)

with st.sidebar:
    st.header("Filters")
    hoods = sorted(df["neighbourhood_cleansed"].dropna().unique())
    pick_hood = st.multiselect("Neighbourhood", hoods, default=hoods[:5])
    rooms = sorted(df["room_type"].dropna().unique())
    pick_room = st.multiselect("Room type", rooms, default=rooms)

view = df[df["neighbourhood_cleansed"].isin(pick_hood) & df["room_type"].isin(pick_room)]

c1, c2, c3 = st.columns(3)
c1.metric("Listings", f"{len(view):,}")
c2.metric("Median price", f"${view['price'].median():,.0f}" if len(view) else "-")
c3.metric(
    "Median dist. to nearest stop (m)",
    f"{view['dist_nearest_stop_m'].median():,.0f}"
    if "dist_nearest_stop_m" in view and len(view) else "-",
)

tab_map, tab_premium, tab_model = st.tabs(["Map", "Transit premium", "Model"])

with tab_map:
    pts = view.rename(columns={"latitude": "lat", "longitude": "lon"})
    st.map(pts[["lat", "lon"]].dropna())

with tab_premium:
    st.info(
        "After Step 5 runs: show the OLS vs SHAP implicit price of transit "
        "access here, with the before/after-centrality comparison."
    )

with tab_model:
    st.info("After Step 5 runs: predicted vs actual, SHAP summary, honest error.")
