# Transit Access & Price — what is good transit access worth?

A hedonic pricing model on open Montreal listings data. For each listing I
engineer a **transit-accessibility score** from GTFS feeds (STM, REM, exo) and
estimate how much transit access is capitalised into price — *controlling for
centrality*, so the effect isn't just picking up "central = expensive."

The point of the project is two-fold:
- a **one-command, reproducible pipeline** that collects and processes raw open
  data into an analysis-ready dataset, and
- a **dashboard** that answers ad-hoc questions ("premium for a 1-bed near a
  metro in the Plateau?") without anyone rebuilding a spreadsheet.

## How to reproduce

```bash
# 1. create + activate a virtual environment (see VS Code notes below)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. add the data URLs in config.yaml (see "Data" below), then:
python run_pipeline.py

# 4. launch the dashboard
streamlit run dashboard/app.py
```

## Data

- **Listings** — Inside Airbnb, Montreal (detailed `listings.csv.gz`). Licensed
  CC BY 4.0; attribution: *Inside Airbnb (insideairbnb.com)*. Use a **pre-2025**
  archived snapshot — Montreal's 2025 short-term-rental rules thinned the market
  sharply, so a recent snapshot is sparse and seasonally distorted.
- **Transit** — STM static GTFS (+ REM, exo for rail), for stop locations and
  service frequency.

Raw files live in `data/raw/` (gitignored). The committed analysis-ready output
is `data/processed/listings_features.parquet`.

## Method (the honest pairing)

- **OLS hedonic** (statsmodels): interpretable implicit prices, neighbourhood
  fixed effects, robust SE — the econ baseline.
- **LightGBM + SHAP**: captures nonlinearity / interactions and decomposes the
  implicit price of transit access listing-by-listing.

Report both. The headline analytical move is watching the transit premium
**shrink once `dist_to_core_m` is added** — and reporting that honestly.

## Caveats (stated, not buried)

1. **Identification.** Central areas are both transit-rich and amenity-rich;
   without a centrality control the transit effect is confounded.
2. **Interpretation.** Airbnb nightly prices reflect short-term/tourist
   willingness to pay, not commuter rent — a related but distinct premium.
3. **Vintage.** Built on a pre-2025 snapshot; the 2025 regulation changed the
   Montreal market.

## Layout

```
src/                pipeline steps (run in order via run_pipeline.py)
dashboard/app.py    Streamlit dashboard
data/processed/     analysis-ready outputs
config.yaml         URLs, buffer distances, the downtown reference point
```
