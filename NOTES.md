# Development Notes

A running log of build decisions, design choices, and data gotchas for the
transit-access pricing project. Kept as a record of *why* the code looks the
way it does — not just what it does.

## Scaffold & environment

- **No geopandas.** Spatial work (nearest-stop, walk-buffer counts) is done
  with `sklearn.neighbors.BallTree` on a haversine metric instead of geopandas.
  This deliberately avoids the GDAL/Fiona install pain on Windows; the whole
  stack installs from plain wheels.
- **One-command pipeline.** `run_pipeline.py` chains all five steps so the
  analysis-ready dataset rebuilds from raw open data in a single run — the
  reproducibility signal that matters more than any individual script.
- **Module naming.** Pipeline step files use plain names (no leading digits)
  because Python module names can't start with a number (`import 01_fetch` is a
  syntax error).

## Data sources

- **Listings:** Inside Airbnb, Montreal detailed `listings.csv.gz` (CC BY 4.0).
- **Transit:** STM static GTFS — covers bus, metro, AND the REM rail line in the
  current feed, so separate REM/exo feeds proved unnecessary.

## Step 3 — transit features

- `stops.txt` carries no `route_type`, so metro-vs-bus is recovered by joining
  `stop_times.txt -> trips.txt (route_id) -> routes.txt (route_type)`.
- Stops absent from `stop_times` (parent stations, unused entries) are dropped
  so they don't inflate the stop-density measure.
- Features per listing: distance to nearest stop, distance to nearest rail,
  count of stops within an 800 m walk buffer, and a frequency-weighted access
  score (buffer stops weighted by AM-peak departures).
- Time parser handles GTFS edge cases: hours >= 24 (after-midnight trips) and
  non-zero-padded times like `8:00:00`.
- Real-data sanity check: 8,962 serviced stops, 72 rail; median 92 m to nearest
  stop vs 436 m to nearest rail — the expected dense-bus / sparse-rail ordering.

## Step 5 — modelling (the honest pairing)

- **OLS hedonic** fit in three nested specs on one sample to show the transit
  coefficient attenuate: (1) transit only, (2) + distance-to-core, (3) + nbhd
  fixed effects. The headline analytical move is reporting the *shrunken*,
  centrality-controlled estimate, not the inflated naive one.
- **LightGBM + SHAP** as an independent cross-check: captures nonlinearity and
  gives a per-listing decomposition. If OLS and SHAP agree on the small role of
  transit-once-centrality-is-controlled, that's robustness.
- **Identification trap (documented, not hidden):** central areas are both
  transit-rich and expensive, so a naive transit effect is mostly centrality in
  disguise. `dist_to_core_m` is the control that breaks the confound.
- Writes `models/results.json`, `data/processed/model_predictions.parquet`,
  `data/processed/shap_rail.parquet` for the dashboard to read.

## Step 2 — the blank-price snapshot gotcha

- **Symptom:** Step 5 reported `modeling sample: 0 rows` and patsy threw
  `negative dimensions are not allowed`.
- **Cause:** the chosen Inside Airbnb snapshot (2025-12) ships `price` 100%
  blank (10,041 / 10,041 null). The `dropna` across model columns then wiped
  the entire sample.
- **Fix:** when `price` is >50% null, derive a per-listing price from
  `calendar.csv.gz` as the **median** quoted nightly rate (median, not mean, to
  resist seasonal spikes). `fetch_data.py` now auto-derives the calendar URL
  from the listings URL.
- Also added: impute `bedrooms` from `accommodates` (studios ship blank), fall
  `bathrooms` back to the median, winsorize price to the 1st–99th percentile,
  and drop rows missing coordinates or price — so a single sparse column can
  never wipe the sample again.

## Data vintage

- Snapshot is **post-crackdown (late 2025)**: Montreal's 2025 short-term-rental
  rules reshaped the market. Listing counts and the price level reflect that
  regime. Stated as a caveat rather than hidden.
- NOTE TO SELF: update the `snapshot_date` value and its "pre-2025" comment in
  config.yaml to match the snapshot actually used (currently 2025-12-22).

## Interpretation caveats (carried in README + dashboard)

1. Airbnb nightly prices reflect short-term/tourist willingness to pay, not
   commuter rent — a related but distinct premium.
2. The transit premium is identified only after controlling for centrality.
3. Results are conditioned on the post-2025-regulation market.