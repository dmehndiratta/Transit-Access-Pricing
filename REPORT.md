# What is transit access worth in Montreal?

A hedonic-pricing study of short-term rentals, with a finding that runs against the obvious answer.

**Live dashboard:** [transit-access-airbnb-pricing.streamlit.app](https://transit-access-airbnb-pricing.streamlit.app/)
**Code and pipeline:** see [README.md](README.md)

---

## Summary

The obvious answer to "does being near transit make a place more expensive" is yes. The data agree, until you control for one thing. Once distance to downtown enters the model, the sign on rail proximity flips. Listings closer to a rail station turn out to trade at a slight *discount*, not a premium, within the same neighbourhood and conditional on size, room type, and review quality. That reversal is the finding.

This isn't a paradox. It's what happens when a "transit effect" turns out to be mostly a centrality effect wearing a transit label. Short-term rentals are priced for tourists, who already pay heavily for centrality. Conditional on being central, the marginal value of a metro stop next door appears to be roughly offset by what comes with a metro stop next door: noise, foot traffic, weaker curb appeal.

## The question

Standard hedonic pricing asks how much an unpriced amenity is capitalised into the price of something that *is* priced. Public transit is one of the canonical cases. The Rosen (1974) framework says the implicit price of an amenity can be recovered from the regression coefficient on a measure of access, once enough of everything else has been controlled for.

The condition in italics matters. A naive regression will conflate transit access with whatever else is geographically correlated with transit. In any real city that includes centrality, walkability, density, and a long list of unobserved amenities that cluster together. The interesting question is therefore not "is the naive coefficient positive" but "what happens to it as you add controls."

## Data

Two open feeds carry the whole analysis.

The first is Inside Airbnb's Montreal listings file, June 15 2025 snapshot. Each row is one active listing on the platform at the time of scrape, with structural attributes (capacity, bedrooms, bathrooms, room type), location (latitude, longitude, neighbourhood), price per night, and review information. Why this date specifically: every Montreal snapshot from late 2025 onward ships with the price column stripped out, presumably as a platform response to the 2025 short-term-rental crackdown. Earlier 2024 snapshots exist only as paid archive requests. June 15 2025 is the earliest free snapshot that still carries prices, falling just before the regulatory regime came into full effect.

The second is the Société de transport de Montréal's static GTFS feed. This carries stop locations, route information, and timetables for the bus network, the metro, and (in the current feed) the REM rail line. Both feeds are licensed for reuse with attribution. Inside Airbnb under CC BY 4.0, STM under their open data licence.

Final sample after cleaning: 8,778 listings with valid coordinates, prices, and structural data.

## What the pipeline does

The repository ships a five-step pipeline that runs end to end with one command.

Step one fetches the raw feeds. Step two cleans the listings, parsing the $-prefixed price strings, the bathroom text field, and imputing modest gaps in bedrooms and review scores. Step three is the substantive engineering step: for each listing it computes distance to the nearest serviced transit stop, distance to the nearest rail station, count of stops within an 800-metre walk, and a frequency-weighted access score (stops weighted by AM-peak departures). Stops not appearing in `stop_times.txt` are dropped so unused entries don't inflate density. Metro versus bus is identified by joining `stop_times` through `trips` to `routes`, because `stops.txt` itself carries no route-type information. Step four assembles the analysis-ready dataset and adds the centrality variable, computed as the haversine distance to a downtown reference point. Step five fits the models.

The whole pipeline runs from raw downloads to analysis-ready data in under a minute on a laptop.

## Methods

Two models, one sample. Both are reported.

The first is ordinary least squares on log price, the standard hedonic specification. It is fit in three nested versions on the same sample:

1. Structural and quality controls plus rail distance. No centrality.
2. Same as (1) plus distance to downtown.
3. Same as (2), plus neighbourhood fixed effects.

Reporting all three shows what happens to the rail coefficient as confounders are progressively absorbed. Heteroskedasticity-consistent standard errors throughout.

The second model is LightGBM, a gradient-boosted tree regressor, fit on the same features. The point isn't to replace the hedonic. Trees handle nonlinearity and interactions that linear OLS does not, so if the two models agree on which features matter, the finding is harder to dismiss as an artefact of functional form. SHAP values give the per-listing decomposition of which features drove which prediction.

## Result

Estimated effect of rail proximity on log price, expressed as the percentage change per kilometre of distance from the nearest rail station:

| Specification | Effect per +1 km | p-value | R² |
|---|---:|---:|---:|
| 1. Structural controls only | -3.7% | < 0.001 | 0.48 |
| 2. + distance to downtown | +9.7% | < 0.001 | 0.55 |
| 3. + neighbourhood fixed effects | +4.8% | < 0.001 | 0.56 |

The naive specification looks normal. Each kilometre of distance from a rail station knocks roughly 3.7% off the price, with very tight standard errors. This is the result an analyst running the obvious regression would report and the result a journalist would write up.

Add one variable (distance to downtown) and the sign reverses. Listings farther from rail are now estimated to be more expensive by about 9.7% per kilometre. The third row strengthens the test further: comparing only listings in the same neighbourhood, the effect shrinks but remains positive at 4.8% per kilometre. The naive premium was not a transit premium. It was a centrality premium in disguise.

The LightGBM cross-check confirms the pattern independently. The model's most important predictors are accommodates, distance to downtown, bathrooms, and room type. Distance to the nearest rail station does not crack the top features. The SHAP dependence plot for rail distance is nearly flat, which is what you expect if the variable carries little independent information once everything else is in the model.

## Why

Two things are going on.

First, the naive coefficient was not measuring transit access. It was measuring centrality, because central listings are also close to rail. Adding distance to downtown lets the model separate the two, and almost the entire apparent transit effect turns out to belong to centrality.

The harder part is the second piece: why the residual coefficient flips positive. The most coherent reading is that short-term rentals are priced for a particular kind of buyer (the tourist), and for that buyer centrality and metro access are partial substitutes. If you are already at the attractions, you do not need the metro. Conditional on already being central, the residual marginal value of having a station within walking distance is small. The disamenities that come with being right next to a station then dominate the small remaining premium. The result is a slight negative capitalisation, holding centrality fixed.

A commuter-rent equivalent of this study, if rental data were available, would likely show a different sign. Commuters value transit for itself, not as a substitute for centrality.

## What this study cannot see

Inside Airbnb publishes a generous schema for an open dataset, but several price-relevant variables are simply not in it.

Square footage is not reported. Building age and renovation status are not reported. Interior quality, finishes, and the actual condition of the place are not observable beyond what reviews indirectly capture. Host-specific effects beyond room type are limited. The data are a single snapshot, so seasonal demand variation cannot be separated from listing-level effects.

These omissions matter for absolute prediction accuracy. The model's R² of around 0.6 reflects how much listing variation is genuinely unobservable from open data, not a modelling failure. They do not, however, threaten the identification of the centrality reversal. The same unobservables sit inside every specification in the three-row table. The only thing changing between rows is the addition of centrality, which is fully observed.

## Caveats on interpretation

The estimated effect is conditional on the centrality measure used. Distance to a single downtown reference works as a tourist-relevant centrality proxy in Montreal. A more granular measure built from distance to specific attractions might shift the magnitude, though the qualitative pattern is unlikely to change.

The substitution story is specific to the short-term-rental context. Generalising the finding to commuter housing markets requires a leap that this data cannot support.

Results are conditioned on the post-2025 regulatory environment as of June 2025. The Montreal short-term-rental market has been reshaped since.

## Reproducibility

The repository contains everything needed to reproduce these results from scratch. Clone it, install requirements, point `config.yaml` at the Inside Airbnb listings URL, and run `python run_pipeline.py`. The full chain from raw download to analysis-ready dataset to fitted models takes well under two minutes on a modern laptop. A separate `check_snapshot.py` script lets anyone verify whether a candidate Inside Airbnb snapshot carries pricing data before wiring it in, which was needed during this project after the December 2025 snapshot turned out to be price-stripped.

The dashboard at [transit-access-airbnb-pricing.streamlit.app](https://transit-access-airbnb-pricing.streamlit.app/) reads the same artifacts the pipeline produces. Filters apply on the fly. The OLS table, the SHAP rankings, and the model diagnostics are computed once at pipeline run time and read into the dashboard, not refit on every interaction.

## Data and attribution

Listings data from Inside Airbnb (insideairbnb.com), used under CC BY 4.0. Transit data from the Société de transport de Montréal, used under STM's open data licence. Both are gratefully acknowledged.