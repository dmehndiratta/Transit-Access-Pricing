"""Step 5 - Hedonic baseline + ML, the honest pairing.

OLS hedonic (statsmodels) : interpretable implicit prices - the econ baseline.
LightGBM + SHAP           : nonlinearity / interactions + per-listing decomposition.

Compare them. The headline move: report the transit premium BEFORE and AFTER
adding dist_to_core_m, and show it shrink.
"""
from __future__ import annotations
import pandas as pd

from _common import load_config, processed_dir

FEATURES = [
    "accommodates", "bedrooms", "bathrooms", "room_type",
    "dist_nearest_stop_m", "dist_nearest_rail_m",
    "n_stops_within_buffer", "freq_weighted_access",
    "dist_to_core_m",             # the identification control
    "neighbourhood_cleansed",     # fixed effects
]


def fit_ols(df: pd.DataFrame):
    """Log-price hedonic OLS, neighbourhood FE, robust SE.

    Fit twice - with and without dist_to_core_m - and compare the transit
    coefficients. statsmodels.formula.api.ols(..., 'C(neighbourhood_cleansed)')
    then .fit(cov_type='HC3').
    """
    raise NotImplementedError  # TODO


def fit_gbm(df: pd.DataFrame):
    """LightGBM on log_price; return (model, X_test, y_test)."""
    raise NotImplementedError  # TODO


def explain(model, X) -> None:
    """SHAP -> implicit price of transit access, listing by listing."""
    raise NotImplementedError  # TODO


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    df = pd.read_parquet(processed_dir(cfg) / "listings_features.parquet")
    # ols = fit_ols(df); gbm, Xte, yte = fit_gbm(df); explain(gbm, Xte)
    print("Step 5 complete.")


if __name__ == "__main__":
    main()
