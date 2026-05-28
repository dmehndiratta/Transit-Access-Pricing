"""Step 5 - Hedonic baseline + ML, the honest pairing.

OLS hedonic (statsmodels) : interpretable implicit prices - the econ baseline.
LightGBM + SHAP           : nonlinearity / interactions + per-listing decomposition.

The headline analytical move: fit the SAME sample three ways and watch the
transit coefficient attenuate as location is controlled for -
  (1) transit only
  (2) + distance to downtown core   <- the identification fix
  (3) + neighbourhood fixed effects <- the strict spec
Report the shrinking number honestly.

Writes (consumed by the dashboard):
  models/results.json                      OLS table + GBM metrics + SHAP ranking
  data/processed/model_predictions.parquet test-set actual vs predicted
  data/processed/shap_rail.parquet         per-listing SHAP for rail distance
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from _common import load_config, processed_dir, ROOT

BASE_RHS = ("accommodates + bedrooms + bathrooms + C(room_type) "
            "+ review_scores_rating + number_of_reviews")
TRANSIT = "dist_nearest_rail_km"   # focal capitalisation measure for the OLS story

NUM_FEATURES = [
    "accommodates", "bedrooms", "bathrooms",
    "review_scores_rating", "number_of_reviews",
    "dist_nearest_stop_m", "dist_nearest_rail_m",
    "n_stops_within_buffer", "freq_weighted_access", "dist_to_core_m",
]
CAT_FEATURES = ["room_type", "neighbourhood_cleansed"]

MODEL_COLS = (["log_price"] + NUM_FEATURES + CAT_FEATURES
              + ["dist_nearest_rail_km", "dist_to_core_km"])


def pct_per_km(coef: float) -> float:
    """Semi-log coefficient -> % price change per +1 km."""
    return 100.0 * (np.exp(coef) - 1.0)


def fit_ols(df: pd.DataFrame) -> list[dict]:
    """Fit three nested specs on one sample; report the transit coefficient."""
    specs = {
        "1. transit only": f"log_price ~ {BASE_RHS} + {TRANSIT}",
        "2. + centrality": f"log_price ~ {BASE_RHS} + {TRANSIT} + dist_to_core_km",
        "3. + nbhd FE": f"log_price ~ {BASE_RHS} + {TRANSIT} + dist_to_core_km "
                        f"+ C(neighbourhood_cleansed)",
    }
    rows = []
    for label, formula in specs.items():
        res = smf.ols(formula, data=df).fit(cov_type="HC3")
        rows.append({
            "spec": label,
            "transit_coef": float(res.params[TRANSIT]),
            "transit_se": float(res.bse[TRANSIT]),
            "transit_pval": float(res.pvalues[TRANSIT]),
            "pct_per_km_farther": round(pct_per_km(res.params[TRANSIT]), 2),
            "r2": round(float(res.rsquared), 3),
            "n": int(res.nobs),
        })
    return rows


def fit_gbm(df: pd.DataFrame, cfg: dict):
    X = df[NUM_FEATURES + CAT_FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype("category")
    y = df["log_price"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"])

    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=cfg["model"]["random_state"], verbose=-1)
    model.fit(Xtr, ytr, categorical_feature=CAT_FEATURES)

    pred = model.predict(Xte)
    metrics = {
        "r2": round(float(r2_score(yte, pred)), 3),
        "rmse_log": round(float(np.sqrt(mean_squared_error(yte, pred))), 3),
        "n_test": int(len(yte)),
    }
    return model, Xte, yte, pred, metrics


def explain(model, Xte: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """SHAP feature ranking + per-listing contribution of rail distance."""
    sv = shap.TreeExplainer(model).shap_values(Xte)
    mean_abs = np.abs(sv).mean(axis=0)
    ranking = sorted(
        ({"feature": f, "mean_abs_shap": round(float(v), 4)}
         for f, v in zip(Xte.columns, mean_abs)),
        key=lambda d: -d["mean_abs_shap"])

    rail_idx = list(Xte.columns).index("dist_nearest_rail_m")
    rail_df = pd.DataFrame({
        "dist_nearest_rail_m": Xte["dist_nearest_rail_m"].to_numpy(),
        "shap_log_price": sv[:, rail_idx],
    })
    return ranking, rail_df


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    pdir = processed_dir(cfg)
    df = pd.read_parquet(pdir / "listings_features.parquet")

    df["dist_nearest_rail_km"] = df["dist_nearest_rail_m"] / 1000.0
    df["dist_to_core_km"] = df["dist_to_core_m"] / 1000.0
    df = df.dropna(subset=MODEL_COLS).copy()
    print(f"  modeling sample: {len(df):,} rows")

    ols_rows = fit_ols(df)
    print("  OLS transit coefficient by spec (price change per +1km from rail):")
    for r in ols_rows:
        print(f"    {r['spec']:<18} {r['pct_per_km_farther']:>7.2f}%   "
              f"(p={r['transit_pval']:.3g}, R2={r['r2']})")

    model, Xte, yte, pred, metrics = fit_gbm(df, cfg)
    print(f"  LightGBM  R2={metrics['r2']}  RMSE(log)={metrics['rmse_log']}")

    ranking, rail_df = explain(model, Xte)
    print("  top SHAP features: "
          + ", ".join(d["feature"] for d in ranking[:4]))

    (ROOT / "models").mkdir(exist_ok=True)
    with open(ROOT / "models" / "results.json", "w", encoding="utf-8") as f:
        json.dump({"ols": ols_rows, "gbm": metrics, "shap_ranking": ranking},
                  f, indent=2)

    pd.DataFrame({"actual_log_price": yte.to_numpy(),
                  "predicted_log_price": pred}).to_parquet(
        pdir / "model_predictions.parquet", index=False)
    rail_df.to_parquet(pdir / "shap_rail.parquet", index=False)

    print("Step 5 complete.")


if __name__ == "__main__":
    main()