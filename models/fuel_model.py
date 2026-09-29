"""
QFleet Phase 1 — Fuel Consumption Prediction Models
=====================================================
Trains and evaluates five models on synthetic voyage data.

Models (evaluated on a held-out 20% test split)
------------------------------------------------
  1. OLS Baseline       statsmodels OLS on log(fuel_consumed_t) ~
                        log(speed_kn) + log(distance_nm) + load_factor +
                        weather_factor + C(vessel_class) + C(fuel_type)

  2. Linear Regression  sklearn LinearRegression (raw features + dummies)

  3. Random Forest      sklearn RandomForestRegressor (100 trees)

  4. Plain XGBoost      xgboost XGBRegressor (default params from config)

  5. Physics + XGBoost  Analytic physics prediction (fixed exponent=3, no
                        fouling, no part-load, linear weather), then
                        XGBoost trained on the residual.
                        Final = physics_pred + xgb_residual_pred.

Physics baseline formula (used in model 5)
------------------------------------------
  duration_day  = distance_nm / speed_kn / HOURS_PER_DAY
  vlsfo_equiv_t = k * (speed / design_speed)^3 * duration_day
                  * load_factor * weather_factor        [linear, no interaction]
  fuel_t        = vlsvo_equiv_t * (LHV_VLSFO / LHV_fuel) / eff_ratio

  The generator uses vessel-class-specific exponents (2.7–3.3) + hidden
  effects, so this baseline systematically underfits — residuals carry
  learnable signal.

Outputs
-------
  results/prediction_metrics.csv   — one row per model
  DB table prediction_metrics      — same rows (appended each run)
"""

from __future__ import annotations

import sys
import pathlib

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import statsmodels.formula.api as smf
import xgboost as xgb
from sqlalchemy.orm import Session

import config
import db as database

RANDOM_SEED: int = config.MODEL_RANDOM_SEED


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load voyages parquet; raise if not found."""
    if not config.VOYAGES_PARQUET.exists():
        raise FileNotFoundError(
            f"Voyages file not found: {config.VOYAGES_PARQUET}\n"
            "Run: python data/generate_data.py"
        )
    return pd.read_parquet(config.VOYAGES_PARQUET)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    One-hot encode vessel_class and fuel_type; return (X, y).

    X columns: speed_kn, load_factor, weather_factor, distance_nm,
               vc_<class>..., ft_<fuel>...
    y        : fuel_consumed_t (float array)
    """
    y = df["fuel_consumed_t"].values

    dummies_vc = pd.get_dummies(df["vessel_class"], prefix="vc",
                                 drop_first=False)
    dummies_ft = pd.get_dummies(df["fuel_type"],    prefix="ft",
                                 drop_first=False)

    X = pd.concat(
        [df[["speed_kn", "load_factor", "weather_factor",
             "distance_nm"]].reset_index(drop=True),
         dummies_vc.reset_index(drop=True),
         dummies_ft.reset_index(drop=True)],
        axis=1,
    )
    return X, y


# ---------------------------------------------------------------------------
# Physics baseline (vectorised)
# ---------------------------------------------------------------------------

def compute_physics_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorised analytic physics prediction.

    Uses FIXED exponent=3, NO hull fouling, NO part-load penalty,
    LINEAR weather_factor (no speed interaction).

    formula:
      duration_day  = distance_nm / speed_kn / HOURS_PER_DAY
      vlsfo_equiv   = k * (speed_kn / design_speed)^3 * duration_day
                      * load_factor * weather_factor
      fuel_t        = vlsfo_equiv * (LHV_VLSFO / LHV_fuel) / eff_ratio
    """
    vc_map_design = {c: s["design_speed_kn"]
                     for c, s in config.VESSEL_CLASSES.items()}
    vc_map_k      = {c: s["ref_daily_fuel_t"]
                     for c, s in config.VESSEL_CLASSES.items()}

    design_spd = df["vessel_class"].map(vc_map_design).values.astype(float)
    k_vals     = df["vessel_class"].map(vc_map_k).values.astype(float)
    lhv_fuel   = df["fuel_type"].map(config.LHV_MJ_PER_KG).values.astype(float)
    eff_ratio  = df["fuel_type"].map(config.ENGINE_EFFICIENCY_RATIO).values.astype(float)

    spd = df["speed_kn"].values.astype(float)
    dur = df["distance_nm"].values.astype(float) / spd / config.HOURS_PER_DAY
    wf  = df["weather_factor"].values.astype(float)
    lf  = df["load_factor"].values.astype(float)

    vlsfo_equiv = (
        k_vals
        * (spd / design_spd) ** config.ADMIRALTY_SPEED_EXPONENT
        * dur
        * lf
        * wf
    )
    fuel_t = vlsfo_equiv * (config.LHV_VLSFO_MJ_PER_KG / lhv_fuel) / eff_ratio
    return np.maximum(fuel_t, 1e-6)


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------

def train_ols(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    statsmodels OLS on log-transformed target.

    Note on retransformation bias: exp(E[log y]) ≤ E[y] by Jensen's
    inequality.  We report the naive back-transform (exp of fitted log values)
    which is consistent with the other models' MAE/RMSE comparisons.
    """
    formula = (
        "np.log(fuel_consumed_t) ~ "
        "np.log(speed_kn) + np.log(distance_nm) + "
        "load_factor + weather_factor + "
        "C(vessel_class) + C(fuel_type)"
    )
    model   = smf.ols(formula, data=train_df).fit(disp=False)
    log_hat = model.predict(test_df)
    y_pred  = np.exp(log_hat.values)
    y_true  = test_df["fuel_consumed_t"].values
    return {"model_name": "OLS_Baseline", **compute_metrics(y_true, y_pred)}


def train_linear(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test:  pd.DataFrame, y_test:  np.ndarray,
) -> dict:
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {"model_name": "LinearRegression",
            **compute_metrics(y_test, y_pred)}


def train_random_forest(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test:  pd.DataFrame, y_test:  np.ndarray,
) -> dict:
    model = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {"model_name": "RandomForest",
            **compute_metrics(y_test, y_pred)}


def train_xgboost(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test:  pd.DataFrame, y_test:  np.ndarray,
) -> dict:
    model = xgb.XGBRegressor(**config.XGBOOST_DEFAULT_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {"model_name": "XGBoost",
            **compute_metrics(y_test, y_pred)}


def train_physics_xgboost(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    X_train:  pd.DataFrame, X_test:  pd.DataFrame,
) -> dict:
    """
    Physics analytic prediction + XGBoost on the residual.

    Step 1: Compute analytic physics prediction for train and test sets
            (fixed exponent=3, linear weather, no hidden effects).
    Step 2: Compute residual = actual - physics for training rows.
    Step 3: Fit XGBoost to learn that residual.
    Step 4: Final prediction = physics_test + xgb_residual_test.

    The XGBoost captures the systematic errors introduced by hidden
    effects (class-specific exponent, fouling, part-load, weather
    interaction) that the physics formula ignores.
    """
    phys_train = compute_physics_baseline(train_df)
    phys_test  = compute_physics_baseline(test_df)

    resid_train = train_df["fuel_consumed_t"].values - phys_train

    model = xgb.XGBRegressor(**config.XGBOOST_DEFAULT_PARAMS)
    model.fit(X_train, resid_train)

    y_pred = phys_test + model.predict(X_test)
    y_true = test_df["fuel_consumed_t"].values
    return {"model_name": "Physics+XGBoost",
            **compute_metrics(y_true, y_pred)}


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

def _write_metrics_to_db(results_df: pd.DataFrame) -> None:
    engine = database.get_engine()
    database.Base.metadata.create_all(engine)
    with Session(engine) as session:
        for _, row in results_df.iterrows():
            session.add(database.PredictionMetric(
                model_name=str(row["model_name"]),
                mae=float(row["mae"]),
                rmse=float(row["rmse"]),
                r2=float(row["r2"]),
            ))
        session.commit()
    print("[fuel_model] Metrics written to DB table prediction_metrics.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    """Train all five models; save metrics to CSV and DB. Return results df."""

    print("[fuel_model] Loading voyage data ...")
    df = load_data()
    df = df.reset_index(drop=True)
    print(f"[fuel_model] {len(df):,} voyages loaded.")

    # ------ Split -----------------------------------------------------------
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=config.TRAIN_TEST_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    # Build feature matrices from the full df then subset (ensures same columns)
    X_all, y_all = prepare_features(df)
    X_train = X_all.iloc[train_idx].reset_index(drop=True)
    X_test  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train = y_all[train_idx]
    y_test  = y_all[test_idx]

    results: list[dict] = []

    print("[fuel_model] Training OLS baseline ...")
    results.append(train_ols(train_df, test_df))

    print("[fuel_model] Training Linear Regression ...")
    results.append(train_linear(X_train, y_train, X_test, y_test))

    print("[fuel_model] Training Random Forest ...")
    results.append(train_random_forest(X_train, y_train, X_test, y_test))

    print("[fuel_model] Training XGBoost ...")
    results.append(train_xgboost(X_train, y_train, X_test, y_test))

    print("[fuel_model] Training Physics + XGBoost residual ...")
    results.append(train_physics_xgboost(train_df, test_df, X_train, X_test))

    results_df = pd.DataFrame(results)
    results_df["run_at"] = datetime.now(timezone.utc).isoformat()

    # ------ Save CSV --------------------------------------------------------
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.METRICS_CSV, index=False)
    print(f"[fuel_model] Metrics CSV -> {config.METRICS_CSV}")

    # ------ Save DB ---------------------------------------------------------
    _write_metrics_to_db(results_df)

    return results_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run()
    print("\n=== Prediction Metrics (held-out 20% test set) ===")
    fmt = results[["model_name", "mae", "rmse", "r2"]].copy()
    fmt["mae"]  = fmt["mae"].map("{:.4f}".format)
    fmt["rmse"] = fmt["rmse"].map("{:.4f}".format)
    fmt["r2"]   = fmt["r2"].map("{:.4f}".format)
    print(fmt.to_string(index=False))
