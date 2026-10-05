"""Compare voyage fuel predictors across random and distribution-shift splits."""

from __future__ import annotations

import ast
import json
import pathlib
import sys
from datetime import datetime, timezone

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

import config
import db as database
from models import fuel_model, qi_feature_select


QI_SEARCH_PARAMS = {
    "population_size": 8,
    "n_generations": 5,
    "n_cv_folds": 2,
    "inner_n_estimators": 60,
}
MODEL_NAMES = (
    "LinearRegression",
    "RandomForest",
    "Plain XGBoost",
    "Physics+XGBoost+QI Feature Selection",
)


def build_splits(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return train/test row indices for the random, speed, and class holdouts."""
    indices = np.arange(len(df))
    random_train, random_test = train_test_split(
        indices, train_size=0.8, random_state=config.MODEL_RANDOM_SEED,
        shuffle=True,
    )
    splits = {"Random 80/20": (random_train, random_test)}

    test_count = max(1, int(np.ceil(len(df) * 0.2)))
    speed_order = np.argsort(df["speed_kn"].to_numpy(), kind="stable")
    speed_test = speed_order[-test_count:]
    speed_train = speed_order[:-test_count]
    splits["Highest-speed holdout"] = (speed_train, speed_test)

    for vessel_class in sorted(df["vessel_class"].unique()):
        test_mask = df["vessel_class"].to_numpy() == vessel_class
        splits[f"Vessel class: {vessel_class}"] = (
            indices[~test_mask], indices[test_mask]
        )
    return splits


def _predict_qi_physics_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> np.ndarray:
    vc_cols = [col for col in X_train.columns if col.startswith("vc_")]
    ft_cols = [col for col in X_train.columns if col.startswith("ft_")]
    selection = qi_feature_select.qiea(
        X_train, y_train, X_train.iloc[:0], np.empty(0),
        vc_cols, ft_cols, seed, search_params=QI_SEARCH_PARAMS,
        evaluate_holdout=False,
    )
    selected_groups = set(selection["selected_features"].split(", "))
    bits = np.array([
        int(group in selected_groups)
        for group in qi_feature_select.FEATURE_GROUPS
    ])
    X_train_selected = qi_feature_select.apply_mask(
        X_train, bits, vc_cols, ft_cols
    )
    X_test_selected = qi_feature_select.apply_mask(
        X_test, bits, vc_cols, ft_cols
    )
    params = ast.literal_eval(selection["best_params"])
    physics_train = fuel_model.compute_physics_baseline(train_df)
    physics_test = fuel_model.compute_physics_baseline(test_df)
    residual_model = xgb.XGBRegressor(
        **params,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    residual_model.fit(
        X_train_selected,
        train_df["fuel_consumed_t"].to_numpy() - physics_train,
    )
    return physics_test + residual_model.predict(X_test_selected)


def _predict_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    random_forest = RandomForestRegressor(
        n_estimators=100, random_state=seed, n_jobs=-1
    )
    random_forest.fit(X_train, y_train)

    plain_xgboost = xgb.XGBRegressor(**config.XGBOOST_DEFAULT_PARAMS)
    plain_xgboost.fit(X_train, y_train)

    return {
        "LinearRegression": LinearRegression(n_jobs=-1).fit(
            X_train, y_train
        ).predict(X_test),
        "RandomForest": random_forest.predict(X_test),
        "Plain XGBoost": plain_xgboost.predict(X_test),
        "Physics+XGBoost+QI Feature Selection": _predict_qi_physics_model(
            train_df, test_df, X_train, X_test, y_train, seed
        ),
    }


def _save_results(results: list[dict], run_at: datetime) -> None:
    engine = database.get_engine()
    database.Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.query(database.PredictionBenchmark).delete()
        session.add_all([
            database.PredictionBenchmark(
                model_name=row["model_name"],
                split_name=row["split_name"],
                mae=row["mae"],
                rmse=row["rmse"],
                r2=row["r2"],
                y_true_json=json.dumps(row["y_true"]),
                y_pred_json=json.dumps(row["y_pred"]),
                run_at=run_at,
            )
            for row in results
        ])
        session.commit()


def run_benchmark(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run all four models for every split and persist metrics and predictions."""
    df = (fuel_model.load_data() if df is None else df).reset_index(drop=True)
    X_all, y_all = fuel_model.prepare_features(df)
    seed = config.MODEL_RANDOM_SEED
    rows: list[dict] = []

    for split_name, (train_idx, test_idx) in build_splits(df).items():
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        X_train = X_all.iloc[train_idx].reset_index(drop=True)
        X_test = X_all.iloc[test_idx].reset_index(drop=True)
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        predictions = _predict_models(
            train_df, test_df, X_train, X_test, y_train, seed
        )
        for model_name, y_pred in predictions.items():
            rows.append({
                "model_name": model_name,
                "split_name": split_name,
                **fuel_model.compute_metrics(y_test, y_pred),
                "y_true": y_test.astype(float).tolist(),
                "y_pred": np.asarray(y_pred, dtype=float).tolist(),
            })

    run_at = datetime.now(timezone.utc)
    _save_results(rows, run_at)
    return pd.DataFrame([
        {key: value for key, value in row.items()
         if key not in {"y_true", "y_pred"}}
        for row in rows
    ])


if __name__ == "__main__":
    benchmark = run_benchmark()
    print(benchmark.sort_values(["split_name", "mae"]).to_string(index=False))