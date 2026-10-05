import json

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

import config
import db as database
from models import prediction_benchmark, qi_feature_select
from sqlalchemy import inspect


def test_prediction_benchmark_table_is_additive():
    engine = database.get_engine()
    database.Base.metadata.create_all(engine)

    inspector = inspect(engine)
    legacy_columns = {column["name"] for column in inspector.get_columns("prediction_metrics")}
    benchmark_columns = {column["name"] for column in inspector.get_columns("prediction_benchmarks")}

    assert {"model_name", "mae", "rmse", "r2", "run_at"} <= legacy_columns
    assert {
        "model_name", "split_name", "mae", "rmse", "r2",
        "y_true_json", "y_pred_json", "run_at",
    } <= benchmark_columns


def _small_voyages() -> pd.DataFrame:
    count = 30
    return pd.DataFrame({
        "speed_kn": np.linspace(10.0, 22.0, count),
        "load_factor": np.linspace(0.5, 0.95, count),
        "weather_factor": np.linspace(1.0, 1.3, count),
        "distance_nm": np.linspace(200.0, 2000.0, count),
        "vessel_class": np.resize(["Feeder", "Panamax", "ULCS"], count),
        "fuel_type": np.resize(["HFO", "MGO"], count),
        "fuel_consumed_t": np.linspace(20.0, 150.0, count),
    })


def test_benchmark_splits_hold_out_top_speeds_and_each_vessel_class():
    df = _small_voyages()
    splits = prediction_benchmark.build_splits(df)

    speed_train, speed_test = splits["Highest-speed holdout"]
    assert df.iloc[speed_train]["speed_kn"].max() <= df.iloc[speed_test]["speed_kn"].min()

    for vessel_class in df["vessel_class"].unique():
        train_idx, test_idx = splits[f"Vessel class: {vessel_class}"]
        assert set(df.iloc[test_idx]["vessel_class"]) == {vessel_class}
        assert vessel_class not in set(df.iloc[train_idx]["vessel_class"])


def test_benchmark_persists_all_models_and_prediction_samples(monkeypatch):
    df = _small_voyages()

    def fake_predictions(train_df, test_df, X_train, X_test, y_train, seed):
        return {
            model_name: np.full(len(test_df), float(np.mean(y_train)))
            for model_name in prediction_benchmark.MODEL_NAMES
        }

    monkeypatch.setattr(prediction_benchmark, "_predict_models", fake_predictions)
    metrics = prediction_benchmark.run_benchmark(df)

    assert set(metrics["model_name"]) == set(prediction_benchmark.MODEL_NAMES)
    expected_splits = 2 + df["vessel_class"].nunique()
    assert len(metrics) == expected_splits * len(prediction_benchmark.MODEL_NAMES)
    with Session(database.get_engine()) as session:
        rows = session.query(database.PredictionBenchmark).all()
    assert len(rows) == len(metrics)
    for row in rows:
        actual = json.loads(row.y_true_json)
        predicted = json.loads(row.y_pred_json)
        assert len(actual) == len(predicted)
        assert actual


def test_three_baseline_models_return_finite_predictions(monkeypatch):
    count = 12
    X_train = pd.DataFrame({"speed_kn": np.arange(count), "load_factor": 0.8})
    X_test = pd.DataFrame({"speed_kn": [12, 13], "load_factor": [0.8, 0.9]})
    y_train = np.arange(count, dtype=float) * 2.0
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    monkeypatch.setattr(config, "XGBOOST_DEFAULT_PARAMS", {
        "n_estimators": 3, "max_depth": 2, "n_jobs": 1,
        "tree_method": "hist", "verbosity": 0,
    })
    monkeypatch.setattr(
        prediction_benchmark, "_predict_qi_physics_model",
        lambda *args: np.array([1.0, 2.0]),
    )

    predictions = prediction_benchmark._predict_models(
        train_df, test_df, X_train, X_test, y_train, 42
    )

    assert set(predictions) == set(prediction_benchmark.MODEL_NAMES)
    assert all(np.isfinite(values).all() for values in predictions.values())


def test_qiea_benchmark_selection_does_not_evaluate_holdout(monkeypatch):
    X_train = pd.DataFrame({"speed_kn": np.arange(8, dtype=float)})
    monkeypatch.setattr(qi_feature_select, "evaluate_cv", lambda *args, **kwargs: -1.0)
    monkeypatch.setattr(
        qi_feature_select, "evaluate_test",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("holdout used")),
    )

    result = qi_feature_select.qiea(
        X_train, np.arange(8, dtype=float), X_train.iloc[:0], np.empty(0),
        [], [], 42,
        search_params={
            "population_size": 2,
            "n_generations": 1,
            "n_cv_folds": 2,
            "inner_n_estimators": 2,
        },
        evaluate_holdout=False,
    )

    assert result["n_evaluations"] == 4
    assert np.isnan(result["test_mae"])