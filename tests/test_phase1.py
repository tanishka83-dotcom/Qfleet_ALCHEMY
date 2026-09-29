"""
QFleet Phase 1 — Test Suite
===========================
Validates:
  1. Database creation and seeding (tables, rows, idempotence)
  2. Synthetic voyage generation (row count, columns, parquet metadata)
  3. Prediction metrics output (5 models, valid metric values)
  4. QI-EA comparison output (3 methods, valid metric values)
  5. Seeded reproducibility (identical DataFrames with same seed)
  6. Dynamic computation verification (no hardcoded metrics)
"""

import sys
import pathlib
import pytest
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
import db as database
from data.generate_data import generate_voyages


def test_db_creates():
    """Verify DB initialization and seeding with required tables and rows."""
    engine = database.get_engine()
    database.init_db()
    database.seed_db()

    with database.Session(engine) as session:
        vessel_count = session.query(database.Vessel).count()
        fuel_count = session.query(database.Fuel).count()
        route_count = session.query(database.Route).count()

        assert vessel_count > 0, "No vessels found in database"
        assert fuel_count > 0, "No fuels found in database"
        assert route_count > 0, "No routes found in database"


def test_voyages_generated():
    """Verify synthetic voyages parquet exists, has >=5000 rows, and metadata."""
    parquet_path = config.VOYAGES_PARQUET
    assert parquet_path.exists(), f"Parquet file {parquet_path} does not exist. Run data/generate_data.py first."

    # Verify table and row count
    table = pq.read_table(parquet_path)
    assert table.num_rows >= 5000, f"Expected >= 5000 rows, found {table.num_rows}"

    # Verify parquet metadata
    meta = table.schema.metadata or {}
    meta_decoded = {
        (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
        for k, v in meta.items()
    }
    assert meta_decoded.get("DATA_IS_SYNTHETIC") == "true", "Parquet missing DATA_IS_SYNTHETIC='true' metadata"

    # Verify key columns
    df = table.to_pandas()
    required_cols = [
        "voyage_id", "vessel_class", "fuel_type", "route_name",
        "speed_kn", "load_factor", "weather_factor", "distance_nm",
        "fuel_consumed_t", "co2_ttw_t", "co2_wtw_t"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_metrics_file_exists():
    """Verify results/prediction_metrics.csv exists, has 5 rows and proper metrics."""
    metrics_path = config.METRICS_CSV
    assert metrics_path.exists(), f"Metrics file {metrics_path} does not exist. Run models/fuel_model.py first."

    df = pd.read_csv(metrics_path)
    assert len(df) == 5, f"Expected 5 model metrics rows, found {len(df)}"

    required_cols = ["model_name", "mae", "rmse", "r2"]
    for col in required_cols:
        assert col in df.columns, f"Missing column in prediction_metrics.csv: {col}"

    expected_models = {
        "OLS_Baseline",
        "LinearRegression",
        "RandomForest",
        "XGBoost",
        "Physics+XGBoost",
    }
    assert set(df["model_name"]) == expected_models, f"Model names mismatch: {set(df['model_name'])}"

    for col in ["mae", "rmse"]:
        assert (df[col] > 0).all(), f"Metrics in {col} should be positive"


def test_qi_comparison_exists():
    """Verify results/qi_comparison.csv exists, has >= 3 rows and valid columns."""
    qi_path = config.QI_COMPARISON_CSV
    assert qi_path.exists(), f"QI comparison file {qi_path} does not exist. Run models/qi_feature_select.py first."

    df = pd.read_csv(qi_path)
    assert len(df) >= 3, f"Expected >= 3 comparison rows, found {len(df)}"

    # Check for algorithm column (or method)
    algo_col = "algorithm" if "algorithm" in df.columns else "method"
    assert algo_col in df.columns, "Missing algorithm/method column in qi_comparison.csv"

    # Check for MAE columns
    mae_col = "mean_test_mae" if "mean_test_mae" in df.columns else "test_mae_mean"
    assert mae_col in df.columns, "Missing mean test MAE column in qi_comparison.csv"

    assert (df[mae_col] > 0).all(), "Test MAE mean values must be positive"


def test_seeded_reproducibility():
    """Verify generate_voyages() produces identical DataFrames when called with identical seeds."""
    seed = 12345
    df1 = generate_voyages(n=500, seed=seed)
    df2 = generate_voyages(n=500, seed=seed)

    pd.testing.assert_frame_equal(df1, df2)


def test_no_hardcoded_results():
    """Verify metrics in results files are not fake/dummy constants."""
    metrics_path = config.METRICS_CSV
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        # Verify r2 values are floating point numbers not all identical
        assert df["r2"].nunique() > 1, "R2 scores appear to be hardcoded identical values"
        assert not (df["mae"] == 0).any(), "MAE values must not be zero"
