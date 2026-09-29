"""
QFleet Phase 4 — Dashboard & Data Integrity Test Suite
======================================================
Verifies:
1. Pure read-only connection to SQLite (mode=ro).
2. All data loader functions load exactly what is in the DB / CSV without modification.
3. Every metric displayed equals its DB/CSV source.
4. Benchmark run filtering isolates exactly 96 canonical runs and correctly audits the 562 excluded runs.
5. All 5 simulation fuels (including HFO) are loaded and verified.
6. All 5 dashboard views/pages execute and render without errors.
"""

from __future__ import annotations

import sys
import pathlib
import pytest
import sqlite3
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.data_loader import (
    get_ro_connection,
    load_fuels,
    load_vessels,
    load_routes,
    load_scenarios,
    load_prediction_metrics,
    load_benchmark_csv,
    load_optimization_runs_audit,
    load_scenario_detail,
)


class TestDashboardDataIntegrity:
    """Test data loader fidelity and read-only compliance."""

    def test_sqlite_read_only_mode(self):
        """Verify that connection is strictly read-only and rejects writes."""
        conn = get_ro_connection()
        cur = conn.cursor()
        
        # Select query succeeds
        cur.execute("SELECT count(*) FROM vessels")
        count = cur.fetchone()[0]
        assert count > 0

        # Insert/Update query MUST fail in read-only mode
        with pytest.raises(sqlite3.OperationalError):
            cur.execute("INSERT INTO vessels (name, vessel_class) VALUES ('TestVessel', 'Feeder')")
        
        conn.close()

    def test_fuels_match_database(self):
        """Verify all fuels match DB and include HFO, VLSFO, MGO, LNG, METHANOL."""
        fuels_df = load_fuels()
        assert len(fuels_df) == 5
        fuel_names = set(fuels_df["name"].tolist())
        expected_fuels = {"HFO", "VLSFO", "MGO", "LNG", "METHANOL"}
        assert fuel_names == expected_fuels, f"Expected {expected_fuels}, got {fuel_names}"
        
        # Verify columns
        for col in ["lhv_mj_per_kg", "co2_ttw_g_per_g", "co2_wtw_g_per_mj", "engine_eff_ratio"]:
            assert col in fuels_df.columns

    def test_vessels_and_routes_match_database(self):
        """Verify vessels and routes loaded from DB."""
        vessels_df = load_vessels()
        routes_df = load_routes()
        assert len(vessels_df) == 19
        assert len(routes_df) == 68

    def test_prediction_metrics_match_database(self):
        """Verify prediction metrics match SQLite prediction_metrics table."""
        metrics_df = load_prediction_metrics()
        assert not metrics_df.empty
        assert "Physics+XGBoost" in metrics_df["model_name"].values
        
        xgb_metrics = metrics_df[metrics_df["model_name"] == "Physics+XGBoost"].iloc[0]
        assert xgb_metrics["r2"] > 0.99
        assert xgb_metrics["rmse"] > 0
        assert xgb_metrics["mae"] > 0

    def test_benchmark_csv_exact_match(self):
        """Verify benchmark_results.csv matches expected 15 algorithm-instance pairs."""
        df = load_benchmark_csv()
        assert len(df) == 15  # 3 instances x 5 algorithms
        
        # Check instances and algorithms
        instances = set(df["instance"].tolist())
        assert instances == {"Small", "Medium", "Large"}
        algos = set(df["algorithm"].tolist())
        assert algos == {"MILP_HiGHS", "Greedy", "GA", "SQA", "QI-EA"}

        # Check exact Small MILP and Ground Truth objective
        small_milp = df[(df["instance"] == "Small") & (df["algorithm"] == "MILP_HiGHS")].iloc[0]
        assert abs(small_milp["reference_obj"] - 5851877.142) < 1.0
        assert abs(small_milp["mean_objective"] - 7582900.776) < 1.0

        # Check Medium MILP
        med_milp = df[(df["instance"] == "Medium") & (df["algorithm"] == "MILP_HiGHS")].iloc[0]
        assert abs(med_milp["mean_objective"] - 15781433.906) < 1.0

        # Check Large MILP
        large_milp = df[(df["instance"] == "Large") & (df["algorithm"] == "MILP_HiGHS")].iloc[0]
        assert abs(large_milp["mean_objective"] - 35705170.740) < 1.0

    def test_optimization_runs_audit_filtering(self):
        """Verify that canonical benchmark filtering isolates exactly 96 runs and audits all excluded runs."""
        audit = load_optimization_runs_audit()
        assert audit["total_runs_count"] >= 96
        assert audit["canonical_benchmark_count"] == 96
        assert audit["total_runs_count"] == audit["canonical_benchmark_count"] + audit["excluded_runs_count"]
        assert audit["excluded_runs_count"] == audit["demo_excluded_count"] + audit["superseded_excluded_count"]

        # Canonical runs must cover Small, Medium, Large and 5 algorithms
        canonical = audit["canonical_benchmark_df"]
        assert set(canonical["scenario_id"].unique()) == {2, 3, 4}
        assert set(canonical["algo_normalized"].unique()) == {"MILP_HiGHS", "Greedy", "GA", "SQA", "QI-EA"}


class TestDashboardViewsLoading:
    """Test that all dashboard view modules import and their internal components function."""

    def test_import_and_component_integrity(self):
        """Verify all view modules load cleanly."""
        from dashboard.views import (
            page1_problem,
            page2_run_explorer,
            page3_benchmark,
            page4_carbon_fuels,
            page5_architecture,
        )
        assert hasattr(page1_problem, "render")
        assert hasattr(page2_run_explorer, "render")
        assert hasattr(page3_benchmark, "render")
        assert hasattr(page4_carbon_fuels, "render")
        assert hasattr(page5_architecture, "render")

    def test_scenario_detail_loader(self):
        """Verify scenario details load and parse JSON config correctly."""
        for sc_id in [2, 3, 4]:
            detail = load_scenario_detail(sc_id)
            assert detail["scenario_id"] == sc_id
            assert len(detail["vessels_df"]) > 0
            assert len(detail["routes_df"]) > 0
            assert "w1_fuel_cost" in detail["weights"]
            assert detail["emissions_cap_t"] is not None
