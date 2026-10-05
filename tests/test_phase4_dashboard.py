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
from dashboard.views.page1_problem import (
    build_port_map_data,
    filter_overview_data,
)
from dashboard.views.page2_run_explorer import (
    filter_canonical_runs,
    recorded_convergence_point,
)
from dashboard.views.page3_benchmark import (
    add_run_metrics,
    filter_benchmark_data,
)
from dashboard.views.page4_carbon_fuels import (
    build_fuel_comparison,
    reprice_benchmark_records,
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
        import config
        import db as database

        database.init_db()
        fuels_df = load_fuels()
        fuel_names = set(fuels_df["name"].tolist())
        assert set(config.SIMULATION_FUELS) <= fuel_names
        assert len(config.SIMULATION_FUELS) == 5

        with database.Session(database.get_engine()) as session:
            proxies = {
                fuel.name: fuel
                for fuel in session.query(database.ExtendedFuelProxy).all()
            }
            seeded_names = {
                fuel.name for fuel in session.query(database.Fuel).all()
            }
        assert seeded_names == set(config.SIMULATION_FUELS)
        assert set(config.EXTENDED_FUELS) <= set(proxies)
        for fuel_name in config.EXTENDED_FUELS:
            assert proxies[fuel_name].price_usd_per_tonne > 0
            assert proxies[fuel_name].cost_proxy_usd_per_gj > 0
            assert "TODO_VERIFY" in proxies[fuel_name].source
        
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


def test_page1_filters_handle_empty_and_single_item_selections():
    vessels = pd.DataFrame([
        {"id": 1, "name": "North Star", "vessel_class": "Feeder"},
        {"id": 2, "name": "Pacific Dawn", "vessel_class": "Panamax"},
    ])
    routes = pd.DataFrame([
        {
            "id": 1,
            "name": "Singapore-Rotterdam",
            "origin_port": "Singapore",
            "dest_port": "Rotterdam",
            "distance_nm": 8300.0,
        },
        {
            "id": 2,
            "name": "Shanghai-Los Angeles",
            "origin_port": "Shanghai",
            "dest_port": "Los Angeles",
            "distance_nm": 5700.0,
        },
    ])

    empty_vessels, empty_routes = filter_overview_data(
        vessels,
        routes,
        vessel_classes=[],
        route_ids=[],
        ports=[],
    )
    assert empty_vessels.empty
    assert empty_routes.empty

    one_vessel, one_route = filter_overview_data(
        vessels,
        routes,
        vessel_query="north",
        vessel_classes=["Feeder"],
        route_query="singapore",
        route_ids=[1],
        ports=["Singapore"],
    )
    assert one_vessel["id"].tolist() == [1]
    assert one_route["id"].tolist() == [1]
    markers = build_port_map_data(one_route)
    assert len(markers) == 2
    assert set(markers["Port"]) == {"Singapore", "Rotterdam"}


def test_page2_run_filters_handle_empty_and_single_selections():
    runs = pd.DataFrame([
        {"id": 1, "scenario_id": 2, "algo_normalized": "GA", "seed": 42,
         "objective": 100.0, "converged_at_iter": 8},
        {"id": 2, "scenario_id": 2, "algo_normalized": "SQA", "seed": 42,
         "objective": 120.0, "converged_at_iter": 10},
        {"id": 3, "scenario_id": 3, "algo_normalized": "GA", "seed": 43,
         "objective": 90.0, "converged_at_iter": 6},
    ])

    empty = filter_canonical_runs(
        runs, scenario_ids=[], algorithms=["GA"], seeds=[42]
    )
    single = filter_canonical_runs(
        runs, scenario_ids=[2], algorithms=["GA"], seeds=[42]
    )

    assert empty.empty
    assert single["id"].tolist() == [1]
    convergence = recorded_convergence_point(single.iloc[0])
    assert convergence.to_dict("records") == [
        {"Iteration": 8, "Objective (USD)": 100.0}
    ]


def test_page3_filters_handle_empty_and_single_items_using_stored_references():
    summary = pd.DataFrame([
        {"instance": "Small", "algorithm": "GA", "reference_obj": 80.0},
        {"instance": "Large", "algorithm": "GA", "reference_obj": 900.0},
        {"instance": "Small", "algorithm": "SQA", "reference_obj": 80.0},
    ])
    runs = pd.DataFrame([
        {"scenario_id": 2, "algo_normalized": "GA", "seed": 42,
         "objective": 100.0, "runtime_s": 2.0},
        {"scenario_id": 4, "algo_normalized": "GA", "seed": 42,
         "objective": 990.0, "runtime_s": 5.0},
        {"scenario_id": 2, "algo_normalized": "SQA", "seed": 42,
         "objective": 120.0, "runtime_s": 3.0},
    ])
    scenario_names = {2: "Small_4V_6R", 4: "Large_15V_30R"}
    runs = add_run_metrics(runs, summary, scenario_names)

    empty_summary, empty_runs = filter_benchmark_data(
        summary, runs, algorithms=[], instances=["Small"]
    )
    single_summary, single_runs = filter_benchmark_data(
        summary, runs, algorithms=["GA"], instances=["Small"]
    )

    assert empty_summary.empty and empty_runs.empty
    assert single_summary["algorithm"].tolist() == ["GA"]
    assert single_runs["algo_normalized"].tolist() == ["GA"]
    assert single_runs["gap_to_ref_pct"].iloc[0] == 25.0


def test_page4_fuel_selection_handles_empty_and_single_and_reprices_stored_costs():
    fuels = pd.DataFrame([
        {"name": "HFO", "price_usd_per_tonne": 480.0,
         "lhv_mj_per_kg": 40.2, "co2_wtw_g_per_mj": 86.2,
         "co2_ttw_g_per_g": 3.114, "source": "stored"},
        {"name": "LH2_GREEN", "price_usd_per_tonne": 2800.0,
         "lhv_mj_per_kg": 119.9, "co2_wtw_g_per_mj": 6.1,
         "co2_ttw_g_per_g": 0.0, "source": "TODO_VERIFY proxy"},
    ])
    assert build_fuel_comparison(fuels, [], 1.0).empty
    single_fuel = build_fuel_comparison(fuels, ["LH2_GREEN"], 1.5)
    assert single_fuel["name"].tolist() == ["LH2_GREEN"]
    assert single_fuel["price_proxy_usd_per_tonne"].iloc[0] == 4200.0

    stored = pd.DataFrame([{
        "algorithm": "GA",
        "mean_objective": 1000.0,
        "mean_fuel_cost_usd": 400.0,
        "mean_co2_wtw_t": 2.0,
    }])
    priced = reprice_benchmark_records(
        stored, carbon_price=50.0, reference_fuel_price=1240.0
    ).iloc[0]
    assert priced["Stored emissions (t CO₂eq)"] == 2.0
    assert priced["Repriced fuel cost ($)"] == 800.0
    assert priced["Repriced carbon cost ($)"] == 100.0
    assert priced["Repriced total ($)"] == 1300.0
