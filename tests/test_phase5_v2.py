"""
QFleet Phase 5 — Problem v2, Plan a Fleet CSV Validation & Endpoint Tests
==========================================================================
Verifies problem_v2 mechanics, shore power, route options, CSV validation,
and optimization endpoints against an isolated temporary database.
"""

import io
import pytest
import sqlite3
import numpy as np
import pandas as pd

import config
import db as database
from benchmark.instances import generate_benchmark_scenarios
from optimizer.problem import FleetOptimizationProblem
from optimizer.problem_v2 import FleetOptimizationProblemV2, PORT_SHORE_PROFILES
from optimizer.baselines import milp_optimize
from optimizer.baselines_v2 import milp_optimize_v2
from dashboard.views.page6_plan_fleet import (
    build_fuels_data,
    results_csv_bytes,
    validate_fleet_csv,
)


def test_problem_v2_route_options_and_shore_power():
    """Verify Problem V2 includes 3 route options and shore power calculations."""
    routes = [{
        "id": 1,
        "name": "Test Route: Rotterdam -> Hamburg",
        "origin_port": "Rotterdam",
        "dest_port": "Hamburg",
        "distance_nm": 350.0,
        "cargo_demand_teu": 1000,
        "deadline_days": 2.5,
        "berth_hours": 24.0,
        "route_options": [
            {"option_id": 0, "name": "Direct", "dist_factor": 1.0, "weather_factor": 1.0, "eca_fraction": 0.15},
            {"option_id": 1, "name": "Weather-Optimized", "dist_factor": 1.04, "weather_factor": 0.92, "eca_fraction": 0.15},
            {"option_id": 2, "name": "ECA-Minimizing", "dist_factor": 1.06, "weather_factor": 1.0, "eca_fraction": 0.05},
        ]
    }]
    vessels = [{
        "id": 1,
        "name": "Feeder-Test",
        "vessel_class": "Feeder",
        "capacity_teu": 1500,
        "design_speed_kn": 18.0,
        "ref_daily_fuel_t": 25.0,
    }]
    fuels = [
        {"id": 1, "name": "HFO", "price_usd_per_tonne": 580.0, "lhv_mj_per_kg": 40.5, "co2_wtw_g_per_mj": 91.0},
        {"id": 2, "name": "LNG", "price_usd_per_tonne": 720.0, "lhv_mj_per_kg": 49.0, "co2_wtw_g_per_mj": 68.0},
    ]

    prob = FleetOptimizationProblemV2(routes=routes, vessels=vessels, fuels=fuels, emissions_cap_t=10000.0)
    assert prob.num_routes == 1
    assert len(prob._lookup_cache) == 3 * 1 * 5 * 2

    # Check that Hamburg port shore profile is used
    key_opt0 = (1, 0, 1, 14.4, 1)  # 80% speed of 18.0 = 14.4
    entry0 = prob._lookup_cache[key_opt0]
    assert entry0["port_cost"] > 0
    assert entry0["co2_port_t"] > 0
    assert entry0["used_shore_power"] is True


def test_problem_v2_extended_fuels_default_preserves_standard_results():
    routes = [{
        "id": 1,
        "name": "Default fuel regression",
        "origin_port": "Rotterdam",
        "dest_port": "Hamburg",
        "distance_nm": 350.0,
        "cargo_demand_teu": 1000,
        "deadline_days": 3.0,
        "route_options": [{
            "option_id": 0, "name": "Direct", "dist_factor": 1.0,
            "weather_factor": 1.0, "eca_fraction": 0.15,
        }],
    }]
    vessels = [{
        "id": 1, "name": "Test vessel", "vessel_class": "Feeder",
        "capacity_teu": 1500, "design_speed_kn": 18.0,
    }]
    standard_fuels = [
        {"id": 1, "name": "HFO", "price_usd_per_tonne": 580.0,
         "lhv_mj_per_kg": 40.2, "co2_wtw_g_per_mj": 86.2},
        {"id": 2, "name": "LNG", "price_usd_per_tonne": 720.0,
         "lhv_mj_per_kg": 48.0, "co2_wtw_g_per_mj": 75.7},
    ]
    extended = [
        {"id": 3, "name": "LH2_GREEN", "price_usd_per_tonne": 2800.0,
         "lhv_mj_per_kg": 119.9, "co2_wtw_g_per_mj": 6.1},
        {"id": 4, "name": "AMMONIA_GREEN", "price_usd_per_tonne": 1300.0,
         "lhv_mj_per_kg": 18.6, "co2_wtw_g_per_mj": 3.5},
    ]
    kwargs = {
        "routes": routes,
        "vessels": vessels,
        "weights": {"w1_fuel_cost": 1.0, "w2_co2_emission": 100.0},
        "speed_bins": [0.8],
    }
    baseline = FleetOptimizationProblemV2(
        **kwargs, fuels=standard_fuels, instance_name="Fuel-default-regression"
    )
    default_with_extended_input = FleetOptimizationProblemV2(
        **kwargs,
        fuels=standard_fuels + extended,
        instance_name="Fuel-default-regression",
    )

    assert default_with_extended_input.num_fuels == baseline.num_fuels == 2
    assert default_with_extended_input.fuels == baseline.fuels
    plan = [{
        "route_id": 1, "route_option_id": 0, "vessel_id": 1,
        "speed_kn": 14.4, "fuel_id": 1,
    }]
    assert default_with_extended_input.evaluate(plan) == baseline.evaluate(plan)

    enabled = FleetOptimizationProblemV2(
        **kwargs,
        fuels=standard_fuels + extended,
        instance_name="Fuel-extended-regression",
        extended_fuels=True,
    )
    assert enabled.num_fuels == 4
    assert {fuel["name"] for fuel in enabled.fuels} >= set(config.EXTENDED_FUELS)


def test_live_extended_fuel_records_use_finite_price_proxies():
    fuel_df = pd.DataFrame([
        {"id": 1, "name": "HFO", "price_usd_per_tonne": np.nan,
         "lhv_mj_per_kg": 40.2, "co2_wtw_g_per_mj": 86.2},
        {"id": 6, "name": "LH2_GREEN", "price_usd_per_tonne": 2800.0,
         "lhv_mj_per_kg": 119.9, "co2_wtw_g_per_mj": 6.1},
    ])

    result = build_fuels_data(
        fuel_df, config.SIMULATION_FUELS + config.EXTENDED_FUELS
    )

    assert result[0]["price_usd_per_tonne"] == config.BUNKER_PRICES_USD_PER_TONNE["HFO"]
    assert result[1]["price_usd_per_tonne"] == 2800.0
    assert all(np.isfinite(fuel["price_usd_per_tonne"]) for fuel in result)


def test_page6_results_csv_contains_displayed_summary():
    csv_bytes = results_csv_bytes(pd.DataFrame([
        {"Algorithm": "MILP", "Objective ($)": "$1,000", "Runtime (s)": "0.25"},
    ]))

    assert csv_bytes.startswith(b"Algorithm,Objective ($),Runtime (s)")
    assert b'MILP,"$1,000",0.25' in csv_bytes


def test_problem_v2_regression_against_v1():
    """Verify Problem V2 with option 0 and shore=0 reproduces v1 exactly."""
    conn = database.get_engine()
    with database.Session(conn) as session:
        sc_ids = generate_benchmark_scenarios(session)

    for inst_name in ["Small", "Medium", "Large"]:
        sc_id = sc_ids[inst_name]
        p1 = FleetOptimizationProblem(scenario_id=sc_id)
        sol1, obj1, _ = milp_optimize(p1)

        routes_v2 = []
        for r in p1.routes:
            routes_v2.append({
                "id": r["id"],
                "name": r["name"],
                "origin_port": r["origin_port"],
                "dest_port": r["dest_port"],
                "distance_nm": r["distance_nm"],
                "cargo_demand_teu": r["cargo_demand_teu"],
                "deadline_days": r["deadline_days"],
                "berth_hours": 0.0,
                "route_options": [{"option_id": 0, "name": "Direct", "dist_factor": 1.0, "weather_factor": 1.08, "eca_fraction": 0.15}]
            })
        p2 = FleetOptimizationProblemV2(
            routes=routes_v2,
            vessels=p1.vessels,
            fuels=p1.fuels,
            weights=p1.weights,
            emissions_cap_t=p1.emissions_cap_t,
            vessel_limits=p1.vessel_limits,
            instance_name=inst_name,
        )
        p2.speed_bins = [0.60, 0.70, 0.80, 0.90, 1.00]
        p2._precompute_lookup_table()
        sol2, obj2, det2 = milp_optimize_v2(p2)

        assert abs(obj1 - obj2) < 1e-2, f"Regression mismatch on {inst_name}: v1={obj1}, v2={obj2}"


def test_csv_validation_limits():
    """Verify CSV validation enforces schema and <= 50 routes limit."""
    # Case 1: Valid CSV
    valid_csv = """name,origin_port,dest_port,distance_nm,cargo_demand_teu,deadline_days
Route A,Rotterdam,Hamburg,350,1200,2.5
Route B,Shanghai,Rotterdam,10500,18000,22.0
"""
    df_valid, err = validate_fleet_csv(io.StringIO(valid_csv))
    assert err is None
    assert len(df_valid) == 2

    # Case 2: Missing required column
    invalid_csv = """name,origin_port,dest_port,distance_nm
Route A,Rotterdam,Hamburg,350
"""
    df_inv, err_msg = validate_fleet_csv(io.StringIO(invalid_csv))
    assert df_inv is None
    assert "Missing required columns" in err_msg

    # Case 3: Exceeds 50 routes limit
    rows = ["name,origin_port,dest_port,distance_nm,cargo_demand_teu,deadline_days"]
    for i in range(55):
        rows.append(f"Route-{i},Rotterdam,Hamburg,350,1000,2.5")
    overflow_csv = "\n".join(rows)
    df_over, err_over = validate_fleet_csv(io.StringIO(overflow_csv))
    assert df_over is None
    assert "exceeds maximum allowed limit of 50" in err_over


def test_production_db_isolation():
    """Verify that test execution never touches production qfleet.db (stays 658 rows)."""
    import pathlib
    prod_path = pathlib.Path("data/qfleet.db")
    prod_conn = sqlite3.connect(f"file:{prod_path}?mode=ro", uri=True)
    cursor = prod_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM optimization_runs")
    count = cursor.fetchone()[0]
    prod_conn.close()
    assert count == 658, f"Production optimization_runs count altered! Expected 658, got {count}"

