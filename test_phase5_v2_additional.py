"""
QFleet Phase 5 — Additional Isolated Tests
==========================================
Verifies Problem V2, 3 Route Options, Shore Power, Shared Cap-Repair Operator,
Plan a Fleet CSV Validation, and Temporary Database Isolation.
"""

import io
import sqlite3
import pandas as pd
import pytest
from sqlalchemy import create_engine

import config
import db as database
from dashboard.views import page6_plan_fleet
from optimizer.problem_v2 import FleetOptimizationProblemV2, PORT_SHORE_PROFILES, VESSEL_AUX_POWER_KW
from optimizer.algorithms_v2 import apply_shared_cap_repair, repair_solution_v2
from phase5_todo_verify import register_phase5_todo_items
import web.api as api


def _create_test_problem(emissions_cap_t=None):
    route = {
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
            {"option_id": 2, "name": "ECA-Minimizing", "dist_factor": 1.06, "weather_factor": 1.00, "eca_fraction": 0.05},
        ],
    }
    vessel = {
        "id": 1,
        "name": "Feeder-01",
        "vessel_class": "Feeder",
        "capacity_teu": 1500,
        "teu_capacity": 1500,
        "design_speed_kn": 18.0,
        "ref_daily_fuel_t": 28.0,
    }
    fuels = [
        {"id": 1, "name": "HFO", "price_usd_per_tonne": 580.0, "lhv_mj_per_kg": 40.5, "co2_wtw_g_per_mj": 91.0},
        {"id": 2, "name": "LNG", "price_usd_per_tonne": 720.0, "lhv_mj_per_kg": 49.0, "co2_wtw_g_per_mj": 68.0},
    ]
    return FleetOptimizationProblemV2(
        routes=[route],
        vessels=[vessel],
        fuels=fuels,
        emissions_cap_t=emissions_cap_t,
        vessel_limits={"Feeder-01": 1},
    )


def test_problem_v2_route_options_and_shore_power():
    """Verify Problem V2 precomputes candidate route options and at-berth shore power."""
    problem = _create_test_problem()
    assert problem.num_routes == 1
    assert len(problem._lookup_cache) == 3 * 1 * 5 * 2

    direct_entry = problem._lookup_cache[(1, 0, 1, 14.4, 1)]
    weather_entry = problem._lookup_cache[(1, 1, 1, 14.4, 1)]
    eca_entry = problem._lookup_cache[(1, 2, 1, 14.4, 1)]

    assert direct_entry["used_shore_power"] is True
    assert direct_entry["port_cost"] > 0
    assert weather_entry["fuel_t"] > 0
    assert eca_entry["total_co2_t"] > 0


def test_problem_v2_shared_cap_repair():
    """Verify apply_shared_cap_repair shifts fuel to reduce emissions under tight cap."""
    problem = _create_test_problem(emissions_cap_t=40.0)
    initial_plan = [{
        "route_id": 1,
        "route_name": "Test Route",
        "route_option_id": 0,
        "vessel_id": 1,
        "vessel_name": "Feeder-01",
        "vessel_class": "Feeder",
        "speed_kn": 14.4,
        "fuel_id": 1,  # HFO
        "fuel_type": "HFO",
    }]
    repaired_plan = apply_shared_cap_repair(problem, initial_plan)
    assert repaired_plan[0]["fuel_id"] == 2  # Shifted to LNG (lower CO2)


def test_plan_fleet_page_entrypoint_and_csv_validation():
    """Verify CSV validation rules and Plan a fleet entry point."""
    assert callable(page6_plan_fleet.render)
    
    # Valid CSV
    valid = "name,origin_port,dest_port,distance_nm,cargo_demand_teu,deadline_days\nR1,A,B,10,100,2\n"
    frame, error = page6_plan_fleet.validate_fleet_csv(io.StringIO(valid))
    assert error is None
    assert len(frame) == 1

    # Missing column
    invalid = "name,origin_port,dest_port,distance_nm\nR1,A,B,10\n"
    frame, error = page6_plan_fleet.validate_fleet_csv(io.StringIO(invalid))
    assert frame is None
    assert "Missing required columns" in error

    # Non-numeric value
    bad_val = "name,origin_port,dest_port,distance_nm,cargo_demand_teu,deadline_days\nR1,A,B,not_a_num,100,2\n"
    frame, error = page6_plan_fleet.validate_fleet_csv(io.StringIO(bad_val))
    assert frame is None
    assert "numeric" in error


def test_api_summary_uses_temporary_database(tmp_path, monkeypatch):
    """Verify API summary uses isolated temporary database."""
    db_path = tmp_path / "api-test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    database.Base.metadata.create_all(engine)
    engine.dispose()
    monkeypatch.setattr(api, "_DB_PATH", db_path)

    summary = api._build_summary()
    assert summary["optimization_run_count"] == 0


def test_phase5_todo_registry_and_read_only_results_api(tmp_path, monkeypatch):
    """Verify TODO_VERIFY registry and read-only results CSV reading."""
    items = register_phase5_todo_items(config)
    assert len(items) == len(set(items))
    assert len(items) == len(config.TODO_VERIFY_ITEMS)
    assert len(items) >= 38

    result_path = tmp_path / "phase5_v2.csv"
    pd.DataFrame([{"instance": "Small_v2", "method": "MILP", "is_feasible": True}]).to_csv(result_path, index=False)
    monkeypatch.setattr(api, "_PHASE5_V2_CSV", result_path)
    result = api._build_phase5_v2_benchmark()
    assert result["rows"][0]["instance"] == "Small_v2"
    assert result["rows"][0]["method"] == "MILP"
