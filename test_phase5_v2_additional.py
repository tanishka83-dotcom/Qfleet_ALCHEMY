"""Additional Phase 5 checks kept outside the protected tests/ directory."""

import io
import sqlite3

import pandas as pd
import pytest
from sqlalchemy import create_engine

import config
import db as database
from dashboard.views import page6_plan_fleet
from optimizer.problem_v2 import FleetOptimizationProblemV2
from phase5_todo_verify import register_phase5_todo_items
import web.api as api


def _minimal_problem():
    route = {
        "id": 1,
        "name": "Test Route",
        "origin_port": "Rotterdam",
        "dest_port": "Hamburg",
        "distance_nm": 350.0,
        "cargo_demand_teu": 1000,
        "deadline_days": 2.5,
        "berth_hours": 24.0,
        "route_options": [
            {"option_id": 0, "name": "Direct", "dist_factor": 1.0, "weather_factor": 1.0, "eca_fraction": 0.15},
            {"option_id": 1, "name": "ECA-Minimizing", "dist_factor": 1.06, "weather_factor": 1.0, "eca_fraction": 0.05},
        ],
    }
    vessel = {
        "id": 1,
        "name": "Feeder-Test",
        "vessel_class": "Feeder",
        "capacity_teu": 1500,
        "design_speed_kn": 18.0,
    }
    fuel = {
        "id": 1,
        "name": "HFO",
        "price_usd_per_tonne": 580.0,
        "lhv_mj_per_kg": 40.5,
        "co2_wtw_g_per_mj": 91.0,
    }
    return FleetOptimizationProblemV2(
        routes=[route],
        vessels=[vessel],
        fuels=[fuel],
        vessel_limits={"Feeder-Test": 1},
    )


def test_problem_v2_route_options_and_shore_power():
    problem = _minimal_problem()
    assert problem.num_routes == 1
    assert len(problem._lookup_cache) == 2 * 1 * 5 * 1
    direct = problem._lookup_cache[(1, 0, 1, 14.4, 1)]
    eca = problem._lookup_cache[(1, 1, 1, 14.4, 1)]
    assert direct["used_shore_power"] is True
    assert direct["port_cost"] > 0
    assert eca["total_co2_t"] > 0


def test_plan_fleet_page_entrypoint_and_csv_validation():
    assert callable(page6_plan_fleet.render)
    valid = "name,origin_port,dest_port,distance_nm,cargo_demand_teu,deadline_days\nR1,A,B,10,100,2\n"
    frame, error = page6_plan_fleet.validate_fleet_csv(io.StringIO(valid))
    assert error is None
    assert len(frame) == 1

    invalid = "name,origin_port,dest_port,distance_nm\nR1,A,B,10\n"
    frame, error = page6_plan_fleet.validate_fleet_csv(io.StringIO(invalid))
    assert frame is None
    assert "Missing required columns" in error


def test_api_summary_uses_temporary_database(tmp_path, monkeypatch):
    db_path = tmp_path / "api-test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    database.Base.metadata.create_all(engine)
    engine.dispose()
    monkeypatch.setattr(api, "_DB_PATH", db_path)

    summary = api._build_summary()
    assert summary["optimization_run_count"] == 0


def test_phase5_todo_registry_and_read_only_results_api(tmp_path, monkeypatch):
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
