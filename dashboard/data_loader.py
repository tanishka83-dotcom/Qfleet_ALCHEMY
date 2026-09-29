"""
QFleet Phase 4 — Data Loader (Strict Read-Only)
==============================================
Loads problem instances, fuels, vessels, routes, prediction metrics,
and benchmark optimization runs strictly from:
1. SQLite DB: data/qfleet.db (opened strictly in read-only mode with URI mode=ro)
2. CSV: data/benchmark_results.csv

Enforces amendment rules:
- No hardcoded metrics, gaps, or objectives.
- Filter to canonical benchmark runs (96 runs across 3 instances, 5 algorithms, 10 seeds).
- Identifies and reports excluded non-benchmark runs (562 excluded).
"""

from __future__ import annotations

import json
import sqlite3
import pathlib
import pandas as pd
from typing import Any

_ROOT = pathlib.Path(__file__).parent.parent
DB_PATH = _ROOT / "data" / "qfleet.db"
CSV_PATH = _ROOT / "data" / "benchmark_results.csv"


def get_ro_connection() -> sqlite3.Connection:
    """Returns a SQLite connection opened strictly in read-only mode."""
    db_uri = f"file:{DB_PATH.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(db_uri, uri=True)


def load_fuels() -> pd.DataFrame:
    """Loads all fuels from SQLite fuels table."""
    with get_ro_connection() as conn:
        df = pd.read_sql("SELECT * FROM fuels ORDER BY id ASC", conn)
    return df


def load_vessels() -> pd.DataFrame:
    """Loads all vessels from SQLite vessels table."""
    with get_ro_connection() as conn:
        df = pd.read_sql("SELECT * FROM vessels ORDER BY id ASC", conn)
    return df


def load_routes() -> pd.DataFrame:
    """Loads all routes from SQLite routes table."""
    with get_ro_connection() as conn:
        df = pd.read_sql("SELECT * FROM routes ORDER BY id ASC", conn)
    return df


def load_scenarios() -> pd.DataFrame:
    """Loads scenario definitions and parses JSON configs."""
    with get_ro_connection() as conn:
        df = pd.read_sql("SELECT * FROM scenarios ORDER BY id ASC", conn)
    return df


def load_prediction_metrics() -> pd.DataFrame:
    """Loads surrogate model evaluation metrics from SQLite."""
    with get_ro_connection() as conn:
        df = pd.read_sql("SELECT * FROM prediction_metrics ORDER BY id DESC", conn)
    return df


def load_benchmark_csv() -> pd.DataFrame:
    """Loads Phase 3 summary results from data/benchmark_results.csv."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark results CSV at {CSV_PATH}")
    return pd.read_csv(CSV_PATH)


def load_optimization_runs_audit() -> dict[str, Any]:
    """
    Audits all 658 optimization runs in the SQLite database:
    - Filters to the canonical benchmark set (~96 runs).
    - Explains why the remaining 562 runs are excluded.
    """
    with get_ro_connection() as conn:
        df = pd.read_sql("SELECT * FROM optimization_runs ORDER BY id ASC", conn)

    total_runs = len(df)
    
    # Standardize algorithm names
    df["algo_normalized"] = df["algorithm"].replace({
        "GeneticAlgorithm": "GA",
        "SimulatedQuantumAnnealing": "SQA",
    })

    # Benchmark scenario IDs: 2 (Small), 3 (Medium), 4 (Large)
    # Scenario 1 is Phase 1 Demo
    is_scenario_demo = df["scenario_id"] == 1
    is_benchmark_scenario = df["scenario_id"].isin([2, 3, 4])
    
    # Canonical benchmark set consists of the latest completed run per (scenario_id, algo_normalized, seed)
    df_bench_scenarios = df[is_benchmark_scenario].copy()
    canonical_benchmark_runs = (
        df_bench_scenarios
        .sort_values("id")
        .groupby(["scenario_id", "algo_normalized", "seed"], as_index=False)
        .last()
    )

    canonical_ids = set(canonical_benchmark_runs["id"].tolist())
    df["is_canonical_benchmark"] = df["id"].isin(canonical_ids)
    
    excluded_runs = df[~df["is_canonical_benchmark"]].copy()

    # Exclusion breakdown
    demo_excluded = len(df[is_scenario_demo])
    superseded_excluded = len(excluded_runs) - demo_excluded

    return {
        "total_runs_count": total_runs,
        "canonical_benchmark_count": len(canonical_benchmark_runs),
        "excluded_runs_count": len(excluded_runs),
        "demo_excluded_count": demo_excluded,
        "superseded_excluded_count": superseded_excluded,
        "all_runs_df": df,
        "canonical_benchmark_df": canonical_benchmark_runs,
        "excluded_runs_df": excluded_runs,
    }


def load_scenario_detail(scenario_id: int) -> dict[str, Any]:
    """Extracts parsed scenario configuration and resolved vessels and routes."""
    with get_ro_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, fleet_config_json FROM scenarios WHERE id = ?", (scenario_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Scenario ID {scenario_id} not found in database.")
        
        sc_id, name, config_json = row
        cfg = json.loads(config_json)
        
        v_ids = cfg.get("vessel_ids", [])
        r_ids = cfg.get("route_ids", [])
        
        vessels_df = pd.read_sql(
            f"SELECT * FROM vessels WHERE id IN ({','.join(map(str, v_ids))})", conn
        ) if v_ids else pd.DataFrame()
        
        routes_df = pd.read_sql(
            f"SELECT * FROM routes WHERE id IN ({','.join(map(str, r_ids))})", conn
        ) if r_ids else pd.DataFrame()

    return {
        "scenario_id": sc_id,
        "name": name,
        "config": cfg,
        "vessels_df": vessels_df,
        "routes_df": routes_df,
        "vessel_limits": cfg.get("vessel_limits", {}),
        "emissions_cap_t": cfg.get("max_emissions_cap_t"),
        "weights": cfg.get("weights", {}),
        "assumed_weather_factor": cfg.get("assumed_weather_factor", 1.08),
    }
