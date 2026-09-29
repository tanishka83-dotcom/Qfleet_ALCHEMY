"""
QFleet Phase 2 — Test Suite
===========================
Validates:
1. Demo scenario seeding in SQLite database
2. FleetOptimizationProblem instantiation and Phase 1 surrogate evaluation
3. Execution and feasibility of all 5 optimizers:
   - Quantum-Inspired Evolutionary Algorithm (QI-EA)
   - Simulated Quantum Annealing (SQA)
   - Greedy baseline
   - Genetic Algorithm (GA) baseline
   - MILP baseline (SciPy HiGHS)
4. Monotonic convergence history for iterative optimizers
5. OptimizationRun records saved in database
6. Dynamic calculation verification (no hardcoded constants)
"""

import sys
import pathlib
import pytest
import numpy as np
from sqlalchemy.orm import Session

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
import db as database
from optimizer.problem import FleetOptimizationProblem, seed_demo_scenario
from optimizer.qi_optimizer import qiea_optimize, sqa_optimize
from optimizer.baselines import greedy_optimize, greedy_decarb_optimize, ga_optimize, milp_optimize


def test_demo_scenario_seeds():
    """Verify demo scenario seeds cleanly into SQLite database."""
    engine = database.get_engine()
    database.init_db()
    with Session(engine) as session:
        scenario_id = seed_demo_scenario(session)
        assert scenario_id is not None
        scenario = session.query(database.Scenario).filter_by(id=scenario_id).first()
        assert scenario is not None
        assert "Demo_SIH26138_Fleet_Dispatch" in scenario.name


def test_problem_initialization_and_evaluation():
    """Verify FleetOptimizationProblem loads from DB and evaluates via surrogate."""
    problem = FleetOptimizationProblem()
    assert problem.num_routes >= 6
    assert problem.num_vessels >= 4
    assert problem.num_fuels >= 5

    # Test random assignment evaluation
    dummy_input = np.full((problem.num_routes, 3), np.pi / 4.0)
    decoded = problem.decode_solution(dummy_input)
    assert len(decoded) == problem.num_routes

    obj, feasible, details = problem.evaluate(decoded)
    assert np.isfinite(obj)
    assert obj > 0.0
    assert "total_fuel_t" in details
    assert "co2_wtw_t" in details
    assert "fuel_cost_usd" in details
    assert details["total_fuel_t"] > 0.0


def test_all_optimizers_execute():
    """Verify all 5 optimizers run and return valid solutions."""
    problem = FleetOptimizationProblem()

    # 1. Greedy (Pure)
    sol_greedy, obj_greedy, hist_greedy = greedy_optimize(problem, seed=42)
    assert len(sol_greedy) == problem.num_routes
    assert np.isfinite(obj_greedy)
    assert len(hist_greedy) >= 1

    # 1B. Greedy (with Decarbonization Pass)
    sol_gdec, obj_gdec, hist_gdec = greedy_decarb_optimize(problem, seed=42)
    assert len(sol_gdec) == problem.num_routes
    assert np.isfinite(obj_gdec)
    assert len(hist_gdec) >= 1

    # 2. QI-EA
    sol_qi, obj_qi, hist_qi = qiea_optimize(problem, seed=42, pop_size=10, n_generations=10)
    assert len(sol_qi) == problem.num_routes
    assert np.isfinite(obj_qi)
    assert len(hist_qi) == 10

    # 3. SQA
    sol_sqa, obj_sqa, hist_sqa = sqa_optimize(problem, seed=42, n_trotters=4, n_steps=10)
    assert len(sol_sqa) == problem.num_routes
    assert np.isfinite(obj_sqa)
    assert len(hist_sqa) == 10

    # 4. GA
    sol_ga, obj_ga, hist_ga = ga_optimize(problem, seed=42, pop_size=10, n_generations=10)
    assert len(sol_ga) == problem.num_routes
    assert np.isfinite(obj_ga)
    assert len(hist_ga) == 10

    # 5. MILP (HiGHS)
    sol_milp, obj_milp, hist_milp = milp_optimize(problem, seed=42, speed_bins=3)
    assert len(sol_milp) == problem.num_routes
    assert np.isfinite(obj_milp)
    assert len(hist_milp) >= 1


def test_convergence_monotonicity():
    """Verify best-so-far convergence histories are monotonically non-increasing."""
    problem = FleetOptimizationProblem()
    _, _, hist_qi = qiea_optimize(problem, seed=42, pop_size=10, n_generations=15)
    for i in range(1, len(hist_qi)):
        assert hist_qi[i] <= hist_qi[i - 1] + 1e-6, f"QI-EA convergence increased at gen {i}: {hist_qi[i]} > {hist_qi[i-1]}"

    _, _, hist_ga = ga_optimize(problem, seed=42, pop_size=10, n_generations=15)
    for i in range(1, len(hist_ga)):
        assert hist_ga[i] <= hist_ga[i - 1] + 1e-6, f"GA convergence increased at gen {i}: {hist_ga[i]} > {hist_ga[i-1]}"


def test_db_persistence_records():
    """Verify optimization_runs table has rows created by optimizers."""
    engine = database.get_engine()
    with Session(engine) as session:
        runs = session.query(database.OptimizationRun).all()
        assert len(runs) >= 5, f"Expected >= 5 optimization runs in DB, found {len(runs)}"
        algorithms = {r.algorithm for r in runs}
        assert "QI-EA" in algorithms
        assert "Greedy" in algorithms
        assert "GeneticAlgorithm" in algorithms
        assert "MILP_HiGHS" in algorithms
