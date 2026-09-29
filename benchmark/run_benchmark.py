"""
QFleet Phase 3 — Benchmark Execution Harness
============================================
Runs reproducible benchmark experiments across Small, Medium, and Large instances:
- Deterministic baselines: MILP (SciPy HiGHS, 120s limit) and Greedy.
- Stochastic metaheuristics: GA, SQA, and QI-EA across 10 distinct seeds.
- Precomputes surrogate table once per instance and reports build time separately.
- Guarantees exact equal evaluation budgets across GA, SQA, and QI-EA.
- Re-scores every returned solution with problem.evaluate() and verifies consistency.
- Persists results to SQLite optimization_runs table.
"""

from __future__ import annotations

import sys
import time
import json
import pathlib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
import db as database
from optimizer.problem import FleetOptimizationProblem
from optimizer.baselines import milp_optimize, greedy_optimize, ga_optimize
from optimizer.qi_optimizer import qiea_optimize, sqa_optimize
from benchmark.instances import generate_benchmark_scenarios


def run_full_benchmark(
    seeds: list[int] = config.BENCHMARK_SEEDS,
    save_to_db: bool = True,
) -> dict:
    """
    Executes the benchmark across Small, Medium, and Large instances.
    Returns structured results dictionary containing raw runs, convergence histories,
    and instance metadata.
    """
    print("=" * 80)
    print("QFLEET PHASE 3: COMPREHENSIVE BENCHMARK HARNESS EXECUTION")
    print(f"Random Seeds (N={len(seeds)}): {seeds}")
    print("=" * 80)

    engine = database.get_engine()
    database.init_db()

    with Session(engine) as session:
        scenario_ids = generate_benchmark_scenarios(session)

    benchmark_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seeds": seeds,
            "budgets": config.BENCHMARK_EVAL_BUDGETS,
        },
        "instances": {},
        "raw_runs": [],
        "convergence_histories": {},
    }

    for inst_name in ["Small", "Medium", "Large"]:
        sc_id = scenario_ids[inst_name]
        params = config.BENCHMARK_INSTANCE_PARAMS[inst_name]
        budget = params["eval_budget"]

        print(f"\n[{inst_name} Instance] Loading Scenario ID: {sc_id} ({params['num_vessels']} vessels, {params['num_routes']} routes)...")
        prob = FleetOptimizationProblem(scenario_id=sc_id)
        surrogate_build_time = prob.surrogate_build_time_s
        print(f"[{inst_name} Instance] Cached surrogate grid size: {len(prob._lookup_cache)} entries. Build time: {surrogate_build_time:.4f} s.")

        instance_info = {
            "name": inst_name,
            "scenario_id": sc_id,
            "num_vessels": prob.num_vessels,
            "num_routes": prob.num_routes,
            "emissions_cap_t": prob.emissions_cap_t,
            "eval_budget": budget,
            "surrogate_build_time_s": surrogate_build_time,
            "exact_ground_truth": None,
        }

        # Compute exact ground truth for Small instance
        if inst_name == "Small":
            print(f"[{inst_name} Instance] Computing exact brute-force ground truth (4^6 = 4,096 vessel combinations)...")
            gt_plan, gt_obj, gt_feas, gt_details = prob.exact_brute_force_optimal()
            instance_info["exact_ground_truth"] = {
                "objective": float(gt_obj),
                "is_feasible": bool(gt_feas),
                "fuel_cost": float(gt_details["fuel_cost_usd"]),
                "co2_t": float(gt_details["co2_wtw_t"]),
            }
            print(f"[{inst_name} Instance] Exact Ground Truth J*: ${gt_obj:,.2f} (Feasible: {gt_feas})")

        # -------------------------------------------------------------------
        # 1. Deterministic Baselines: MILP & Greedy
        # -------------------------------------------------------------------
        print(f"[{inst_name} Instance] Running Deterministic Baselines...")

        # A. MILP (SciPy HiGHS)
        prob.eval_count = 0
        t0 = time.perf_counter()
        milp_sol, milp_self_obj, milp_hist = milp_optimize(prob, seed=42, time_limit_s=120)
        milp_runtime = time.perf_counter() - t0
        milp_rescored_obj, milp_feas, milp_details = prob.evaluate(milp_sol)

        _verify_rescore(milp_self_obj, milp_rescored_obj, "MILP_HiGHS", inst_name)
        if save_to_db:
            prob.save_run("MILP_HiGHS", 42, milp_sol, milp_rescored_obj, milp_runtime, milp_hist)

        benchmark_data["raw_runs"].append({
            "instance": inst_name,
            "algorithm": "MILP_HiGHS",
            "seed": 42,
            "self_reported_obj": float(milp_self_obj),
            "objective": float(milp_rescored_obj),
            "fuel_cost_usd": float(milp_details["fuel_cost_usd"]),
            "co2_t": float(milp_details["co2_wtw_t"]),
            "delay_days": float(milp_details["total_delay_days"]),
            "is_feasible": bool(milp_feas),
            "evaluations": 1,
            "runtime_s": float(milp_runtime),
            "surrogate_build_time_s": float(surrogate_build_time),
        })

        # B. Greedy
        prob.eval_count = 0
        t0 = time.perf_counter()
        greedy_sol, greedy_self_obj, greedy_hist = greedy_optimize(prob, seed=42)
        greedy_runtime = time.perf_counter() - t0
        greedy_rescored_obj, greedy_feas, greedy_details = prob.evaluate(greedy_sol)

        _verify_rescore(greedy_self_obj, greedy_rescored_obj, "Greedy", inst_name)
        if save_to_db:
            prob.save_run("Greedy", 42, greedy_sol, greedy_rescored_obj, greedy_runtime, greedy_hist)

        benchmark_data["raw_runs"].append({
            "instance": inst_name,
            "algorithm": "Greedy",
            "seed": 42,
            "self_reported_obj": float(greedy_self_obj),
            "objective": float(greedy_rescored_obj),
            "fuel_cost_usd": float(greedy_details["fuel_cost_usd"]),
            "co2_t": float(greedy_details["co2_wtw_t"]),
            "delay_days": float(greedy_details["total_delay_days"]),
            "is_feasible": bool(greedy_feas),
            "evaluations": 1,
            "runtime_s": float(greedy_runtime),
            "surrogate_build_time_s": float(surrogate_build_time),
        })

        # -------------------------------------------------------------------
        # 2. Stochastic Metaheuristics (GA, SQA, QI-EA) across 10 Seeds
        # -------------------------------------------------------------------
        print(f"[{inst_name} Instance] Running GA, SQA, and QI-EA across {len(seeds)} seeds (Budget={budget} evals)...")

        for seed in seeds:
            # GA
            prob.eval_count = 0
            t0 = time.perf_counter()
            ga_sol, ga_self_obj, ga_hist = ga_optimize(
                prob,
                seed=seed,
                pop_size=params["ga_params"]["pop_size"],
                n_generations=params["ga_params"]["n_generations"],
            )
            ga_runtime = time.perf_counter() - t0
            ga_evals = prob.eval_count
            ga_rescored_obj, ga_feas, ga_details = prob.evaluate(ga_sol)
            _verify_rescore(ga_self_obj, ga_rescored_obj, "GA", inst_name, seed)

            # SQA
            prob.eval_count = 0
            t0 = time.perf_counter()
            sqa_sol, sqa_self_obj, sqa_hist = sqa_optimize(
                prob,
                seed=seed,
                n_trotters=params["sqa_params"]["n_trotters"],
                n_steps=params["sqa_params"]["n_steps"],
            )
            sqa_runtime = time.perf_counter() - t0
            sqa_evals = prob.eval_count
            sqa_rescored_obj, sqa_feas, sqa_details = prob.evaluate(sqa_sol)
            _verify_rescore(sqa_self_obj, sqa_rescored_obj, "SQA", inst_name, seed)

            # QI-EA
            prob.eval_count = 0
            t0 = time.perf_counter()
            qiea_sol, qiea_self_obj, qiea_hist = qiea_optimize(
                prob,
                seed=seed,
                pop_size=params["qiea_params"]["pop_size"],
                n_generations=params["qiea_params"]["n_generations"],
            )
            qiea_runtime = time.perf_counter() - t0
            qiea_evals = prob.eval_count
            qiea_rescored_obj, qiea_feas, qiea_details = prob.evaluate(qiea_sol)
            _verify_rescore(qiea_self_obj, qiea_rescored_obj, "QI-EA", inst_name, seed)

            # Assert evaluation budget parity across methods
            assert ga_evals == sqa_evals == qiea_evals == budget, (
                f"Evaluation count mismatch on {inst_name} (seed {seed}): "
                f"GA={ga_evals}, SQA={sqa_evals}, QI-EA={qiea_evals} vs expected budget={budget}"
            )

            # Store convergence history for medium instance reference run (seed 42)
            if inst_name == "Medium" and seed == 42:
                benchmark_data["convergence_histories"]["GA"] = [float(x) for x in ga_hist]
                benchmark_data["convergence_histories"]["SQA"] = [float(x) for x in sqa_hist]
                benchmark_data["convergence_histories"]["QI-EA"] = [float(x) for x in qiea_hist]

            if save_to_db:
                prob.save_run("GA", seed, ga_sol, ga_rescored_obj, ga_runtime, ga_hist)
                prob.save_run("SimulatedQuantumAnnealing", seed, sqa_sol, sqa_rescored_obj, sqa_runtime, sqa_hist)
                prob.save_run("QI-EA", seed, qiea_sol, qiea_rescored_obj, qiea_runtime, qiea_hist)

            for algo_name, s_obj, r_obj, feas, det, r_time, ev in [
                ("GA", ga_self_obj, ga_rescored_obj, ga_feas, ga_details, ga_runtime, ga_evals),
                ("SQA", sqa_self_obj, sqa_rescored_obj, sqa_feas, sqa_details, sqa_runtime, sqa_evals),
                ("QI-EA", qiea_self_obj, qiea_rescored_obj, qiea_feas, qiea_details, qiea_runtime, qiea_evals),
            ]:
                benchmark_data["raw_runs"].append({
                    "instance": inst_name,
                    "algorithm": algo_name,
                    "seed": seed,
                    "self_reported_obj": float(s_obj),
                    "objective": float(r_obj),
                    "fuel_cost_usd": float(det["fuel_cost_usd"]),
                    "co2_t": float(det["co2_wtw_t"]),
                    "delay_days": float(det["total_delay_days"]),
                    "is_feasible": bool(feas),
                    "evaluations": int(ev),
                    "runtime_s": float(r_time),
                    "surrogate_build_time_s": float(surrogate_build_time),
                })

        benchmark_data["instances"][inst_name] = instance_info

    print("\nBenchmark execution complete across all instances and seeds.")
    return benchmark_data


def _verify_rescore(
    self_reported: float,
    rescored: float,
    algo: str,
    instance: str,
    seed: int | None = None,
) -> None:
    """Flag and assert that self-reported and re-scored objectives match within 1e-6."""
    diff = abs(self_reported - rescored)
    if diff > 1e-6:
        raise ValueError(
            f"CRITICAL RESCORE MISMATCH: {algo} on {instance} (seed={seed}) "
            f"reported {self_reported:.6f} but rescored {rescored:.6f} (diff={diff:.6e})"
        )


if __name__ == "__main__":
    data = run_full_benchmark()
