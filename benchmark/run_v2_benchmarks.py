"""
QFleet Phase 5 — Version 2 Benchmark & Cap Sweep Runner
======================================================
Executes:
1. GA, SQA, QI-EA, Greedy (pure & +decarb), and MILP on Small_v2, Medium_v2, Large_v2, Mega_30V_100R
2. "with shared cap-repair" variants for GA, SQA, and QI-EA
3. Cap sweep on Large_v2 and Mega_30V_100R from unconstrained down to -40%
4. Saves results to results/phase5_v2_benchmark.csv and results/phase5_v2_cap_sweep.csv
5. All operations isolated with zero writes to data/qfleet.db
"""

from __future__ import annotations

import sys
import time
import math
import pathlib
import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from benchmark.instances_v2 import build_v2_instance
from optimizer.problem_v2 import FleetOptimizationProblemV2
from optimizer.baselines_v2 import milp_optimize_v2
from optimizer.algorithms_v2 import (
    greedy_optimize_v2,
    ga_optimize_v2,
    sqa_optimize_v2,
    qiea_optimize_v2,
)


def run_benchmark_experiments():
    instances = ["Small_v2", "Medium_v2", "Large_v2", "Mega_30V_100R"]
    seeds = [42, 123, 456, 789, 999]  # 5 standard seeds

    # Budget per instance
    budgets = {
        "Small_v2": {"pop": 20, "gen": 60, "trotter": 10, "sqa_steps": 60},
        "Medium_v2": {"pop": 30, "gen": 80, "trotter": 10, "sqa_steps": 80},
        "Large_v2": {"pop": 30, "gen": 100, "trotter": 10, "sqa_steps": 100},
        "Mega_30V_100R": {"pop": 30, "gen": 100, "trotter": 10, "sqa_steps": 100},
    }

    results = []
    out_path = _ROOT / "results" / "phase5_v2_benchmark.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for inst_name in instances:
        print(f"\n================ Running Benchmark on {inst_name} ================", flush=True)
        prob = build_v2_instance(inst_name)
        b_cfg = budgets[inst_name]

        # 1. Exact Reference: MILP
        print("Solving MILP...", flush=True)
        sol_m, obj_m, det_m = milp_optimize_v2(prob)
        ref_obj = obj_m
        results.append({
            "instance": inst_name,
            "algorithm": "MILP (Exact)",
            "reference_type": "Exact Lower Bound",
            "reference_obj": ref_obj,
            "n_runs": 1,
            "n_feasible": 1 if det_m.get("is_feasible", False) else 0,
            "feasibility_rate_pct": 100.0 if det_m.get("is_feasible", False) else 0.0,
            "mean_objective": ref_obj if det_m.get("is_feasible", False) else None,
            "std_objective": 0.0 if det_m.get("is_feasible", False) else None,
            "best_objective": ref_obj if det_m.get("is_feasible", False) else None,
            "worst_objective": ref_obj if det_m.get("is_feasible", False) else None,
            "gap_to_ref_pct": 0.0 if det_m.get("is_feasible", False) else None,
            "mean_runtime_s": det_m.get("runtime_s", 0.0),
            "actual_eval_count": 1,
            "is_failed": not det_m.get("is_feasible", False),
            "display_status": "OK" if det_m.get("is_feasible", False) else "Failed: no feasible plan",
        })

        # 2. Greedy (pure)
        print("Solving Greedy (pure)...", flush=True)
        sol_gp, obj_gp, det_gp = greedy_optimize_v2(prob, decarb_pass=False)
        is_feas_gp = det_gp.get("is_feasible", False)
        gap_gp = ((obj_gp - ref_obj) / ref_obj * 100.0) if (is_feas_gp and ref_obj < float("inf")) else None
        results.append({
            "instance": inst_name,
            "algorithm": "Greedy (pure)",
            "reference_type": "MILP (Exact)",
            "reference_obj": ref_obj,
            "n_runs": 1,
            "n_feasible": 1 if is_feas_gp else 0,
            "feasibility_rate_pct": 100.0 if is_feas_gp else 0.0,
            "mean_objective": obj_gp if is_feas_gp else None,
            "std_objective": 0.0 if is_feas_gp else None,
            "best_objective": obj_gp if is_feas_gp else None,
            "worst_objective": obj_gp if is_feas_gp else None,
            "gap_to_ref_pct": gap_gp,
            "mean_runtime_s": det_gp.get("runtime_s", 0.0),
            "actual_eval_count": det_gp.get("eval_count", 0),
            "is_failed": not is_feas_gp,
            "display_status": "OK" if is_feas_gp else "Failed: no feasible plan",
        })

        # 3. Greedy (+decarb pass)
        print("Solving Greedy (+decarb)...", flush=True)
        sol_gd, obj_gd, det_gd = greedy_optimize_v2(prob, decarb_pass=True)
        is_feas_gd = det_gd.get("is_feasible", False)
        gap_gd = ((obj_gd - ref_obj) / ref_obj * 100.0) if (is_feas_gd and ref_obj < float("inf")) else None
        results.append({
            "instance": inst_name,
            "algorithm": "Greedy (+decarb pass)",
            "reference_type": "MILP (Exact)",
            "reference_obj": ref_obj,
            "n_runs": 1,
            "n_feasible": 1 if is_feas_gd else 0,
            "feasibility_rate_pct": 100.0 if is_feas_gd else 0.0,
            "mean_objective": obj_gd if is_feas_gd else None,
            "std_objective": 0.0 if is_feas_gd else None,
            "best_objective": obj_gd if is_feas_gd else None,
            "worst_objective": obj_gd if is_feas_gd else None,
            "gap_to_ref_pct": gap_gd,
            "mean_runtime_s": det_gd.get("runtime_s", 0.0),
            "actual_eval_count": det_gd.get("eval_count", 0),
            "is_failed": not is_feas_gd,
            "display_status": "OK" if is_feas_gd else "Failed: no feasible plan",
        })

        # Heuristic methods across seeds
        methods = [
            ("GeneticAlgorithm (GA)", lambda s: ga_optimize_v2(prob, pop_size=b_cfg["pop"], n_generations=b_cfg["gen"], seed=s, with_cap_repair=False)),
            ("SimulatedQuantumAnnealing (SQA)", lambda s: sqa_optimize_v2(prob, n_trotter=b_cfg["trotter"], n_steps=b_cfg["sqa_steps"], seed=s, with_cap_repair=False)),
            ("QI-EA", lambda s: qiea_optimize_v2(prob, pop_size=b_cfg["pop"], n_generations=b_cfg["gen"], seed=s, with_cap_repair=False)),
            ("GeneticAlgorithm (with cap-repair)", lambda s: ga_optimize_v2(prob, pop_size=b_cfg["pop"], n_generations=b_cfg["gen"], seed=s, with_cap_repair=True)),
            ("SimulatedQuantumAnnealing (with cap-repair)", lambda s: sqa_optimize_v2(prob, n_trotter=b_cfg["trotter"], n_steps=b_cfg["sqa_steps"], seed=s, with_cap_repair=True)),
            ("QI-EA (with cap-repair)", lambda s: qiea_optimize_v2(prob, pop_size=b_cfg["pop"], n_generations=b_cfg["gen"], seed=s, with_cap_repair=True)),
        ]

        for alg_name, runner in methods:
            print(f"Running {alg_name}...", flush=True)
            runs = []
            for s in seeds:
                sol, obj, det = runner(s)
                runs.append({
                    "seed": s,
                    "objective": obj,
                    "is_feasible": det.get("is_feasible", False),
                    "runtime_s": det.get("runtime_s", 0.0),
                    "eval_count": det.get("eval_count", 0),
                })

            n_total = len(runs)
            feas_runs = [r for r in runs if r["is_feasible"]]
            n_feas = len(feas_runs)
            feas_rate = (n_feas / n_total) * 100.0
            mean_evals = int(np.mean([r["eval_count"] for r in runs]))
            mean_rt = float(np.mean([r["runtime_s"] for r in runs]))

            if n_feas > 0:
                feas_objs = [r["objective"] for r in feas_runs]
                mean_obj = float(np.mean(feas_objs))
                std_obj = float(np.std(feas_objs)) if n_feas > 1 else 0.0
                best_obj = float(np.min(feas_objs))
                worst_obj = float(np.max(feas_objs))
                gap = ((mean_obj - ref_obj) / ref_obj * 100.0) if ref_obj < float("inf") else None
                is_failed = False
                disp = "OK"
            else:
                mean_obj = None
                std_obj = None
                best_obj = None
                worst_obj = None
                gap = None
                is_failed = True
                disp = "Failed: no feasible plan"

            results.append({
                "instance": inst_name,
                "algorithm": alg_name,
                "reference_type": "MILP (Exact)",
                "reference_obj": ref_obj,
                "n_runs": n_total,
                "n_feasible": n_feas,
                "feasibility_rate_pct": feas_rate,
                "mean_objective": mean_obj,
                "std_objective": std_obj,
                "best_objective": best_obj,
                "worst_objective": worst_obj,
                "gap_to_ref_pct": gap,
                "mean_runtime_s": mean_rt,
                "actual_eval_count": mean_evals,
                "is_failed": is_failed,
                "display_status": disp,
            })

        # Save intermediate
        pd.DataFrame(results).to_csv(out_path, index=False)

    df_results = pd.DataFrame(results)
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved v2 benchmark table to {out_path}", flush=True)
    return df_results


def run_cap_sweep():
    """Cap sweep on Large_v2 and Mega_30V_100R from 100% (unconstrained) down to 60% (-40%)."""
    sweep_fractions = [1.00, 0.90, 0.80, 0.70, 0.60]  # 0% to -40%
    instances = ["Large_v2", "Mega_30V_100R"]
    seed = 42

    sweep_results = []
    out_sweep_path = _ROOT / "results" / "phase5_v2_cap_sweep.csv"

    for inst_name in instances:
        print(f"\n========== Cap Sweep on {inst_name} ==========", flush=True)
        base_prob = build_v2_instance(inst_name)
        
        # Get unconstrained CO2
        unconstrained_prob = FleetOptimizationProblemV2(
            routes=base_prob.routes,
            vessels=base_prob.vessels,
            fuels=base_prob.fuels,
            weights=base_prob.weights,
            emissions_cap_t=None,
            vessel_limits=base_prob.vessel_limits,
            instance_name=inst_name,
        )
        _, _, det_u = milp_optimize_v2(unconstrained_prob)
        unconstrained_co2 = det_u.get("co2_t", 50000.0)

        for frac in sweep_fractions:
            cap_val = round(unconstrained_co2 * frac, 1)
            reduction_pct = round((1.0 - frac) * 100.0, 1)
            print(f"Testing cap reduction: -{reduction_pct}% (Cap = {cap_val:,.1f} t CO2)", flush=True)

            prob = FleetOptimizationProblemV2(
                routes=base_prob.routes,
                vessels=base_prob.vessels,
                fuels=base_prob.fuels,
                weights=base_prob.weights,
                emissions_cap_t=cap_val,
                vessel_limits=base_prob.vessel_limits,
                instance_name=inst_name,
            )

            methods = [
                ("MILP", lambda: milp_optimize_v2(prob, seed=seed)),
                ("Greedy (pure)", lambda: greedy_optimize_v2(prob, decarb_pass=False, seed=seed)),
                ("Greedy (+decarb)", lambda: greedy_optimize_v2(prob, decarb_pass=True, seed=seed)),
                ("GA", lambda: ga_optimize_v2(prob, pop_size=20, n_generations=30, seed=seed, with_cap_repair=False)),
                ("SQA", lambda: sqa_optimize_v2(prob, n_trotter=10, n_steps=30, seed=seed, with_cap_repair=False)),
                ("QI-EA", lambda: qiea_optimize_v2(prob, pop_size=20, n_generations=30, seed=seed, with_cap_repair=False)),
                ("GA (+cap-repair)", lambda: ga_optimize_v2(prob, pop_size=20, n_generations=30, seed=seed, with_cap_repair=True)),
                ("SQA (+cap-repair)", lambda: sqa_optimize_v2(prob, n_trotter=10, n_steps=30, seed=seed, with_cap_repair=True)),
                ("QI-EA (+cap-repair)", lambda: qiea_optimize_v2(prob, pop_size=20, n_generations=30, seed=seed, with_cap_repair=True)),
            ]

            for m_name, fn in methods:
                sol, obj, det = fn()
                is_feas = det.get("is_feasible", False)
                sweep_results.append({
                    "instance": inst_name,
                    "reduction_pct": reduction_pct,
                    "emissions_cap_t": cap_val,
                    "algorithm": m_name,
                    "is_feasible": is_feas,
                    "objective": obj if is_feas else None,
                    "co2_t": det.get("co2_t"),
                    "cost_usd": det.get("cost_usd"),
                    "runtime_s": det.get("runtime_s"),
                    "display_status": "OK" if is_feas else "Failed: no feasible plan",
                })

            pd.DataFrame(sweep_results).to_csv(out_sweep_path, index=False)

    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv(out_sweep_path, index=False)
    print(f"Saved cap sweep to {out_sweep_path}", flush=True)
    return df_sweep


if __name__ == "__main__":
    print("Starting Phase 5 V2 Benchmarks...", flush=True)
    t0 = time.perf_counter()
    run_benchmark_experiments()
    run_cap_sweep()
    print(f"Completed all experiments in {time.perf_counter() - t0:.2f}s", flush=True)
