"""
QFleet Phase 5 — Optimizers for Version 2 Problem Formulation
=============================================================
Provides MILP, Greedy, GA, SQA, and QI-EA for FleetOptimizationProblemV2.
Isolated in a separate module without touching Phase 2/3 baselines.
"""

from __future__ import annotations

import sys
import time
import pathlib
import numpy as np
from scipy.optimize import milp, LinearConstraint
from scipy.sparse import csc_matrix
from typing import Any

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from optimizer.problem_v2 import FleetOptimizationProblemV2


def milp_optimize_v2(problem: FleetOptimizationProblemV2, seed: int = 42) -> tuple[list[dict], float, dict[str, Any]]:
    """
    Exact MILP formulation over (route, option, vessel, speed, fuel) decision tensor.
    Returns optimal dispatch plan, objective value, and status details.
    """
    start_time = time.perf_counter()
    candidates = []

    for r_idx, route in enumerate(problem.routes):
        route_options = route.get("route_options", [{"option_id": 0}])
        for opt in route_options:
            opt_id = opt["option_id"]
            for v_idx, vessel in enumerate(problem.vessels):
                v_cap = vessel.get("capacity_teu", vessel.get("teu_capacity", 0))
                if v_cap < route["cargo_demand_teu"]:
                    continue
                des_spd = float(vessel["design_speed_kn"])
                for s_factor in problem.speed_bins:
                    spd = round(float(s_factor * des_spd), 4)
                    for f_idx, fuel in enumerate(problem.fuels):
                        key = (route["id"], opt_id, vessel["id"], spd, fuel["id"])
                        entry = problem._lookup_cache.get(key)
                        if entry:
                            if entry["delay_days"] > 1e-4:
                                continue
                            candidates.append({
                                "r_idx": r_idx,
                                "route_id": route["id"],
                                "route_name": route["name"],
                                "opt_id": opt_id,
                                "opt_name": opt.get("name", f"Option {opt_id}"),
                                "v_idx": v_idx,
                                "vessel_id": vessel["id"],
                                "vessel_name": vessel["name"],
                                "vessel_class": vessel["vessel_class"],
                                "spd": spd,
                                "f_idx": f_idx,
                                "fuel_id": fuel["id"],
                                "fuel_type": fuel["name"],
                                "cost": entry["total_cost"],
                                "co2": entry["total_co2_t"],
                                "delay": entry["delay_days"],
                                "used_shore": entry["used_shore_power"],
                            })

    num_vars = len(candidates)
    if num_vars == 0:
        return [], float("inf"), {"status": "error", "message": "No candidates"}

    w1 = problem.weights.get("w1_fuel_cost", 1.0)
    w2 = problem.weights.get("w2_co2_emission", 100.0)
    w3 = problem.weights.get("w3_schedule_penalty", 100000.0)

    c = np.zeros(num_vars)
    for i, cand in enumerate(candidates):
        c[i] = (w1 * cand["cost"]) + (w2 * cand["co2"]) + (w3 * cand["delay"])

    # 1. Exactly one assignment per route: sum_{i in route r} x_i = 1
    A_eq = np.zeros((problem.num_routes, num_vars))
    for i, cand in enumerate(candidates):
        A_eq[cand["r_idx"], i] = 1.0
    b_eq_l = np.ones(problem.num_routes)
    b_eq_u = np.ones(problem.num_routes)

    # 2. Vessel availability limits: sum_{i for vessel v} x_i <= quota_v
    A_vessel = np.zeros((problem.num_vessels, num_vars))
    b_vessel_u = np.zeros(problem.num_vessels)
    for v_idx, v in enumerate(problem.vessels):
        max_q = problem.vessel_limits.get(v["vessel_class"], problem.vessel_limits.get(v["name"], 2))
        b_vessel_u[v_idx] = max_q

    for i, cand in enumerate(candidates):
        A_vessel[cand["v_idx"], i] = 1.0
    b_vessel_l = np.zeros(problem.num_vessels)

    # 3. Emissions cap: sum_{i} co2_i * x_i <= emissions_cap_t
    if problem.emissions_cap_t is not None:
        A_co2 = np.zeros((1, num_vars))
        for i, cand in enumerate(candidates):
            A_co2[0, i] = cand["co2"]
        b_co2_l = np.array([0.0])
        b_co2_u = np.array([problem.emissions_cap_t])

        A_all = np.vstack([A_eq, A_vessel, A_co2])
        lhs = np.concatenate([b_eq_l, b_vessel_l, b_co2_l])
        rhs = np.concatenate([b_eq_u, b_vessel_u, b_co2_u])
    else:
        A_all = np.vstack([A_eq, A_vessel])
        lhs = np.concatenate([b_eq_l, b_vessel_l])
        rhs = np.concatenate([b_eq_u, b_vessel_u])

    constraints = LinearConstraint(csc_matrix(A_all), lhs, rhs)
    integrality = np.ones(num_vars)

    res = milp(
        c=c,
        integrality=integrality,
        constraints=constraints,
        options={"time_limit": 30.0, "mip_rel_gap": 0.01},
    )
    runtime_s = time.perf_counter() - start_time

    if not res.success and res.x is None:
        return [], float("inf"), {"status": "infeasible", "runtime_s": runtime_s, "is_feasible": False}

    chosen_indices = np.where(res.x > 0.5)[0]
    solution = []
    for idx in chosen_indices:
        cand = candidates[idx]
        solution.append({
            "route_id": cand["route_id"],
            "route_name": cand["route_name"],
            "route_option_id": cand["opt_id"],
            "route_option_name": cand["opt_name"],
            "vessel_id": cand["vessel_id"],
            "vessel_name": cand["vessel_name"],
            "vessel_class": cand["vessel_class"],
            "speed_kn": cand["spd"],
            "fuel_id": cand["fuel_id"],
            "fuel_type": cand["fuel_type"],
            "used_shore_power": cand["used_shore"],
        })

    # Sort solution by route order
    solution.sort(key=lambda s: next(i for i, r in enumerate(problem.routes) if r["id"] == s["route_id"]))
    final_obj, is_feas, details = problem.evaluate(solution)

    return solution, final_obj, {
        "status": "optimal" if is_feas else "infeasible",
        "runtime_s": runtime_s,
        "is_feasible": is_feas,
        "co2_t": details["co2_wtw_t"],
        "cost_usd": details["cost_usd"],
    }
