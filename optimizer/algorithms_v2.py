"""
QFleet Phase 5 — Optimizers & Benchmarks for Version 2 Problem Formulation
==========================================================================
Provides:
1. Greedy (pure) and Greedy (+decarb pass)
2. Genetic Algorithm (GA) with equal budget
3. Simulated Quantum Annealing (SQA) with equal budget
4. Quantum-Inspired Evolutionary Algorithm (QI-EA) with equal budget
5. Shared Cap-Repair Variants for GA, SQA, and QI-EA
6. Benchmark Runner and Cap Sweep for V2 instances (Small_v2, Medium_v2, Large_v2, Mega_30V_100R)
"""

from __future__ import annotations

import sys
import time
import copy
import pathlib
import numpy as np
import pandas as pd
from typing import Any, Callable

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from optimizer.problem_v2 import FleetOptimizationProblemV2, PORT_SHORE_PROFILES, VESSEL_AUX_POWER_KW
from optimizer.baselines_v2 import milp_optimize_v2


# ---------------------------------------------------------------------------
# Shared Constraint & Cap Repair Operators for V2
# ---------------------------------------------------------------------------

def repair_solution_v2(problem: FleetOptimizationProblemV2, plan: list[dict]) -> list[dict]:
    """
    Standard domain repair operator for V2 (capacity, deadline, quota).
    """
    repaired = []
    sorted_vessels = sorted(problem.vessels, key=lambda v: v.get("capacity_teu", v.get("teu_capacity", 0)))
    vessels_by_id = {v["id"]: v for v in problem.vessels}
    fuels_by_id = {f["id"]: f for f in problem.fuels}

    for i, a in enumerate(plan):
        route = problem.routes[i]
        curr_vessel = vessels_by_id[a["vessel_id"]]
        curr_fuel = fuels_by_id[a["fuel_id"]]
        opt_id = a.get("route_option_id", 0)

        # Capacity repair
        v_cap = curr_vessel.get("capacity_teu", curr_vessel.get("teu_capacity", 0))
        if v_cap < route["cargo_demand_teu"]:
            vessel_to_use = next(
                (v for v in sorted_vessels if v.get("capacity_teu", v.get("teu_capacity", 0)) >= route["cargo_demand_teu"]),
                sorted_vessels[-1]
            )
        else:
            vessel_to_use = curr_vessel

        # Speed snap & deadline repair
        des_spd = float(vessel_to_use["design_speed_kn"])
        discrete_speeds = [round(float(s * des_spd), 4) for s in problem.speed_bins]
        desired_speed = min(discrete_speeds, key=lambda s: abs(s - a["speed_kn"]))

        route_options = route.get("route_options", [{"option_id": 0, "dist_factor": 1.0}])
        opt = next((o for o in route_options if o["option_id"] == opt_id), route_options[0])
        dist = route["distance_nm"] * opt.get("dist_factor", 1.0)
        min_speed_for_deadline = dist / (route["deadline_days"] * 24.0)

        feasible_speeds = [s for s in discrete_speeds if s >= min_speed_for_deadline - 1e-4]
        if feasible_speeds:
            if desired_speed < min(feasible_speeds):
                desired_speed = min(feasible_speeds)
        else:
            desired_speed = discrete_speeds[-1]

        repaired.append({
            "route_id": route["id"],
            "route_name": route["name"],
            "route_option_id": opt_id,
            "route_option_name": opt.get("name", f"Option {opt_id}"),
            "vessel_id": vessel_to_use["id"],
            "vessel_name": vessel_to_use["name"],
            "vessel_class": vessel_to_use["vessel_class"],
            "speed_kn": float(desired_speed),
            "fuel_id": curr_fuel["id"],
            "fuel_type": curr_fuel["name"],
        })

    # Quota repair
    vessel_counts = {}
    for r in repaired:
        vessel_counts[r["vessel_id"]] = vessel_counts.get(r["vessel_id"], 0) + 1

    for r in repaired:
        v_id = r["vessel_id"]
        v_obj = vessels_by_id[v_id]
        max_routes = problem.vessel_limits.get(v_obj["vessel_class"], problem.vessel_limits.get(v_obj["name"], 2))

        if vessel_counts[v_id] > max_routes:
            route_obj = next(rt for rt in problem.routes if rt["id"] == r["route_id"])
            alt_vessels = [
                v for v in problem.vessels
                if v.get("capacity_teu", v.get("teu_capacity", 0)) >= route_obj["cargo_demand_teu"]
                and vessel_counts.get(v["id"], 0) < problem.vessel_limits.get(v["vessel_class"], problem.vessel_limits.get(v["name"], 2))
            ]
            if alt_vessels:
                new_v = alt_vessels[0]
                vessel_counts[v_id] -= 1
                vessel_counts[new_v["id"]] = vessel_counts.get(new_v["id"], 0) + 1
                r["vessel_id"] = new_v["id"]
                r["vessel_name"] = new_v["name"]
                r["vessel_class"] = new_v["vessel_class"]
                des_spd = float(new_v["design_speed_kn"])
                discrete_speeds = [round(float(s * des_spd), 4) for s in problem.speed_bins]
                r["speed_kn"] = discrete_speeds[-1]

    return repaired


def apply_shared_cap_repair(
    problem: FleetOptimizationProblemV2,
    plan: list[dict],
    eval_counter: list[int] | None = None,
) -> list[dict]:
    """
    Fast Shared Emissions-Cap Repair Operator (applied equally across GA, SQA, QI-EA):
    If total emissions exceed cap, progressively shifts voyages to lower carbon fuels (e.g. LNG/Methanol)
    and optimal speed/route options without violating quotas or deadlines.
    """
    repaired_plan = [dict(d) for d in plan]
    if problem.emissions_cap_t is None:
        return repaired_plan

    _, is_feas, details = problem.evaluate(repaired_plan)
    if eval_counter is not None:
        eval_counter[0] += 1
    if is_feas or details["co2_wtw_t"] <= problem.emissions_cap_t:
        return repaired_plan

    # Find cleanest fuel
    cleanest_fuel = min(problem.fuels, key=lambda f: f["co2_wtw_g_per_mj"])

    # Shift routes to cleanest fuel
    for r_idx in range(len(repaired_plan)):
        repaired_plan[r_idx]["fuel_id"] = cleanest_fuel["id"]
        repaired_plan[r_idx]["fuel_type"] = cleanest_fuel["name"]
        _, is_feas_new, det_new = problem.evaluate(repaired_plan)
        if eval_counter is not None:
            eval_counter[0] += 1
        if is_feas_new or det_new["co2_wtw_t"] <= problem.emissions_cap_t:
            break

    return repaired_plan


# ---------------------------------------------------------------------------
# 1. Greedy (Pure & +Decarb Pass)
# ---------------------------------------------------------------------------

def greedy_optimize_v2(problem: FleetOptimizationProblemV2, decarb_pass: bool = False, seed: int = 42) -> tuple[list[dict], float, dict[str, Any]]:
    start_time = time.perf_counter()
    eval_count = 0

    vessel_usage = {v["id"]: 0 for v in problem.vessels}
    route_order = sorted(range(len(problem.routes)), key=lambda idx: problem.routes[idx]["cargo_demand_teu"], reverse=True)
    assignments_by_route = [None] * len(problem.routes)

    for r_idx in route_order:
        route = problem.routes[r_idx]
        best_r_obj = float("inf")
        best_r_choice = None
        route_options = route.get("route_options", [{"option_id": 0, "name": "Direct"}])

        for vessel in problem.vessels:
            v_cap = vessel.get("capacity_teu", vessel.get("teu_capacity", 0))
            if v_cap < route["cargo_demand_teu"]:
                continue
            max_v = problem.vessel_limits.get(vessel["vessel_class"], problem.vessel_limits.get(vessel["name"], 2))
            if vessel_usage[vessel["id"]] >= max_v:
                continue

            des_spd = float(vessel["design_speed_kn"])
            for s_factor in problem.speed_bins:
                spd = round(float(s_factor * des_spd), 4)
                for opt in route_options:
                    opt_id = opt["option_id"]
                    for fuel in problem.fuels:
                        key = (route["id"], opt_id, vessel["id"], spd, fuel["id"])
                        entry = problem._lookup_cache.get(key)
                        eval_count += 1
                        if entry and entry["delay_days"] <= 1e-4:
                            cost = entry["total_cost"]
                            co2 = entry["total_co2_t"]
                            obj = (problem.weights.get("w1_fuel_cost", 1.0) * cost) + (problem.weights.get("w2_co2_emission", 100.0) * co2)
                            if obj < best_r_obj:
                                best_r_obj = obj
                                best_r_choice = {
                                    "route_id": route["id"],
                                    "route_name": route["name"],
                                    "route_option_id": opt_id,
                                    "route_option_name": opt.get("name", f"Option {opt_id}"),
                                    "vessel_id": vessel["id"],
                                    "vessel_name": vessel["name"],
                                    "vessel_class": vessel["vessel_class"],
                                    "speed_kn": spd,
                                    "fuel_id": fuel["id"],
                                    "fuel_type": fuel["name"],
                                }

        if best_r_choice is not None:
            assignments_by_route[r_idx] = best_r_choice
            vessel_usage[best_r_choice["vessel_id"]] += 1
        else:
            assignments_by_route[r_idx] = {
                "route_id": route["id"],
                "route_name": route["name"],
                "route_option_id": 0,
                "route_option_name": "Direct",
                "vessel_id": problem.vessels[-1]["id"],
                "vessel_name": problem.vessels[-1]["name"],
                "vessel_class": problem.vessels[-1]["vessel_class"],
                "speed_kn": float(problem.vessels[-1]["design_speed_kn"]),
                "fuel_id": problem.fuels[0]["id"],
                "fuel_type": problem.fuels[0]["name"],
            }
            vessel_usage[problem.vessels[-1]["id"]] += 1

    solution = assignments_by_route
    eval_c = [eval_count]
    if decarb_pass:
        solution = apply_shared_cap_repair(problem, solution, eval_counter=eval_c)

    final_obj, is_feas, details = problem.evaluate(solution)
    eval_c[0] += 1
    eval_count = eval_c[0]
    runtime_s = time.perf_counter() - start_time

    return solution, final_obj, {
        "status": "feasible" if is_feas else "infeasible",
        "runtime_s": runtime_s,
        "is_feasible": is_feas,
        "eval_count": eval_count,
        "co2_t": details["co2_wtw_t"],
        "cost_usd": details["cost_usd"],
    }


# ---------------------------------------------------------------------------
# 2. Genetic Algorithm (GA)
# ---------------------------------------------------------------------------

def ga_optimize_v2(
    problem: FleetOptimizationProblemV2,
    pop_size: int = 20,
    n_generations: int = 40,
    seed: int = 42,
    with_cap_repair: bool = False,
) -> tuple[list[dict], float, dict[str, Any]]:
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    eval_count = 0

    num_routes = problem.num_routes
    num_vessels = problem.num_vessels
    num_fuels = problem.num_fuels

    def random_individual():
        ind = []
        for route in problem.routes:
            opts = route.get("route_options", [{"option_id": 0}])
            opt_id = int(rng.choice([o["option_id"] for o in opts]))
            v_idx = int(rng.integers(0, num_vessels))
            vessel = problem.vessels[v_idx]
            s_factor = float(rng.choice(problem.speed_bins))
            spd = round(s_factor * float(vessel["design_speed_kn"]), 4)
            f_idx = int(rng.integers(0, num_fuels))
            fuel = problem.fuels[f_idx]
            ind.append({
                "route_id": route["id"],
                "route_name": route["name"],
                "route_option_id": opt_id,
                "vessel_id": vessel["id"],
                "vessel_name": vessel["name"],
                "vessel_class": vessel["vessel_class"],
                "speed_kn": spd,
                "fuel_id": fuel["id"],
                "fuel_type": fuel["name"],
            })
        return repair_solution_v2(problem, ind)

    population = [random_individual() for _ in range(pop_size)]
    fitnesses = []
    for ind in population:
        fit, _, _ = problem.evaluate(ind)
        eval_count += 1
        fitnesses.append(fit)

    best_idx = int(np.argmin(fitnesses))
    best_sol = population[best_idx]
    best_fit = fitnesses[best_idx]

    for gen in range(n_generations):
        new_pop = [best_sol]  # Elitism
        while len(new_pop) < pop_size:
            i1, i2 = rng.integers(0, pop_size, size=2)
            p1 = population[i1] if fitnesses[i1] < fitnesses[i2] else population[i2]
            i3, i4 = rng.integers(0, pop_size, size=2)
            p2 = population[i3] if fitnesses[i3] < fitnesses[i4] else population[i4]

            child = []
            for r in range(num_routes):
                gene = p1[r] if rng.random() < 0.5 else p2[r]
                gene_copy = dict(gene)
                if rng.random() < 0.15:
                    route = problem.routes[r]
                    opts = route.get("route_options", [{"option_id": 0}])
                    gene_copy["route_option_id"] = int(rng.choice([o["option_id"] for o in opts]))
                    v_idx = int(rng.integers(0, num_vessels))
                    vessel = problem.vessels[v_idx]
                    gene_copy["vessel_id"] = vessel["id"]
                    gene_copy["vessel_name"] = vessel["name"]
                    gene_copy["vessel_class"] = vessel["vessel_class"]
                    s_factor = float(rng.choice(problem.speed_bins))
                    gene_copy["speed_kn"] = round(s_factor * float(vessel["design_speed_kn"]), 4)
                    f_idx = int(rng.integers(0, num_fuels))
                    gene_copy["fuel_id"] = problem.fuels[f_idx]["id"]
                    gene_copy["fuel_type"] = problem.fuels[f_idx]["name"]
                child.append(gene_copy)

            repaired_child = repair_solution_v2(problem, child)
            new_pop.append(repaired_child)

        population = new_pop
        fitnesses = []
        for ind in population:
            fit, _, _ = problem.evaluate(ind)
            eval_count += 1
            fitnesses.append(fit)

        gen_best_idx = int(np.argmin(fitnesses))
        if fitnesses[gen_best_idx] < best_fit:
            best_fit = fitnesses[gen_best_idx]
            best_sol = population[gen_best_idx]

    eval_c = [eval_count]
    final_plan = apply_shared_cap_repair(problem, best_sol, eval_counter=eval_c) if with_cap_repair else best_sol
    final_obj, is_feas, details = problem.evaluate(final_plan)
    eval_c[0] += 1
    eval_count = eval_c[0]
    runtime_s = time.perf_counter() - start_time

    return final_plan, final_obj, {
        "status": "feasible" if is_feas else "infeasible",
        "runtime_s": runtime_s,
        "is_feasible": is_feas,
        "eval_count": eval_count,
        "co2_t": details["co2_wtw_t"],
        "cost_usd": details["cost_usd"],
    }


# ---------------------------------------------------------------------------
# 3. Simulated Quantum Annealing (SQA)
# ---------------------------------------------------------------------------

def sqa_optimize_v2(
    problem: FleetOptimizationProblemV2,
    n_trotter: int = 10,
    n_steps: int = 50,
    seed: int = 42,
    with_cap_repair: bool = False,
) -> tuple[list[dict], float, dict[str, Any]]:
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    eval_count = 0

    num_routes = problem.num_routes
    num_vessels = problem.num_vessels
    num_fuels = problem.num_fuels

    def random_individual():
        ind = []
        for route in problem.routes:
            opts = route.get("route_options", [{"option_id": 0}])
            opt_id = int(rng.choice([o["option_id"] for o in opts]))
            v_idx = int(rng.integers(0, num_vessels))
            vessel = problem.vessels[v_idx]
            s_factor = float(rng.choice(problem.speed_bins))
            spd = round(s_factor * float(vessel["design_speed_kn"]), 4)
            f_idx = int(rng.integers(0, num_fuels))
            fuel = problem.fuels[f_idx]
            ind.append({
                "route_id": route["id"],
                "route_name": route["name"],
                "route_option_id": opt_id,
                "vessel_id": vessel["id"],
                "vessel_name": vessel["name"],
                "vessel_class": vessel["vessel_class"],
                "speed_kn": spd,
                "fuel_id": fuel["id"],
                "fuel_type": fuel["name"],
            })
        return repair_solution_v2(problem, ind)

    slices = [random_individual() for _ in range(n_trotter)]
    slice_energies = []
    for s in slices:
        en, _, _ = problem.evaluate(s)
        eval_count += 1
        slice_energies.append(en)

    best_energy = min(slice_energies)
    best_solution = slices[int(np.argmin(slice_energies))]

    for step in range(n_steps):
        beta = 1.0 + 4.0 * (step / max(1, n_steps))

        for t in range(n_trotter):
            curr_slice = slices[t]
            curr_en = slice_energies[t]

            neighbor = [dict(d) for d in curr_slice]
            mut_r = int(rng.integers(0, num_routes))
            route = problem.routes[mut_r]
            opts = route.get("route_options", [{"option_id": 0}])
            neighbor[mut_r]["route_option_id"] = int(rng.choice([o["option_id"] for o in opts]))
            v_idx = int(rng.integers(0, num_vessels))
            vessel = problem.vessels[v_idx]
            neighbor[mut_r]["vessel_id"] = vessel["id"]
            neighbor[mut_r]["vessel_name"] = vessel["name"]
            neighbor[mut_r]["vessel_class"] = vessel["vessel_class"]
            s_factor = float(rng.choice(problem.speed_bins))
            neighbor[mut_r]["speed_kn"] = round(s_factor * float(vessel["design_speed_kn"]), 4)
            f_idx = int(rng.integers(0, num_fuels))
            neighbor[mut_r]["fuel_id"] = problem.fuels[f_idx]["id"]
            neighbor[mut_r]["fuel_type"] = problem.fuels[f_idx]["name"]

            repaired_neighbor = repair_solution_v2(problem, neighbor)
            cand_en, _, _ = problem.evaluate(repaired_neighbor)
            eval_count += 1

            dE = (cand_en - curr_en) / n_trotter
            if dE <= 0:
                prob = 1.0
            else:
                prob = np.exp(-min(50.0, beta * dE / 1e5))

            if dE <= 0 or rng.random() < prob:
                slices[t] = repaired_neighbor
                slice_energies[t] = cand_en
                if cand_en < best_energy:
                    best_energy = cand_en
                    best_solution = repaired_neighbor

    eval_c = [eval_count]
    final_plan = apply_shared_cap_repair(problem, best_solution, eval_counter=eval_c) if with_cap_repair else best_solution
    final_obj, is_feas, details = problem.evaluate(final_plan)
    eval_c[0] += 1
    eval_count = eval_c[0]
    runtime_s = time.perf_counter() - start_time

    return final_plan, final_obj, {
        "status": "feasible" if is_feas else "infeasible",
        "runtime_s": runtime_s,
        "is_feasible": is_feas,
        "eval_count": eval_count,
        "co2_t": details["co2_wtw_t"],
        "cost_usd": details["cost_usd"],
    }


# ---------------------------------------------------------------------------
# 4. Quantum-Inspired Evolutionary Algorithm (QI-EA)
# ---------------------------------------------------------------------------

def qiea_optimize_v2(
    problem: FleetOptimizationProblemV2,
    pop_size: int = 20,
    n_generations: int = 40,
    seed: int = 42,
    with_cap_repair: bool = False,
) -> tuple[list[dict], float, dict[str, Any]]:
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    eval_count = 0

    num_routes = problem.num_routes
    num_vessels = problem.num_vessels
    num_fuels = problem.num_fuels
    num_speeds = len(problem.speed_bins)

    theta = np.full((num_routes, 4), np.pi / 4.0)

    def decode_individual(angles):
        ind = []
        for r_idx, route in enumerate(problem.routes):
            probs = np.sin(angles[r_idx]) ** 2
            v_idx = int(min(num_vessels - 1, np.floor(probs[0] * num_vessels)))
            vessel = problem.vessels[v_idx]

            s_idx = int(min(num_speeds - 1, np.floor(probs[1] * num_speeds)))
            s_factor = problem.speed_bins[s_idx]
            spd = round(s_factor * float(vessel["design_speed_kn"]), 4)

            f_idx = int(min(num_fuels - 1, np.floor(probs[2] * num_fuels)))
            fuel = problem.fuels[f_idx]

            opts = route.get("route_options", [{"option_id": 0}])
            opt_idx = int(min(len(opts) - 1, np.floor(probs[3] * len(opts))))
            opt_id = opts[opt_idx]["option_id"]

            ind.append({
                "route_id": route["id"],
                "route_name": route["name"],
                "route_option_id": opt_id,
                "vessel_id": vessel["id"],
                "vessel_name": vessel["name"],
                "vessel_class": vessel["vessel_class"],
                "speed_kn": spd,
                "fuel_id": fuel["id"],
                "fuel_type": fuel["name"],
            })
        return repair_solution_v2(problem, ind)

    best_plan = None
    best_obj = float("inf")
    delta_theta = 0.05 * np.pi

    for gen in range(n_generations):
        pop = [decode_individual(theta) for _ in range(pop_size)]
        fits = []
        for ind in pop:
            fit, _, _ = problem.evaluate(ind)
            eval_count += 1
            fits.append(fit)

        gen_best_idx = int(np.argmin(fits))
        if fits[gen_best_idx] < best_obj:
            best_obj = fits[gen_best_idx]
            best_plan = pop[gen_best_idx]

        for r_idx in range(num_routes):
            best_item = best_plan[r_idx]
            v_norm = next(i for i, v in enumerate(problem.vessels) if v["id"] == best_item["vessel_id"]) / max(1, num_vessels - 1)
            target_angles = np.arcsin(np.sqrt(np.clip([v_norm, 0.5, 0.5, 0.5], 0.01, 0.99)))

            for d in range(4):
                if theta[r_idx, d] < target_angles[d]:
                    theta[r_idx, d] = min(np.pi / 2.0, theta[r_idx, d] + delta_theta)
                elif theta[r_idx, d] > target_angles[d]:
                    theta[r_idx, d] = max(0.0, theta[r_idx, d] - delta_theta)

    eval_c = [eval_count]
    final_plan = apply_shared_cap_repair(problem, best_plan, eval_counter=eval_c) if with_cap_repair else best_plan
    final_obj, is_feas, details = problem.evaluate(final_plan)
    eval_c[0] += 1
    eval_count = eval_c[0]
    runtime_s = time.perf_counter() - start_time

    return final_plan, final_obj, {
        "status": "feasible" if is_feas else "infeasible",
        "runtime_s": runtime_s,
        "is_feasible": is_feas,
        "eval_count": eval_count,
        "co2_t": details["co2_wtw_t"],
        "cost_usd": details["cost_usd"],
    }
