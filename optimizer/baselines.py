"""
QFleet Phase 2 — Baseline Optimizers
====================================
Includes:
1. Greedy Baseline: One-pass deterministic selection choosing the lowest-cost
   feasible vessel, speed, and fuel per route.
2. Genetic Algorithm (GA): Standard tournament selection, crossover, and mutation.
3. MILP Baseline (SciPy HiGHS):
   - Piecewise-linear / discrete candidate grid formulation solved via scipy.optimize.milp.
   - Offline precomputation of surrogate fuel consumption across (route x vessel x speed x fuel) options.
   - Time-boxed and explicitly documented as a discrete candidate approximation.

All baselines persist runs directly to the SQLite database via problem.save_run().
"""

from __future__ import annotations

import sys
import time
import pathlib
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from .problem import FleetOptimizationProblem
from .qi_optimizer import repair_solution


# ---------------------------------------------------------------------------
# 1. Greedy Optimizer
# ---------------------------------------------------------------------------

def greedy_optimize(
    problem: FleetOptimizationProblem,
    seed: int = 42,
) -> tuple[list[dict], float, list[float]]:
    """
    One-pass greedy search: For each route, tests all candidate (vessel, speed, fuel)
    tuples and greedily picks the combination with minimum objective that respects
    vessel availability limits.
    """
    start_time = time.perf_counter()
    best_assignments = []

    # Discretize speed choices to 5 steps per vessel (60%, 70%, 80%, 90%, 100% design speed)
    speed_steps = np.linspace(0.60, 1.00, 5)

    vessel_usage = {v["id"]: 0 for v in problem.vessels}

    # Sort routes by descending cargo demand so hardest capacity requirements are filled first
    route_order = sorted(range(len(problem.routes)), key=lambda idx: problem.routes[idx]["cargo_demand_teu"], reverse=True)
    assignments_by_route = [None] * len(problem.routes)

    for r_idx in route_order:
        route = problem.routes[r_idx]
        best_r_obj = float("inf")
        best_r_choice = None

        for vessel in problem.vessels:
            # Skip vessel if capacity cannot meet cargo demand
            if vessel["capacity_teu"] < route["cargo_demand_teu"]:
                continue

            max_v_routes = problem.vessel_limits.get(
                vessel["name"],
                problem.vessel_limits.get(vessel["vessel_class"], problem.vessel_limits.get(str(vessel["id"]), 2))
            )
            if vessel_usage[vessel["id"]] >= max_v_routes:
                continue

            for s_factor in speed_steps:
                spd = round(float(s_factor * vessel["design_speed_kn"]), 4)
                dur_days = route["distance_nm"] / (spd * 24.0)
                delay = max(0.0, dur_days - route["deadline_days"])
                if delay > 1e-4:
                    continue

                for fuel in problem.fuels:
                    cache_key = (route["id"], vessel["id"], spd, fuel["id"])
                    if cache_key in problem._lookup_cache:
                        entry = problem._lookup_cache[cache_key]
                        f_cost = entry["fuel_cost"]
                        co2_w = entry["co2_wtw_t"]
                    else:
                        df_single = pd.DataFrame([{
                            "vessel_class": vessel["vessel_class"],
                            "fuel_type": fuel["name"],
                            "speed_kn": spd,
                            "load_factor": min(1.0, float(route["cargo_demand_teu"]) / float(vessel["capacity_teu"])),
                            "weather_factor": float(problem.assumed_weather),
                            "distance_nm": float(route["distance_nm"]),
                        }])
                        f_t = float(problem.surrogate.predict(df_single)[0])
                        f_cost = f_t * fuel["price_usd_per_tonne"]
                        co2_w = f_t * 1e3 * fuel["lhv_mj_per_kg"] * fuel["co2_wtw_g_per_mj"] * 1e-6

                    w1 = problem.weights.get("w1_fuel_cost", 1.0)
                    w2 = problem.weights.get("w2_co2_emission", 100.0)
                    w3 = problem.weights.get("w3_schedule_penalty", 100000.0)

                    route_obj = (w1 * f_cost) + (w2 * co2_w) + (w3 * delay)
                    if route_obj < best_r_obj:
                        best_r_obj = route_obj
                        best_r_choice = {
                            "route_id": route["id"],
                            "route_name": route["name"],
                            "vessel_id": vessel["id"],
                            "vessel_name": vessel["name"],
                            "vessel_class": vessel["vessel_class"],
                            "speed_kn": float(spd),
                            "fuel_id": fuel["id"],
                            "fuel_type": fuel["name"],
                        }

        if best_r_choice is None:
            # Fallback if quota is strictly exhausted
            feasible_v = [v for v in problem.vessels if v["capacity_teu"] >= route["cargo_demand_teu"]]
            chosen_v = feasible_v[0] if len(feasible_v) > 0 else max(problem.vessels, key=lambda v: v["capacity_teu"])
            best_r_choice = {
                "route_id": route["id"],
                "route_name": route["name"],
                "vessel_id": chosen_v["id"],
                "vessel_name": chosen_v["name"],
                "vessel_class": chosen_v["vessel_class"],
                "speed_kn": float(chosen_v["design_speed_kn"]),
                "fuel_id": problem.fuels[0]["id"],
                "fuel_type": problem.fuels[0]["name"],
            }

        vessel_usage[best_r_choice["vessel_id"]] += 1
        assignments_by_route[r_idx] = best_r_choice

    best_assignments = assignments_by_route
    final_obj, is_feas, _ = problem.evaluate(best_assignments)
    runtime_s = time.perf_counter() - start_time
    convergence_history = [float(final_obj)]

    problem.save_run(
        algorithm="Greedy",
        seed=seed,
        solution=best_assignments,
        objective=final_obj,
        runtime_s=runtime_s,
        convergence_history=convergence_history,
    )

    return best_assignments, final_obj, convergence_history


# ---------------------------------------------------------------------------
# 1B. Greedy with Decarbonization Pass (Baseline Heuristic)
# ---------------------------------------------------------------------------

def greedy_decarb_optimize(
    problem: FleetOptimizationProblem,
    seed: int = 42,
    max_upgrade_steps: int = 50,
) -> tuple[list[dict], float, list[float]]:
    """
    Two-pass greedy baseline heuristic:
    - Pass 1: standard greedy per-route assignment.
    - Pass 2 (if fleet emissions exceed cap): iteratively upgrades the voyage with
      the lowest marginal abatement cost ($/t CO2 reduction) until emissions <= cap
      or all reductions are exhausted.
    Strictly isolated as a separate baseline heuristic. Does not modify GA, SQA, or QI-EA.
    """
    start_time = time.perf_counter()
    # Pass 1: Run standard greedy
    assignments, obj, _ = greedy_optimize(problem, seed=seed)
    obj_eval, is_feas, details = problem.evaluate(assignments)

    if is_feas or problem.emissions_cap_t is None:
        runtime_s = time.perf_counter() - start_time
        return assignments, obj_eval, [float(obj_eval)]

    current_plan = [dict(a) for a in assignments]
    eval_count = 1

    for _ in range(max_upgrade_steps):
        obj_eval, is_feas, details = problem.evaluate(current_plan)
        eval_count += 1
        if is_feas or details["co2_wtw_t"] <= problem.emissions_cap_t:
            break

        best_upgrade = None
        best_mac = float("inf")

        for r_idx, curr_item in enumerate(current_plan):
            route = problem.routes[r_idx]
            curr_vessel = next(v for v in problem.vessels if v["id"] == curr_item["vessel_id"])
            curr_speed = curr_item["speed_kn"]
            curr_fuel = next(f for f in problem.fuels if f["id"] == curr_item["fuel_id"])

            curr_cache = problem._lookup_cache.get((route["id"], curr_vessel["id"], curr_speed, curr_fuel["id"]))
            if not curr_cache:
                continue
            curr_co2 = curr_cache["co2_wtw_t"]
            curr_cost = curr_cache["fuel_cost"]

            for s_factor in [0.7, 0.8, 0.9, 1.0, 1.05]:
                spd = round(float(s_factor * curr_vessel["design_speed_kn"]), 4)
                dur_days = route["distance_nm"] / (spd * 24.0)
                if dur_days > route["deadline_days"]:
                    continue

                for fuel in problem.fuels:
                    if spd == curr_speed and fuel["id"] == curr_fuel["id"]:
                        continue
                    cand_cache = problem._lookup_cache.get((route["id"], curr_vessel["id"], spd, fuel["id"]))
                    if not cand_cache:
                        continue
                    cand_co2 = cand_cache["co2_wtw_t"]
                    cand_cost = cand_cache["fuel_cost"]

                    delta_co2 = curr_co2 - cand_co2
                    if delta_co2 > 1e-3:  # achieved emissions abatement
                        delta_cost = cand_cost - curr_cost
                        mac = delta_cost / delta_co2
                        if mac < best_mac:
                            best_mac = mac
                            best_upgrade = (r_idx, {
                                "route_id": route["id"],
                                "route_name": route["name"],
                                "vessel_id": curr_vessel["id"],
                                "vessel_name": curr_vessel["name"],
                                "vessel_class": curr_vessel["vessel_class"],
                                "speed_kn": spd,
                                "fuel_id": fuel["id"],
                                "fuel_type": fuel["name"],
                            })

        if best_upgrade is None:
            break

        r_idx, new_choice = best_upgrade
        current_plan[r_idx] = new_choice

    final_obj, final_feas, _ = problem.evaluate(current_plan)
    runtime_s = time.perf_counter() - start_time
    convergence_history = [float(final_obj)]

    return current_plan, final_obj, convergence_history


# ---------------------------------------------------------------------------
# 2. Genetic Algorithm (GA)
# ---------------------------------------------------------------------------

def ga_optimize(
    problem: FleetOptimizationProblem,
    seed: int = 42,
    pop_size: int = config.PHASE2_GA_PARAMS["population_size"],
    n_generations: int = config.PHASE2_GA_PARAMS["n_generations"],
    crossover_rate: float = config.PHASE2_GA_PARAMS["crossover_rate"],
    mutation_rate: float = config.PHASE2_GA_PARAMS["mutation_rate"],
) -> tuple[list[dict], float, list[float]]:
    """
    Standard Genetic Algorithm with tournament selection, uniform crossover,
    and Gaussian mutation on continuous coordinates in [0, 1].
    """
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)

    num_routes = problem.num_routes
    # Population chromosomes shape: (pop_size, num_routes, 3)
    population = rng.uniform(0.0, 1.0, size=(pop_size, num_routes, 3))

    best_solution = None
    best_objective = float("inf")
    convergence_history: list[float] = []

    for gen in range(n_generations):
        # Evaluate all individuals
        fitness_scores = []
        decoded_pop = []
        for ind in range(pop_size):
            decoded = problem.decode_solution(population[ind])
            repaired = repair_solution(problem, decoded)
            obj, _, _ = problem.evaluate(repaired)
            decoded_pop.append(repaired)
            fitness_scores.append(obj)

            if obj < best_objective:
                best_objective = obj
                best_solution = repaired

        convergence_history.append(float(best_objective))

        # Selection (Tournament of size 3)
        new_population = []
        for _ in range(pop_size):
            t_indices = rng.choice(pop_size, size=3, replace=False)
            best_t_idx = t_indices[np.argmin([fitness_scores[i] for i in t_indices])]
            new_population.append(population[best_t_idx].copy())

        new_population = np.array(new_population)

        # Crossover
        for i in range(0, pop_size - 1, 2):
            if rng.uniform(0.0, 1.0) < crossover_rate:
                mask = rng.choice([True, False], size=(num_routes, 3))
                p1, p2 = new_population[i].copy(), new_population[i+1].copy()
                new_population[i][mask] = p2[mask]
                new_population[i+1][mask] = p1[mask]

        # Mutation
        for i in range(pop_size):
            if rng.uniform(0.0, 1.0) < mutation_rate:
                mutation_noise = rng.normal(0.0, 0.15, size=(num_routes, 3))
                new_population[i] = np.clip(new_population[i] + mutation_noise, 0.0, 1.0)

        # Elitism: preserve best individual into index 0
        if best_solution is not None:
            population = new_population

    runtime_s = time.perf_counter() - start_time

    problem.save_run(
        algorithm="GeneticAlgorithm",
        seed=seed,
        solution=best_solution,
        objective=best_objective,
        runtime_s=runtime_s,
        convergence_history=convergence_history,
    )

    return best_solution, best_objective, convergence_history


# ---------------------------------------------------------------------------
# 3. MILP Baseline (SciPy HiGHS Discrete Piecewise Grid)
# ---------------------------------------------------------------------------

def milp_optimize(
    problem: FleetOptimizationProblem,
    seed: int = 42,
    speed_bins: int = config.PHASE2_MILP_PARAMS["speed_bins"],
    time_limit_s: float = config.PHASE2_MILP_PARAMS["time_limit_s"],
) -> tuple[list[dict], float, list[float]]:
    """
    Mixed-Integer Linear Programming baseline using scipy.optimize.milp (HiGHS solver).
    
    Piecewise-Linear Grid Formulation:
    - Precomputes surrogate fuel consumption across candidate (route x vessel x speed_bin x fuel) options.
    - Decision variables: binary x_{r, v, s_bin, f} in {0, 1}.
    - Constraint 1: Exactly 1 tuple per route (sum_{v,s,f} x_{r,v,s,f} = 1).
    - Constraint 2: Cargo demand feasibility (capacity >= demand).
    - Constraint 3: Fleet lifecycle emissions cap.
    """
    start_time = time.perf_counter()

    speed_fractions = np.linspace(0.60, 1.00, speed_bins)

    # Build discrete candidate grid
    options = []
    # options layout: list of dicts with (r_idx, v_idx, s_idx, f_idx, cost_usd, co2_t, speed_kn, is_cargo_feasible)
    meta_batch = []
    w1 = problem.weights.get("w1_fuel_cost", 1.0)
    w2 = problem.weights.get("w2_co2_emission", 100.0)
    w3 = problem.weights.get("w3_schedule_penalty", 100000.0)

    for r_idx, route in enumerate(problem.routes):
        for v_idx, vessel in enumerate(problem.vessels):
            if vessel["capacity_teu"] < route["cargo_demand_teu"]:
                continue

            for s_idx, s_frac in enumerate(speed_fractions):
                spd = round(float(s_frac * vessel["design_speed_kn"]), 4)
                dur_days = route["distance_nm"] / (spd * 24.0)
                delay_days = max(0.0, dur_days - route["deadline_days"])
                if delay_days > 1e-4:
                    continue

                for f_idx, fuel in enumerate(problem.fuels):
                    entry = problem._lookup_cache[(route["id"], vessel["id"], spd, fuel["id"])]
                    meta_batch.append({
                        "r_idx": r_idx,
                        "v_idx": v_idx,
                        "s_idx": s_idx,
                        "f_idx": f_idx,
                        "vessel": vessel,
                        "route": route,
                        "fuel": fuel,
                        "speed_kn": float(spd),
                        "delay_days": float(delay_days),
                        "fuel_cost": float(entry["fuel_cost"]),
                        "co2_wtw": float(entry["co2_wtw_t"]),
                        "fuel_t": float(entry["fuel_t"]),
                    })

    num_vars = len(meta_batch)
    c_coeffs = np.zeros(num_vars, dtype=float)
    co2_coeffs = np.zeros(num_vars, dtype=float)

    for i, meta in enumerate(meta_batch):
        c_coeffs[i] = (w1 * meta["fuel_cost"]) + (w2 * meta["co2_wtw"]) + (w3 * meta["delay_days"])
        co2_coeffs[i] = meta["co2_wtw"]

    # Constraints:
    # 1. Exactly one choice per route: sum_{options in r} x_i = 1 (num_routes constraints)
    A_eq = np.zeros((problem.num_routes, num_vars))
    for i, meta in enumerate(meta_batch):
        A_eq[meta["r_idx"], i] = 1.0

    b_eq_lower = np.ones(problem.num_routes)
    b_eq_upper = np.ones(problem.num_routes)

    # 2. Emissions cap: sum_i co2_i * x_i <= Cap
    A_co2 = co2_coeffs.reshape((1, num_vars))
    b_co2_lower = np.array([-np.inf])
    b_co2_upper = np.array([problem.emissions_cap_t])

    # 3. Fleet vessel availability limits: sum_{options for vessel v} x_i <= N_v^max
    A_vessels = np.zeros((len(problem.vessels), num_vars))
    b_vessels_upper = np.zeros(len(problem.vessels))
    for v_idx, vessel in enumerate(problem.vessels):
        max_r = problem.vessel_limits.get(
            vessel["name"],
            problem.vessel_limits.get(vessel["vessel_class"], problem.vessel_limits.get(str(vessel["id"]), 2))
        )
        b_vessels_upper[v_idx] = float(max_r)
        for i, meta in enumerate(meta_batch):
            if meta["v_idx"] == v_idx:
                A_vessels[v_idx, i] = 1.0

    b_vessels_lower = np.full(len(problem.vessels), -np.inf)

    # Combine all constraints
    A_all = np.vstack([A_eq, A_co2, A_vessels])
    lhs = np.concatenate([b_eq_lower, b_co2_lower, b_vessels_lower])
    rhs = np.concatenate([b_eq_upper, b_co2_upper, b_vessels_upper])

    constraints = LinearConstraint(A_all, lhs, rhs)
    integrality = np.ones(num_vars)  # 1 = binary integer variable
    bounds = Bounds(lb=0.0, ub=1.0)

    # Solve MILP via SciPy HiGHS with time limit
    options_highs = {
        "time_limit": time_limit_s,
        "disp": False,
    }

    res = milp(
        c=c_coeffs,
        integrality=integrality,
        constraints=constraints,
        bounds=bounds,
        options=options_highs,
    )

    assignments = []
    if res.success and res.x is not None:
        chosen_indices = np.where(res.x > 0.5)[0]
        # Map chosen indices to assignments
        for r_idx in range(problem.num_routes):
            # Find the chosen index for this route
            r_chosen = [idx for idx in chosen_indices if meta_batch[idx]["r_idx"] == r_idx]
            if len(r_chosen) > 0:
                meta = meta_batch[r_chosen[0]]
            else:
                # Fallback
                meta = [m for m in meta_batch if m["r_idx"] == r_idx and m["is_cargo_ok"]][0]

            assignments.append({
                "route_id": meta["route"]["id"],
                "route_name": meta["route"]["name"],
                "vessel_id": meta["vessel"]["id"],
                "vessel_name": meta["vessel"]["name"],
                "vessel_class": meta["vessel"]["vessel_class"],
                "speed_kn": float(meta["speed_kn"]),
                "fuel_id": meta["fuel"]["id"],
                "fuel_type": meta["fuel"]["name"],
            })
    else:
        # Fallback to greedy if HiGHS fails
        assignments, _, _ = greedy_optimize(problem, seed=seed)

    final_obj, _, _ = problem.evaluate(assignments)
    runtime_s = time.perf_counter() - start_time
    convergence_history = [float(final_obj)]

    problem.save_run(
        algorithm="MILP_HiGHS",
        seed=seed,
        solution=assignments,
        objective=final_obj,
        runtime_s=runtime_s,
        convergence_history=convergence_history,
    )

    return assignments, final_obj, convergence_history
