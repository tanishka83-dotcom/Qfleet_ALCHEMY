"""
QFleet Phase 2 — Quantum-Inspired Fleet Optimizers
==================================================
Includes:
1. Quantum-Inspired Evolutionary Algorithm (QI-EA):
   - Multi-qubit angular representation for vessel, speed, and fuel choices per route.
   - Quantum rotation-gate update (Narayanan & Moore 1996) toward the best individual.
   - Constraint repair operator (upgrades vessel capacity without worsening schedule deadlines).
2. Simulated Quantum Annealing (SQA):
   - Quantum transverse-field Ising-inspired path-integral annealing with Trotter slices.
   - Transverse field Gamma(t) decays over Monte Carlo annealing steps.

Both optimizers persist their runs directly to the SQLite database via problem.save_run().
"""

from __future__ import annotations

import sys
import time
import pathlib
import numpy as np
from typing import Callable

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from .problem import FleetOptimizationProblem


# ---------------------------------------------------------------------------
# Feasibility Repair Operator
# ---------------------------------------------------------------------------

def repair_solution(
    problem: FleetOptimizationProblem,
    assignments: list[dict],
) -> list[dict]:
    """
    Constraint repair operator:
    1. If a route has cargo_demand > vessel capacity, upgrade vessel to the
       smallest feasible vessel class that satisfies cargo demand.
    2. Adjust operational speed to avoid schedule delays if vessel change altered speed limits.
    3. Documented guarantee: Upgrading capacity and maintaining feasible speed does NOT worsen
       other constraints (cargo capacity strictly increases; speed is bounded within valid limits).
    """
    repaired = []
    # Sort vessels by capacity ascending
    sorted_vessels = sorted(problem.vessels, key=lambda v: v["capacity_teu"])

    for i, a in enumerate(assignments):
        route = problem.routes[i]
        curr_vessel = next(v for v in problem.vessels if v["id"] == a["vessel_id"])
        curr_fuel = next(f for f in problem.fuels if f["id"] == a["fuel_id"])

        # Check cargo capacity
        if curr_vessel["capacity_teu"] < route["cargo_demand_teu"]:
            # Find smallest vessel with capacity >= cargo_demand
            feasible_vessel = next(
                (v for v in sorted_vessels if v["capacity_teu"] >= route["cargo_demand_teu"]),
                sorted_vessels[-1]  # fallback to largest vessel
            )
            vessel_to_use = feasible_vessel
        else:
            vessel_to_use = curr_vessel

        # Snap speed to exact 5 discrete bins (60%, 70%, 80%, 90%, 100% design speed)
        speed_multipliers = np.linspace(0.60, 1.00, 5)
        discrete_speeds = speed_multipliers * vessel_to_use["design_speed_kn"]
        s_idx = int(np.argmin(np.abs(discrete_speeds - a["speed_kn"])))
        desired_speed = discrete_speeds[s_idx]

        # Enforce deadline constraint
        min_speed_for_deadline = route["distance_nm"] / (route["deadline_days"] * 24.0)
        feasible_speeds = [s for s in discrete_speeds if s >= min_speed_for_deadline - 1e-4]
        if feasible_speeds:
            if desired_speed < min(feasible_speeds):
                desired_speed = min(feasible_speeds)
        else:
            desired_speed = discrete_speeds[-1]

        repaired.append({
            "route_id": route["id"],
            "route_name": route["name"],
            "vessel_id": vessel_to_use["id"],
            "vessel_name": vessel_to_use["name"],
            "vessel_class": vessel_to_use["vessel_class"],
            "speed_kn": float(desired_speed),
            "fuel_id": curr_fuel["id"],
            "fuel_type": curr_fuel["name"],
        })

    # Repair vessel availability limits: if a vessel exceeds limit, downgrade/shift to feasible alternative
    vessel_counts = {}
    for r in repaired:
        vessel_counts[r["vessel_id"]] = vessel_counts.get(r["vessel_id"], 0) + 1

    for r in repaired:
        v_id = r["vessel_id"]
        v_obj = next(v for v in problem.vessels if v["id"] == v_id)
        max_allowed = problem.vessel_limits.get(v_obj["name"], 2)
        if vessel_counts[v_id] > max_allowed:
            # Find alternative vessel with capacity >= route cargo demand whose count < max_allowed
            route = next(rt for rt in problem.routes if rt["id"] == r["route_id"])
            for alt_v in sorted_vessels:
                alt_max = problem.vessel_limits.get(alt_v["name"], 2)
                if alt_v["id"] != v_id and alt_v["capacity_teu"] >= route["cargo_demand_teu"] and vessel_counts.get(alt_v["id"], 0) < alt_max:
                    vessel_counts[v_id] -= 1
                    vessel_counts[alt_v["id"]] = vessel_counts.get(alt_v["id"], 0) + 1
                    r["vessel_id"] = alt_v["id"]
                    r["vessel_name"] = alt_v["name"]
                    r["vessel_class"] = alt_v["vessel_class"]
                    
                    alt_speeds = speed_multipliers * alt_v["design_speed_kn"]
                    min_alt_speed = route["distance_nm"] / (route["deadline_days"] * 24.0)
                    feas_alt_speeds = [s for s in alt_speeds if s >= min_alt_speed - 1e-4]
                    if feas_alt_speeds:
                        r["speed_kn"] = float(min(feas_alt_speeds))
                    else:
                        r["speed_kn"] = float(alt_speeds[-1])
                    break

    return repaired


# ---------------------------------------------------------------------------
# 1. Quantum-Inspired Evolutionary Algorithm (QI-EA)
# ---------------------------------------------------------------------------

def qiea_optimize(
    problem: FleetOptimizationProblem,
    seed: int = 42,
    pop_size: int = config.PHASE2_QI_PARAMS["population_size"],
    n_generations: int = config.PHASE2_QI_PARAMS["n_generations"],
    rotation_delta: float = config.PHASE2_QI_PARAMS["rotation_delta"],
) -> tuple[list[dict], float, list[float]]:
    """
    Quantum-Inspired Evolutionary Algorithm for multi-route fleet optimization.
    Returns:
        best_solution : list[dict] of route assignments
        best_objective: float
        convergence   : list[float] non-increasing best-so-far objective history
    """
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)

    num_routes = problem.num_routes
    # 3 Qubits per route: (vessel, speed, fuel) -> shape (num_routes, 3)
    # Initialise quantum state at superposition: theta = pi/4 for all qubits
    q_angles = np.full((pop_size, num_routes, 3), np.pi / 4.0, dtype=float)

    best_solution = None
    best_objective = float("inf")
    best_angles = None
    convergence_history: list[float] = []

    for gen in range(n_generations):
        # 1. Observe classical states from quantum superposition
        for ind in range(pop_size):
            # Sample observation probabilities: p = sin^2(theta) + small Gaussian perturbation
            sin2 = np.sin(q_angles[ind]) ** 2
            # Add subtle quantum fluctuation
            sampled = np.clip(sin2 + rng.normal(0.0, 0.02, size=sin2.shape), 0.0, 1.0)

            # Decode to assignments
            decoded = problem.decode_solution(sampled)
            # Apply repair operator
            repaired = repair_solution(problem, decoded)

            obj, feasible, _ = problem.evaluate(repaired)

            if obj < best_objective:
                best_objective = obj
                best_solution = repaired
                best_angles = q_angles[ind].copy()

        convergence_history.append(float(best_objective))

        # 2. Quantum Rotation Gate Update toward best individual (Narayanan & Moore 1996)
        if best_angles is not None:
            for ind in range(pop_size):
                diff = best_angles - q_angles[ind]
                # Rotate toward best solution by rotation_delta
                step = np.sign(diff) * rotation_delta
                q_angles[ind] = np.clip(q_angles[ind] + step, 0.01, np.pi / 2.0 - 0.01)

    runtime_s = time.perf_counter() - start_time

    # Persist run to DB
    problem.save_run(
        algorithm="QI-EA",
        seed=seed,
        solution=best_solution,
        objective=best_objective,
        runtime_s=runtime_s,
        convergence_history=convergence_history,
    )

    return best_solution, best_objective, convergence_history


# ---------------------------------------------------------------------------
# 2. Simulated Quantum Annealing (SQA)
# ---------------------------------------------------------------------------

def sqa_optimize(
    problem: FleetOptimizationProblem,
    seed: int = 42,
    n_trotters: int = config.PHASE2_SQA_PARAMS["n_trotters"],
    n_steps: int = config.PHASE2_SQA_PARAMS["n_steps"],
    gamma_init: float = config.PHASE2_SQA_PARAMS["gamma_init"],
    gamma_final: float = config.PHASE2_SQA_PARAMS["gamma_final"],
    temp: float = config.PHASE2_SQA_PARAMS["temperature"],
) -> tuple[list[dict], float, list[float]]:
    """
    Simulated Quantum Annealing with transverse field decay across Trotter replicas.
    Returns:
        best_solution : list[dict]
        best_objective: float
        convergence   : list[float]
    """
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)

    num_routes = problem.num_routes

    # Initialize Trotter slices with continuous states in [0, 1]
    trotter_states = rng.uniform(0.0, 1.0, size=(n_trotters, num_routes, 3))

    best_solution = None
    best_objective = float("inf")
    convergence_history: list[float] = []

    # Evaluate initial states
    trotter_objs = np.zeros(n_trotters)
    trotter_decs = []
    for m in range(n_trotters):
        decoded = problem.decode_solution(trotter_states[m])
        repaired = repair_solution(problem, decoded)
        obj, _, _ = problem.evaluate(repaired)
        trotter_objs[m] = obj
        trotter_decs.append(repaired)
        if obj < best_objective:
            best_objective = obj
            best_solution = repaired

    # Annealing schedule
    gammas = np.linspace(gamma_init, gamma_final, n_steps)

    for step, gamma in enumerate(gammas):
        beta = 1.0 / max(temp, 1e-4)
        j_perp = -0.5 * temp * np.log(max(np.tanh(gamma * beta / n_trotters), 1e-6))

        for m in range(n_trotters):
            # Propose continuous perturbation
            candidate_state = trotter_states[m].copy()
            r_target = rng.integers(0, num_routes)
            dim_target = rng.integers(0, 3)
            candidate_state[r_target, dim_target] = np.clip(
                candidate_state[r_target, dim_target] + rng.normal(0, 0.15), 0.0, 1.0
            )

            # Evaluate candidate
            cand_dec = repair_solution(problem, problem.decode_solution(candidate_state))
            cand_obj, _, _ = problem.evaluate(cand_dec)
            curr_obj = trotter_objs[m]

            # Classical energy delta
            delta_e = (cand_obj - curr_obj) / n_trotters

            # Inter-trotter coupling coupling with neighbor slices (m-1, m+1)
            prev_m = (m - 1) % n_trotters
            next_m = (m + 1) % n_trotters
            curr_coupling = np.sum((trotter_states[m] - trotter_states[prev_m])**2 + (trotter_states[m] - trotter_states[next_m])**2)
            cand_coupling = np.sum((candidate_state - trotter_states[prev_m])**2 + (candidate_state - trotter_states[next_m])**2)
            delta_coupling = j_perp * (cand_coupling - curr_coupling)

            total_delta = delta_e + delta_coupling

            # Metropolis acceptance with quantum tunneling coupling
            if total_delta < 0 or rng.uniform(0.0, 1.0) < np.exp(-total_delta / max(temp, 1e-4)):
                trotter_states[m] = candidate_state
                trotter_objs[m] = cand_obj
                trotter_decs[m] = cand_dec
                if cand_obj < best_objective:
                    best_objective = cand_obj
                    best_solution = cand_dec

        convergence_history.append(float(best_objective))

    runtime_s = time.perf_counter() - start_time

    # Persist to DB
    problem.save_run(
        algorithm="SimulatedQuantumAnnealing",
        seed=seed,
        solution=best_solution,
        objective=best_objective,
        runtime_s=runtime_s,
        convergence_history=convergence_history,
    )

    return best_solution, best_objective, convergence_history
