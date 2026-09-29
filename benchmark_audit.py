import db
import config
from optimizer.problem import FleetOptimizationProblem
from optimizer.baselines import milp_optimize, greedy_optimize, ga_optimize
from optimizer.qi_optimizer import qiea_optimize, sqa_optimize

problem = FleetOptimizationProblem(scenario_id=1)

print("=== 1. EXACT GROUND TRUTH ===")
exact_plan, exact_obj, is_feas, viols = problem.exact_brute_force_optimal()
print(f"Coupled Exact Optimum J*: {exact_obj:,.2f} (Feasible: {is_feas})")

uncoupled_plan, uncoupled_obj, u_feas, u_details = problem.uncoupled_brute_force_optimal()
print(f"Uncoupled Optimum (no fleet quotas): {uncoupled_obj:,.2f} (Base Obj: ${u_details['base_objective']:,.2f}, Feasible: {u_feas})")

print("\n=== 2. RUNNING 5 OPTIMIZERS (IDENTICAL 1,200 EVAL BUDGET FOR HEURISTICS) ===")
algos = [
    ("MILP (HiGHS)", lambda: milp_optimize(problem, seed=42)),
    ("Greedy", lambda: greedy_optimize(problem, seed=42)),
    ("GA", lambda: ga_optimize(problem, seed=42, pop_size=30, n_generations=40)),
    ("SQA", lambda: sqa_optimize(problem, seed=42, n_trotters=24, n_steps=50)),
    ("QI-EA", lambda: qiea_optimize(problem, seed=42, pop_size=30, n_generations=40))
]

import time

results = []
for name, func in algos:
    problem.eval_count = 0
    t0 = time.perf_counter()
    sol, obj, hist = func()
    t1 = time.perf_counter()
    full_obj, feas, v_list = problem.evaluate(sol)
    evals = problem.eval_count
    runtime_s = t1 - t0
    
    f_cost, co2_t, delay = 0.0, 0.0, 0.0
    for a in sol:
        k = (a['route_id'], a['vessel_id'], round(a['speed_kn'], 4), a['fuel_id'])
        e = problem._lookup_cache[k]
        f_cost += e['fuel_cost']
        co2_t += e['co2_wtw_t']
        r = [rt for rt in problem.routes if rt['id'] == a['route_id']][0]
        delay += max(0.0, r['distance_nm'] / (a['speed_kn'] * 24.0) - r['deadline_days'])
    
    gap = ((full_obj - exact_obj) / exact_obj) * 100.0
    results.append((name, full_obj, gap, f_cost, co2_t, delay, feas, evals, runtime_s))

header = f"{'Algorithm':<15} | {'Objective ($)':<14} | {'Gap (%)':<8} | {'Fuel Cost ($)':<14} | {'CO2 (t)':<10} | {'Delay (d)':<10} | {'Feasible':<8} | {'Evals':<6} | {'Runtime (s)':<11}"
print(header)
print("-" * len(header))
for name, full_obj, gap, f_cost, co2_t, delay, feas, evals, runtime_s in results:
    print(f"{name:<15} | {full_obj:14,.2f} | {gap:7.2f}% | {f_cost:14,.2f} | {co2_t:10.2f} | {delay:10.4f} | {str(feas):<8} | {evals:<6} | {runtime_s:11.4f}")

print("\n=== 3. FUEL TYPE BREAKDOWN & BREAKEVEN CARBON PRICE ===")
# Evaluate total fleet cost and CO2 if fleet forced to use each fuel on exact optimal vessel/speed plan
for fuel in problem.fuels:
    total_fuel_cost = 0.0
    total_co2_wtw = 0.0
    for a in exact_plan:
        k = (a['route_id'], a['vessel_id'], round(a['speed_kn'], 4), fuel['id'])
        e = problem._lookup_cache[k]
        total_fuel_cost += e['fuel_cost']
        total_co2_wtw += e['co2_wtw_t']
    total_obj = total_fuel_cost + 100.0 * total_co2_wtw
    print(f"Fuel: {fuel['name']:<12} | Fuel Cost: ${total_fuel_cost:12,.2f} | CO2 (WTW): {total_co2_wtw:8.2f} t | Total J (w2=100): ${total_obj:12,.2f}")
