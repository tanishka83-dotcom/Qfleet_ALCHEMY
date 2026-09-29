"""
Diagnostics for Item 4: QI-EA In-Depth Analysis
"""

import numpy as np
import pandas as pd
from optimizer.problem import FleetOptimizationProblem
from optimizer.qi_optimizer import qiea_optimize, repair_solution

print("=" * 80)
print("ITEM 4(b): QI-EA PLANS AT GENERATION 0 VS END FOR 3 SEEDS (MEDIUM)")
print("=" * 80)

prob_m = FleetOptimizationProblem(scenario_id=3)

for seed in [42, 43, 44]:
    rng = np.random.default_rng(seed)
    pop_size = 50
    n_generations = 60
    num_routes = prob_m.num_routes
    
    # Generation 0: Initial angles theta = pi/4 -> sin^2(pi/4) = 0.5
    q_angles = np.full((pop_size, num_routes, 3), np.pi / 4.0, dtype=float)
    
    # Observe ind 0
    sin2 = np.sin(q_angles[0]) ** 2
    sampled_gen0 = np.clip(sin2 + rng.normal(0.0, 0.02, size=sin2.shape), 0.0, 1.0)
    decoded_gen0 = prob_m.decode_solution(sampled_gen0)
    repaired_gen0 = repair_solution(prob_m, decoded_gen0)
    obj_gen0, feas_gen0, det_gen0 = prob_m.evaluate(repaired_gen0)
    
    # End of run:
    sol_end, obj_end, _ = qiea_optimize(prob_m, seed=seed, pop_size=pop_size, n_generations=n_generations)
    plan_end = prob_m.decode_solution(sol_end)
    obj_end, feas_end, det_end = prob_m.evaluate(plan_end)
    
    print(f"\n--- SEED {seed} ---")
    print(f"Generation 0:")
    print(f"  Obj: ${obj_gen0:,.2f} | Feas: {feas_gen0} | CO2: {det_gen0['co2_wtw_t']:,.1f} t")
    print(f"  Vessels: {[p['vessel_name'] for p in repaired_gen0]}")
    print(f"  Fuels:   {[p['fuel_type'] for p in repaired_gen0]}")
    print(f"End (Gen {n_generations}):")
    print(f"  Obj: ${obj_end:,.2f} | Feas: {feas_end} | CO2: {det_end['co2_wtw_t']:,.1f} t")
    print(f"  Vessels: {[p['vessel_name'] for p in plan_end]}")
    print(f"  Fuels:   {[p['fuel_type'] for p in plan_end]}")

print("\n" + "=" * 80)
print("ITEM 4(e) & (f): RANDOM SEARCH & REPAIR-CHANGE RATE (MEDIUM & LARGE)")
print("=" * 80)

for inst_name, sc_id, budget in [("Medium", 3, 3000), ("Large", 4, 6000)]:
    prob = FleetOptimizationProblem(scenario_id=sc_id)
    rng = np.random.default_rng(42)
    
    best_rs_obj = float("inf")
    best_rs_feas = False
    best_rs_det = None
    repairs_changed = 0
    
    for _ in range(budget):
        raw = rng.uniform(0.0, 1.0, size=(prob.num_routes, 3))
        decoded = prob.decode_solution(raw)
        repaired = repair_solution(prob, decoded)
        
        changed = any(d["vessel_id"] != r["vessel_id"] or d["speed_kn"] != r["speed_kn"] for d, r in zip(decoded, repaired))
        if changed:
            repairs_changed += 1
            
        obj, feas, det = prob.evaluate(repaired)
        if obj < best_rs_obj:
            best_rs_obj = obj
            best_rs_feas = feas
            best_rs_det = det
            
    print(f"[{inst_name}] Random Search ({budget} evals):")
    print(f"  Best Obj:      ${best_rs_obj:,.2f}")
    print(f"  Feasible:      {best_rs_feas}")
    print(f"  CO2 Emitted:   {best_rs_det['co2_wtw_t']:,.1f} t vs Cap {prob.emissions_cap_t:,.1f} t")
    print(f"  Repair Rate:   {repairs_changed}/{budget} ({repairs_changed/budget:.2%})")
