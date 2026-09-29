"""
Diagnostics for Items 1, 2, and 3
"""

import numpy as np
import pandas as pd
from optimizer.problem import FleetOptimizationProblem
from optimizer.baselines import milp_optimize

print("=" * 80)
print("ITEM 1: TRUE UNCONSTRAINED OPTIMUM VS MILP VS CAP")
print("=" * 80)

for inst_name, sc_id in [("Small", 2), ("Medium", 3), ("Large", 4)]:
    prob = FleetOptimizationProblem(scenario_id=sc_id)
    
    # True unconstrained plan: each route independently chooses the lowest-cost (vessel, speed, fuel)
    # that satisfies cargo demand (capacity >= demand) and transit deadline (delay == 0),
    # ignoring global fleet vessel limits and emissions cap.
    unconstrained_plan = []
    for r in prob.routes:
        best_cost = float("inf")
        best_choice = None
        for v in prob.vessels:
            if v["capacity_teu"] < r["cargo_demand_teu"]:
                continue  # Must satisfy cargo demand
            for s_mult in [0.6, 0.7, 0.8, 0.9, 1.0]:
                spd = round(float(s_mult * v["design_speed_kn"]), 4)
                duration_days = r["distance_nm"] / (spd * 24.0)
                delay_days = max(0.0, duration_days - r["deadline_days"])
                for f in prob.fuels:
                    key = (r["id"], v["id"], spd, f["id"])
                    e = prob._lookup_cache[key]
                    cost = e["fuel_cost"] + (prob.weights["w2_co2_emission"] * e["co2_wtw_t"]) + (prob.weights["w3_schedule_penalty"] * delay_days)
                    if cost < best_cost:
                        best_cost = cost
                        best_choice = {
                            "route_id": r["id"],
                            "route_name": r["name"],
                            "vessel_id": v["id"],
                            "vessel_name": v["name"],
                            "vessel_class": v["vessel_class"],
                            "speed_kn": spd,
                            "fuel_id": f["id"],
                            "fuel_type": f["name"],
                        }
        unconstrained_plan.append(best_choice)
        
    uncon_obj, uncon_feas, uncon_det = prob.evaluate(unconstrained_plan)
    
    # MILP plan
    milp_sol, milp_obj, _ = milp_optimize(prob, seed=42)
    milp_plan = prob.decode_solution(milp_sol)
    m_obj, m_feas, m_det = prob.evaluate(milp_plan)
    
    print(f"\n--- {inst_name} Instance ---")
    print(f"  Emissions Cap: {prob.emissions_cap_t:,.1f} t CO2eq")
    print(f"  True Unconstrained Plan (Cargo-Feasible, No Quota/Cap Constraints):")
    print(f"    Objective J:   ${uncon_obj:,.2f}")
    print(f"    Fuel Cost:     ${uncon_det['fuel_cost_usd']:,.2f}")
    print(f"    CO2 Emissions: {uncon_det['co2_wtw_t']:,.1f} t CO2eq (Cap Headroom: {prob.emissions_cap_t - uncon_det['co2_wtw_t']:,.1f} t)")
    print(f"    Delay Days:    {uncon_det['total_delay_days']:.2f} d")
    print(f"    Vessel Usage Breakdown:")
    counts = {}
    for p in unconstrained_plan:
        counts[p["vessel_name"]] = counts.get(p["vessel_name"], 0) + 1
    for v_name, cnt in counts.items():
        lim = prob.vessel_limits.get(v_name, 99)
        status = "OK" if cnt <= lim else f"EXCEEDED (+{cnt-lim})"
        print(f"      {v_name}: used {cnt} times (limit {lim}) -> {status}")

    print(f"  MILP Constrained Plan:")
    print(f"    Objective J:   ${m_obj:,.2f}")
    print(f"    Fuel Cost:     ${m_det['fuel_cost_usd']:,.2f}")
    print(f"    CO2 Emissions: {m_det['co2_wtw_t']:,.1f} t CO2eq (Headroom: {prob.emissions_cap_t - m_det['co2_wtw_t']:,.1f} t)")
    print(f"    Delay Days:    {m_det['total_delay_days']:.2f} d")
    print(f"    Feasible:      {m_feas}")

print("\n" + "=" * 80)
print("ITEM 2: SMALL INSTANCE ROUTE-06 & LNG DISPATCH RECONCILIATION")
print("=" * 80)

prob_s = FleetOptimizationProblem(scenario_id=2)
r6 = prob_s.routes[5]  # Route-06: Rotterdam -> Hamburg (350 nm, Demand: 1400 TEU, Deadline: 2.5 d)
print(f"Route-06 details: {r6['name']}, Dist={r6['distance_nm']} nm, Demand={r6['cargo_demand_teu']} TEU, Deadline={r6['deadline_days']} days")

print("\nCandidate choices on Route-06 with Panamax-01:")
for s_mult in [0.6, 0.7, 0.8, 0.9, 1.0]:
    v = next(v for v in prob_s.vessels if v["name"] == "Panamax-01")
    spd = round(float(s_mult * v["design_speed_kn"]), 4)
    duration_days = r6["distance_nm"] / (spd * 24.0)
    delay_days = max(0.0, duration_days - r6["deadline_days"])
    for f in prob_s.fuels:
        key = (r6["id"], v["id"], spd, f["id"])
        e = prob_s._lookup_cache[key]
        total_j = e["fuel_cost"] + (100.0 * e["co2_wtw_t"]) + (100000.0 * delay_days)
        print(f"  Speed {spd:4.1f} kn ({s_mult:.0%}) | Fuel {f['name']:8s} | Fuel Cost: ${e['fuel_cost']:8,.2f} | CO2: {e['co2_wtw_t']:6.2f} t | Delay: {delay_days:.2f} d | J: ${total_j:10,.2f}")

print("\nCandidate choices on Route-06 with Feeder-01 (Capacity 1,000 TEU vs Demand 1,400 TEU -> 400 TEU shortfall penalty):")
for s_mult in [0.6, 0.7, 0.8, 0.9, 1.0]:
    v = next(v for v in prob_s.vessels if v["name"] == "Feeder-01")
    spd = round(float(s_mult * v["design_speed_kn"]), 4)
    duration_days = r6["distance_nm"] / (spd * 24.0)
    delay_days = max(0.0, duration_days - r6["deadline_days"])
    shortfall = max(0, r6["cargo_demand_teu"] - v["capacity_teu"])
    shortfall_pen = shortfall * 200.0
    for f in prob_s.fuels:
        key = (r6["id"], v["id"], spd, f["id"])
        e = prob_s._lookup_cache[key]
        total_j = e["fuel_cost"] + (100.0 * e["co2_wtw_t"]) + (100000.0 * delay_days) + shortfall_pen
        if f["name"] in ["HFO", "LNG"]:
            print(f"  Speed {spd:4.1f} kn ({s_mult:.0%}) | Fuel {f['name']:8s} | Fuel Cost: ${e['fuel_cost']:8,.2f} | CO2: {e['co2_wtw_t']:6.2f} t | Shortfall Pen: ${shortfall_pen:6,.0f} | J: ${total_j:10,.2f}")
