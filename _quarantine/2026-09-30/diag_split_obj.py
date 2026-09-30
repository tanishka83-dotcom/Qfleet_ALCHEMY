"""
Diagnostics for Item 3: Objective Breakdown into Cost and Penalties
"""

import pandas as pd
import numpy as np
import config
from optimizer.problem import FleetOptimizationProblem
from optimizer.baselines import milp_optimize, greedy_optimize, ga_optimize
from optimizer.qi_optimizer import sqa_optimize, qiea_optimize

print("=" * 80)
print("ITEM 3: OBJECTIVE SPLIT (COST VS PENALTY) & PENALTY FORMULA")
print("=" * 80)

rows = []
for inst_name, sc_id in [("Small", 2), ("Medium", 3), ("Large", 4)]:
    prob = FleetOptimizationProblem(scenario_id=sc_id)
    
    algos = [
        ("MILP_HiGHS", lambda p, s: milp_optimize(p, seed=s)),
        ("Greedy", lambda p, s: greedy_optimize(p, seed=s)),
        ("GA", lambda p, s: ga_optimize(p, seed=s, pop_size=30 if inst_name=="Small" else (50 if inst_name=="Medium" else 60), n_generations=40 if inst_name=="Small" else (60 if inst_name=="Medium" else 100))),
        ("SQA", lambda p, s: sqa_optimize(p, seed=s, n_trotters=24 if inst_name=="Small" else (30 if inst_name=="Medium" else 40), n_steps=49 if inst_name=="Small" else (99 if inst_name=="Medium" else 149))),
        ("QI-EA", lambda p, s: qiea_optimize(p, seed=s, pop_size=30 if inst_name=="Small" else (50 if inst_name=="Medium" else 60), n_generations=40 if inst_name=="Small" else (60 if inst_name=="Medium" else 100))),
    ]
    
    for a_name, fn in algos:
        seeds = [42] if a_name in ["MILP_HiGHS", "Greedy"] else range(42, 52)
        
        fuel_costs = []
        co2_costs = []
        delay_pens = []
        cargo_pens = []
        emiss_pens = []
        overlap_pens = []
        total_objs = []
        feass = []
        
        for s in seeds:
            sol, _, _ = fn(prob, s)
            plan = prob.decode_solution(sol)
            tot_obj, is_feas, det = prob.evaluate(plan)
            
            w1 = prob.weights.get("w1_fuel_cost", 1.0)
            w2 = prob.weights.get("w2_co2_emission", 100.0)
            w3 = prob.weights.get("w3_schedule_penalty", 100000.0)
            
            f_cost = w1 * det["fuel_cost_usd"]
            c_cost = w2 * det["co2_wtw_t"]
            d_pen = w3 * det["total_delay_days"]
            cg_pen = det["cargo_unmet_teu"] * config.CONSTRAINT_PENALTIES.get("cargo_capacity_per_teu", 200.0)
            em_pen = det["emissions_excess_t"] * config.CONSTRAINT_PENALTIES.get("emissions_cap_per_tonne", 500.0)
            ov_pen = det["vessel_overlap_conflicts"] * config.CONSTRAINT_PENALTIES.get("vessel_overlap", 500000.0)
            
            fuel_costs.append(f_cost)
            co2_costs.append(c_cost)
            delay_pens.append(d_pen)
            cargo_pens.append(cg_pen)
            emiss_pens.append(em_pen)
            overlap_pens.append(ov_pen)
            total_objs.append(tot_obj)
            feass.append(is_feas)
            
        rows.append({
            "Instance": inst_name,
            "Algorithm": a_name,
            "Feas Rate": f"{np.mean(feass):.0%}",
            "Mean Fuel Cost ($)": f"${np.mean(fuel_costs):,.0f}",
            "Mean CO2 Cost ($)": f"${np.mean(co2_costs):,.0f}",
            "Delay Pen ($)": f"${np.mean(delay_pens):,.0f}",
            "Cargo Pen ($)": f"${np.mean(cargo_pens):,.0f}",
            "Emiss Cap Pen ($)": f"${np.mean(emiss_pens):,.0f}",
            "Quota Overlap Pen ($)": f"${np.mean(overlap_pens):,.0f}",
            "Mean Total J ($)": f"${np.mean(total_objs):,.0f}",
        })

df_split = pd.DataFrame(rows)
print(df_split.to_string(index=False))
