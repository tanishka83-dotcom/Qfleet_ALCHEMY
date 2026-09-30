import db
import config
from benchmark.instances import generate_benchmark_scenarios
from optimizer.problem import FleetOptimizationProblem
from optimizer.baselines import milp_optimize, greedy_optimize, ga_optimize

def greedy_decarb_optimize(problem, seed=42):
    # Pass 1: standard greedy
    assignments, obj, _ = greedy_optimize(problem, seed=seed)
    obj_eval, is_feas, details = problem.evaluate(assignments)
    if is_feas or problem.emissions_cap_t is None:
        return assignments, obj_eval, is_feas, details

    current_plan = [dict(a) for a in assignments]
    max_iter = 50
    for _ in range(max_iter):
        obj_eval, is_feas, details = problem.evaluate(current_plan)
        if is_feas or details["co2_wtw_t"] <= problem.emissions_cap_t:
            break
        
        # Find best single route upgrade with lowest marginal abatement cost ($/t CO2)
        best_upgrade = None
        best_mac = float("inf")
        
        for r_idx, curr_item in enumerate(current_plan):
            route = problem.routes[r_idx]
            curr_vessel = next(v for v in problem.vessels if v["id"] == curr_item["vessel_id"])
            curr_speed = curr_item["speed_kn"]
            curr_fuel = next(f for f in problem.fuels if f["id"] == curr_item["fuel_id"])
            
            # Current route emissions and cost
            curr_cache = problem._lookup_cache.get((route["id"], curr_vessel["id"], curr_speed, curr_fuel["id"]))
            if not curr_cache:
                continue
            curr_co2 = curr_cache["co2_wtw_t"]
            curr_cost = curr_cache["fuel_cost"]
            
            # Test alternate fuels and speeds for this route and vessel
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
                    if delta_co2 > 1e-3:  # achieves CO2 reduction
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
                                "fuel_type": fuel["name"]
                            })
        
        if best_upgrade is None:
            break
        
        r_idx, new_choice = best_upgrade
        current_plan[r_idx] = new_choice

    final_obj, final_feas, final_details = problem.evaluate(current_plan)
    return current_plan, final_obj, final_feas, final_details

if __name__ == "__main__":
    engine = db.get_engine()
    with db.Session(engine) as session:
        sc_ids = generate_benchmark_scenarios(session)

    print("=== STAGE 2A: GREEDY (PURE) vs GREEDY (WITH DECARB PASS) ===")
    for inst, sc_id in sc_ids.items():
        prob = FleetOptimizationProblem(scenario_id=sc_id)
        
        # Pure Greedy
        plan_pure, obj_pure, _ = greedy_optimize(prob, seed=42)
        eval_pure, feas_pure, det_pure = prob.evaluate(plan_pure)
        
        # Greedy + Decarb Pass
        plan_dec, obj_dec, feas_dec, det_dec = greedy_decarb_optimize(prob, seed=42)
        
        # MILP
        milp_plan, milp_obj, _ = milp_optimize(prob)
        
        gap_pure = ((eval_pure - milp_obj) / milp_obj) * 100 if feas_pure else None
        gap_dec = ((obj_dec - milp_obj) / milp_obj) * 100 if feas_dec else None
        
        print(f"\n--- {inst} Instance (Cap: {prob.emissions_cap_t:,.0f} t) ---")
        print(f"  MILP (Exact):          Obj=${milp_obj:12,.2f} | Feasible={True} | CO2={prob.evaluate(milp_plan)[2]['co2_wtw_t']:8,.1f}t | Gap= 0.00%")
        if feas_pure:
            print(f"  Greedy (Pure):         Obj=${eval_pure:12,.2f} | Feasible={feas_pure} | CO2={det_pure['co2_wtw_t']:8,.1f}t | Gap={gap_pure:6.2f}%")
        else:
            print(f"  Greedy (Pure):         Failed: no feasible plan (CO2={det_pure['co2_wtw_t']:,.1f}t > Cap {prob.emissions_cap_t:,.0f}t)")
        
        if feas_dec:
            print(f"  Greedy (+Decarb Pass): Obj=${obj_dec:12,.2f} | Feasible={feas_dec} | CO2={det_dec['co2_wtw_t']:8,.1f}t | Gap={gap_dec:6.2f}%")
        else:
            print(f"  Greedy (+Decarb Pass): Failed: no feasible plan")
