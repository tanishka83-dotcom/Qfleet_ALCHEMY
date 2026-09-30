"""
Stage 2 (Phase 5) Comprehensive Benchmark Runner
=================================================
Tests:
- 2b: Route Alternatives (3 candidate routes per voyage)
- 2c: In-Port Shore Power
- 2d: Mega Scale (100 voyages, 30 vessels) & Emissions Cap Tightness Sweep
"""

import time
import numpy as np
import pandas as pd

import config
from optimizer.problem_v2 import FleetOptimizationProblemV2
from optimizer.baselines_v2 import milp_optimize_v2

def create_sample_fleet_v2(num_routes: int = 6, num_vessels: int = 4, cap_factor: float = 0.9):
    # Port pairs pool
    ports = [
        ("Shanghai", "Rotterdam", 10500.0, 18000, 24.0, 48.0),
        ("Singapore", "Rotterdam", 8300.0, 16500, 20.0, 42.0),
        ("Shenzhen", "Los Angeles", 6500.0, 12000, 16.0, 36.0),
        ("Busan", "Seattle", 4900.0, 8500, 12.0, 30.0),
        ("Tokyo", "Singapore", 2900.0, 4200, 8.0, 24.0),
        ("Rotterdam", "Hamburg", 350.0, 1400, 2.5, 18.0),
        ("Shanghai", "Hamburg", 10800.0, 15000, 25.0, 48.0),
        ("Ningbo", "Long Beach", 5800.0, 11000, 15.0, 36.0),
        ("Kaohsiung", "Antwerp", 9200.0, 13500, 22.0, 40.0),
        ("Port Klang", "Felixstowe", 8100.0, 14000, 20.0, 38.0),
    ]

    routes = []
    for i in range(num_routes):
        p = ports[i % len(ports)]
        routes.append({
            "id": i + 1,
            "name": f"Voyage-{i+1:03d} ({p[0]}→{p[1]})",
            "origin_port": p[0],
            "dest_port": p[1],
            "distance_nm": p[2] * (1.0 + 0.05 * (i // len(ports))),
            "cargo_demand_teu": p[3],
            "deadline_days": p[4] * (1.0 + 0.05 * (i // len(ports))),
            "berth_hours": p[5],
            "route_options": [
                {"option_id": 0, "name": "Direct / Standard", "dist_factor": 1.0, "weather_factor": 1.0, "eca_fraction": 0.15},
                {"option_id": 1, "name": "Weather-Optimized (Calm)", "dist_factor": 1.04, "weather_factor": 0.92, "eca_fraction": 0.15},
                {"option_id": 2, "name": "ECA-Minimizing", "dist_factor": 1.06, "weather_factor": 1.00, "eca_fraction": 0.05},
            ]
        })

    # Vessel catalogue pool
    vessel_templates = [
        {"name": "ULCS Alpha", "vessel_class": "ULCS", "capacity_teu": 20000, "design_speed_kn": 24.5},
        {"name": "ULCS Beta", "vessel_class": "ULCS", "capacity_teu": 20000, "design_speed_kn": 24.5},
        {"name": "PostPanamax 1", "vessel_class": "PostPanamax", "capacity_teu": 13000, "design_speed_kn": 23.5},
        {"name": "PostPanamax 2", "vessel_class": "PostPanamax", "capacity_teu": 13000, "design_speed_kn": 23.5},
        {"name": "Panamax Ocean", "vessel_class": "Panamax", "capacity_teu": 5000, "design_speed_kn": 22.0},
        {"name": "Panamax Star", "vessel_class": "Panamax", "capacity_teu": 5000, "design_speed_kn": 22.0},
        {"name": "Feeder Coastal", "vessel_class": "Feeder", "capacity_teu": 1850, "design_speed_kn": 18.0},
        {"name": "Feeder Express", "vessel_class": "Feeder", "capacity_teu": 1850, "design_speed_kn": 18.0},
    ]

    vessels = []
    quota = max(2, int(np.ceil(num_routes / num_vessels) + 1))
    vessel_limits = {}
    for i in range(num_vessels):
        vt = vessel_templates[i % len(vessel_templates)]
        v_name = f"{vt['name']}-{i+1}"
        vessels.append({
            "id": i + 1,
            "name": v_name,
            "vessel_class": vt["vessel_class"],
            "capacity_teu": vt["capacity_teu"],
            "design_speed_kn": vt["design_speed_kn"],
        })
        vessel_limits[v_name] = quota

    fuels = [
        {"id": 1, "name": "HFO", "price_usd_per_tonne": 510.0, "lhv_mj_per_kg": 40.2, "co2_wtw_g_per_mj": 86.2},
        {"id": 2, "name": "VLSFO", "price_usd_per_tonne": 620.0, "lhv_mj_per_kg": 40.5, "co2_wtw_g_per_mj": 86.2},
        {"id": 3, "name": "MGO", "price_usd_per_tonne": 820.0, "lhv_mj_per_kg": 42.7, "co2_wtw_g_per_mj": 89.0},
        {"id": 4, "name": "LNG", "price_usd_per_tonne": 750.0, "lhv_mj_per_kg": 48.0, "co2_wtw_g_per_mj": 75.7},
        {"id": 5, "name": "METHANOL", "price_usd_per_tonne": 890.0, "lhv_mj_per_kg": 19.9, "co2_wtw_g_per_mj": 105.8},
    ]

    # Pre-evaluate unconstrained to set binding cap
    prob_unconstrained = FleetOptimizationProblemV2(
        routes=routes,
        vessels=vessels,
        fuels=fuels,
        vessel_limits=vessel_limits,
        emissions_cap_t=None,
    )
    _, _, det_unconstrained = milp_optimize_v2(prob_unconstrained)
    unconstrained_co2 = det_unconstrained["co2_t"]
    cap = unconstrained_co2 * cap_factor

    prob = FleetOptimizationProblemV2(
        routes=routes,
        vessels=vessels,
        fuels=fuels,
        vessel_limits=vessel_limits,
        emissions_cap_t=cap,
        instance_name=f"Instance_{num_vessels}V_{num_routes}R",
    )
    return prob, unconstrained_co2, cap


def run_stage2_benchmarks():
    print("=================================================================")
    print("STAGE 2: PHASE 5 BENCHMARK RESULTS (Route Options & Shore Power)")
    print("=================================================================\n")

    instances = [
        ("Small_v2", 6, 4, 0.95),
        ("Medium_v2", 15, 8, 0.92),
        ("Large_v2", 30, 15, 0.90),
        ("Mega_30V_100R", 100, 30, 0.88),
    ]

    results = []
    for name, n_routes, n_vessels, cap_f in instances:
        print(f"Running {name} ({n_vessels} vessels, {n_routes} routes)...")
        prob, uncons_co2, cap = create_sample_fleet_v2(num_routes=n_routes, num_vessels=n_vessels, cap_factor=cap_f)
        
        sol, obj, stats = milp_optimize_v2(prob)
        shore_count = sum(1 for s in sol if s.get("used_shore_power", False))
        weather_opt_count = sum(1 for s in sol if s.get("route_option_id") == 1)
        eca_opt_count = sum(1 for s in sol if s.get("route_option_id") == 2)
        
        results.append({
            "Instance": name,
            "Voyages": n_routes,
            "Vessels": n_vessels,
            "Unconstrained CO2 (t)": round(uncons_co2, 1),
            "Cap (t)": round(cap, 1),
            "MILP Status": stats["status"],
            "Objective ($)": round(obj, 2),
            "CO2 Result (t)": round(stats["co2_t"], 1),
            "Shore Power Calls": f"{shore_count}/{n_routes}",
            "Weather/ECA Routes Picked": f"{weather_opt_count + eca_opt_count}/{n_routes}",
            "Solve Time (s)": round(stats["runtime_s"], 4),
        })

    df_res = pd.DataFrame(results)
    print("\n=== Phase 5 (V2) Scale & Features Benchmark Table ===")
    print(df_res.to_string(index=False))

    # Cap Tightness Sweep on Large_v2
    print("\n=== Emissions Cap Tightness Sweep (Large_v2: 30 Routes) ===")
    sweep_results = []
    prob_base, uncons_co2, _ = create_sample_fleet_v2(num_routes=30, num_vessels=15, cap_factor=1.0)
    for reduction_pct in [0, 5, 10, 15, 20]:
        cap_val = uncons_co2 * (1.0 - reduction_pct / 100.0)
        prob_sweep = FleetOptimizationProblemV2(
            routes=prob_base.routes,
            vessels=prob_base.vessels,
            fuels=prob_base.fuels,
            vessel_limits=prob_base.vessel_limits,
            emissions_cap_t=cap_val,
        )
        sol, obj, stats = milp_optimize_v2(prob_sweep)
        sweep_results.append({
            "Decarbonization Target": f"-{reduction_pct}%",
            "Cap (t)": round(cap_val, 1),
            "Feasible": stats["is_feasible"],
            "Cost ($)": round(stats["cost_usd"], 2),
            "Total Objective ($)": round(obj, 2),
            "Runtime (s)": round(stats["runtime_s"], 4),
        })
    df_sweep = pd.DataFrame(sweep_results)
    print(df_sweep.to_string(index=False))

if __name__ == "__main__":
    run_stage2_benchmarks()
