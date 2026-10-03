"""
QFleet Phase 5 — Benchmark Instance Generator for V2
===================================================
Generates in-memory representations of:
1. Small_v2 (4 vessels, 6 routes)
2. Medium_v2 (8 vessels, 15 routes)
3. Large_v2 (15 vessels, 30 routes)
4. Mega_30V_100R (30 vessels, 100 routes)

Includes:
- 3 candidate route options per route: Direct (opt 0), Weather-Optimized (opt 1), ECA-Minimizing (opt 2)
- Port shore power at-berth profile
- Vessel limits per vessel: max(2, ceil(num_routes / num_vessels) + 1)
- Emission caps: Small_v2 95%, Medium_v2 92%, Large_v2 90%, Mega 88% of unconstrained emissions
"""

from __future__ import annotations

import sys
import math
import pathlib
import numpy as np

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from benchmark.instances import STANDARD_ROUTE_TEMPLATES, VESSEL_CLASS_TEMPLATES
from optimizer.problem_v2 import FleetOptimizationProblemV2
from optimizer.baselines_v2 import milp_optimize_v2

# Standard Fuels
FUELS_V2 = [
    {"id": 1, "name": "HFO", "price_usd_per_tonne": 580.0, "lhv_mj_per_kg": 40.5, "co2_wtw_g_per_mj": 91.0},
    {"id": 2, "name": "VLSFO", "price_usd_per_tonne": 650.0, "lhv_mj_per_kg": 41.0, "co2_wtw_g_per_mj": 89.0},
    {"id": 3, "name": "MGO", "price_usd_per_tonne": 820.0, "lhv_mj_per_kg": 42.7, "co2_wtw_g_per_mj": 87.0},
    {"id": 4, "name": "LNG", "price_usd_per_tonne": 720.0, "lhv_mj_per_kg": 49.0, "co2_wtw_g_per_mj": 68.0},
    {"id": 5, "name": "METHANOL", "price_usd_per_tonne": 920.0, "lhv_mj_per_kg": 19.9, "co2_wtw_g_per_mj": 32.0},
]


def _build_route_options(origin: str, dest: str, dist: float) -> list[dict]:
    """Build 3 candidate route options for each voyage."""
    return [
        {"option_id": 0, "name": "Direct", "dist_factor": 1.00, "weather_factor": 1.00, "eca_fraction": 0.15},
        {"option_id": 1, "name": "Weather-Optimized", "dist_factor": 1.04, "weather_factor": 0.92, "eca_fraction": 0.15},
        {"option_id": 2, "name": "ECA-Minimizing", "dist_factor": 1.06, "weather_factor": 1.00, "eca_fraction": 0.05},
    ]


def build_v2_instance(instance_name: str) -> FleetOptimizationProblemV2:
    """Instantiate a FleetOptimizationProblemV2 for Small_v2, Medium_v2, Large_v2, or Mega_30V_100R."""
    if instance_name == "Small_v2":
        num_vessels = 4
        num_routes = 6
        cap_fraction = 0.95
    elif instance_name == "Medium_v2":
        num_vessels = 8
        num_routes = 15
        cap_fraction = 0.92
    elif instance_name == "Large_v2":
        num_vessels = 15
        num_routes = 30
        cap_fraction = 0.90
    elif instance_name == "Mega_30V_100R":
        num_vessels = 30
        num_routes = 100
        cap_fraction = 0.88
    else:
        raise ValueError(f"Unknown instance: {instance_name}")

    # Generate vessels
    vessels = []
    classes = ["Feeder", "Panamax", "PostPanamax", "ULCS"]
    v_id = 1
    for i in range(num_vessels):
        vc = classes[i % len(classes)]
        v_tmpl = VESSEL_CLASS_TEMPLATES[vc]
        vessels.append({
            "id": v_id,
            "name": f"{vc}-{v_id:02d}",
            "vessel_class": vc,
            "capacity_teu": v_tmpl["teu_capacity"],
            "teu_capacity": v_tmpl["teu_capacity"],
            "design_speed_kn": v_tmpl["design_speed_kn"],
            "ref_daily_fuel_t": v_tmpl["ref_daily_fuel_t"],
        })
        v_id += 1

    # Generate routes
    routes = []
    r_id = 1
    for i in range(num_routes):
        tmpl = STANDARD_ROUTE_TEMPLATES[i % len(STANDARD_ROUTE_TEMPLATES)]
        routes.append({
            "id": r_id,
            "name": f"Route-{r_id:03d}: {tmpl['origin']} -> {tmpl['dest']}",
            "origin_port": tmpl["origin"],
            "dest_port": tmpl["dest"],
            "distance_nm": float(tmpl["dist"]),
            "cargo_demand_teu": int(tmpl["demand"]),
            "deadline_days": float(tmpl["deadline"]),
            "berth_hours": 24.0 if tmpl["demand"] < 5000 else (36.0 if tmpl["demand"] < 13000 else 48.0),
            "route_options": _build_route_options(tmpl["origin"], tmpl["dest"], tmpl["dist"]),
        })
        r_id += 1

    # Quotas: max(2, ceil(num_routes / num_vessels) + 1)
    quota = max(2, math.ceil(num_routes / num_vessels) + 1)
    vessel_limits = {v["name"]: quota for v in vessels}

    # Step 1: Compute unconstrained optimum using MILP to set the emissions cap
    unconstrained_prob = FleetOptimizationProblemV2(
        routes=routes,
        vessels=vessels,
        fuels=FUELS_V2,
        weights=config.OPTIMIZATION_WEIGHTS.copy(),
        emissions_cap_t=None,
        vessel_limits=vessel_limits,
        instance_name=instance_name,
    )
    sol_u, obj_u, det_u = milp_optimize_v2(unconstrained_prob)
    unconstrained_co2 = det_u.get("co2_t", 50000.0)

    # Set emissions cap according to rule
    emissions_cap_t = round(unconstrained_co2 * cap_fraction, 1)

    # Return constrained problem
    constrained_prob = FleetOptimizationProblemV2(
        routes=routes,
        vessels=vessels,
        fuels=FUELS_V2,
        weights=config.OPTIMIZATION_WEIGHTS.copy(),
        emissions_cap_t=emissions_cap_t,
        vessel_limits=vessel_limits,
        instance_name=instance_name,
    )
    return constrained_prob
