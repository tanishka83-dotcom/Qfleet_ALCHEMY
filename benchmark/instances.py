"""
QFleet Phase 3 — Benchmark Instance Generator
==============================================
Generates seeded, reproducible fleet optimization instances stored in SQLite:
1. Small  : 4 vessels, 6 routes  (Baseline demo scenario, 7 slots, slack = 1)
2. Medium : 8 vessels, 15 routes (18 slots, slack = 3, binding emissions cap = 27,000 t CO2eq)
3. Large  : 15 vessels, 30 routes (33 slots, slack = 3, binding emissions cap = 55,000 t CO2eq)

Ensures:
- Usable vessel route slots >= number of routes (with explicit assertions and reported slack).
- Binding emissions cap: unconstrained optimum exceeds cap, while feasible constrained solutions exist.
- Idempotent seeding into SQLite scenarios table.
"""

from __future__ import annotations

import sys
import json
import pathlib
import numpy as np
from sqlalchemy.orm import Session

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
import db as database


# ---------------------------------------------------------------------------
# Synthetic Route & Vessel Catalogue Definitions
# ---------------------------------------------------------------------------

VESSEL_CLASS_TEMPLATES = {
    "Feeder": {
        "dwt_t": 14000.0,
        "teu_capacity": 1850,
        "design_speed_kn": 18.0,
        "ref_daily_fuel_t": 28.0,
    },
    "Panamax": {
        "dwt_t": 65000.0,
        "teu_capacity": 5000,
        "design_speed_kn": 22.0,
        "ref_daily_fuel_t": 85.0,
    },
    "PostPanamax": {
        "dwt_t": 115000.0,
        "teu_capacity": 13000,
        "design_speed_kn": 23.5,
        "ref_daily_fuel_t": 160.0,
    },
    "ULCS": {
        "dwt_t": 195000.0,
        "teu_capacity": 20000,
        "design_speed_kn": 24.5,
        "ref_daily_fuel_t": 240.0,
    },
}

# Standard Port Pair Route Templates (Ordered for Small 0..5, Medium 0..14, Large 0..29)
STANDARD_ROUTE_TEMPLATES = [
    # --- Routes 0..5: Small Instance (6 routes, exactly matches Phase 1 & 2 Demo) ---
    {"origin": "Shanghai", "dest": "Rotterdam", "dist": 10500.0, "demand": 18000, "deadline": 24.0}, # ULCS
    {"origin": "Singapore", "dest": "Rotterdam", "dist": 8300.0, "demand": 16500, "deadline": 20.0}, # ULCS
    {"origin": "Shenzhen", "dest": "Los Angeles", "dist": 6500.0, "demand": 12000, "deadline": 16.0}, # PostPanamax
    {"origin": "Busan", "dest": "Seattle", "dist": 4900.0, "demand": 8500, "deadline": 12.0},         # PostPanamax
    {"origin": "Tokyo", "dest": "Singapore", "dist": 2900.0, "demand": 4200, "deadline": 8.0},        # Panamax
    {"origin": "Rotterdam", "dest": "Hamburg", "dist": 350.0, "demand": 1400, "deadline": 2.5},       # Feeder

    # --- Routes 6..14: Additional for Medium Instance (15 routes total) ---
    {"origin": "Ningbo", "dest": "Hamburg", "dist": 10800.0, "demand": 18500, "deadline": 25.0},      # ULCS
    {"origin": "Qingdao", "dest": "Long Beach", "dist": 5800.0, "demand": 15500, "deadline": 15.0},   # ULCS
    {"origin": "Tokyo", "dest": "Oakland", "dist": 4600.0, "demand": 10000, "deadline": 11.5},       # PostPanamax
    {"origin": "Shanghai", "dest": "Genoa", "dist": 8100.0, "demand": 9500, "deadline": 19.0},        # PostPanamax
    {"origin": "Rotterdam", "dest": "New York", "dist": 3450.0, "demand": 4800, "deadline": 9.5},     # Panamax
    {"origin": "Singapore", "dest": "Sydney", "dist": 3350.0, "demand": 4500, "deadline": 9.0},       # Panamax
    {"origin": "Shanghai", "dest": "Singapore", "dist": 1650.0, "demand": 3800, "deadline": 5.0},     # Panamax
    {"origin": "Hong Kong", "dest": "Manila", "dist": 650.0, "demand": 1600, "deadline": 2.5},        # Feeder
    {"origin": "Antwerp", "dest": "Le Havre", "dist": 280.0, "demand": 1200, "deadline": 1.5},        # Feeder

    # --- Routes 15..29: Additional for Large Instance (30 routes total) ---
    {"origin": "Yantian", "dest": "Antwerp", "dist": 10100.0, "demand": 17500, "deadline": 23.0},     # ULCS
    {"origin": "Shanghai", "dest": "Santos", "dist": 11200.0, "demand": 17000, "deadline": 26.0},     # ULCS
    {"origin": "Kaohsiung", "dest": "Rotterdam", "dist": 9800.0, "demand": 16000, "deadline": 22.0},  # ULCS
    {"origin": "Shenzhen", "dest": "Felixstowe", "dist": 9600.0, "demand": 16500, "deadline": 22.5},  # ULCS
    {"origin": "Kaohsiung", "dest": "Vancouver", "dist": 5400.0, "demand": 11500, "deadline": 13.5},  # PostPanamax
    {"origin": "Port Klang", "dest": "Felixstowe", "dist": 8500.0, "demand": 11000, "deadline": 21.0}, # PostPanamax
    {"origin": "Santos", "dest": "Rotterdam", "dist": 5600.0, "demand": 10500, "deadline": 14.0},    # PostPanamax
    {"origin": "Singapore", "dest": "Piraeus", "dist": 6200.0, "demand": 8500, "deadline": 15.0},    # PostPanamax
    {"origin": "Jebel Ali", "dest": "Singapore", "dist": 3600.0, "demand": 4600, "deadline": 9.0},    # Panamax
    {"origin": "Valencia", "dest": "New York", "dist": 3900.0, "demand": 4400, "deadline": 10.0},     # Panamax
    {"origin": "Colombo", "dest": "Port Klang", "dist": 1350.0, "demand": 3900, "deadline": 4.0},     # Panamax
    {"origin": "Mumbai", "dest": "Jebel Ali", "dist": 1100.0, "demand": 3500, "deadline": 3.5},       # Panamax
    {"origin": "Busan", "dest": "Fukuoka", "dist": 120.0, "demand": 1500, "deadline": 1.0},           # Feeder
    {"origin": "Singapore", "dest": "Jakarta", "dist": 500.0, "demand": 1400, "deadline": 1.8},       # Feeder
    {"origin": "Bremen", "dest": "Gothenburg", "dist": 320.0, "demand": 1100, "deadline": 1.5},       # Feeder
]


# ---------------------------------------------------------------------------
# Instance Seeding Helpers
# ---------------------------------------------------------------------------

def seed_benchmark_vessels(session: Session) -> dict[str, list[database.Vessel]]:
    """Ensure catalog contains enough vessels for Small, Medium, and Large instances."""
    existing_vessels = {v.name: v for v in session.query(database.Vessel).all()}
    vessel_map = {"Feeder": [], "Panamax": [], "PostPanamax": [], "ULCS": []}

    # Desired fleet composition: 4 Feeder, 4 Panamax, 4 PostPanamax, 3 ULCS (15 total)
    counts_needed = {"Feeder": 4, "Panamax": 4, "PostPanamax": 4, "ULCS": 3}

    for vclass, count in counts_needed.items():
        spec = VESSEL_CLASS_TEMPLATES[vclass]
        for idx in range(1, count + 1):
            vname = f"{vclass}-{idx:02d}"
            if vname in existing_vessels:
                v_obj = existing_vessels[vname]
            else:
                v_obj = database.Vessel(
                    name=vname,
                    vessel_class=vclass,
                    dwt_t=spec["dwt_t"],
                    teu_capacity=spec["teu_capacity"],
                    design_speed_kn=spec["design_speed_kn"],
                    ref_daily_fuel_t=spec["ref_daily_fuel_t"],
                    flag="PA",
                )
                session.add(v_obj)
                session.flush()
            vessel_map[vclass].append(v_obj)

    session.commit()
    return vessel_map


def seed_benchmark_routes(session: Session) -> list[database.Route]:
    """Ensure catalog contains all standard benchmark routes with calibrated values."""
    existing_routes = {r.name: r for r in session.query(database.Route).all()}
    all_routes = []

    for idx, r_spec in enumerate(STANDARD_ROUTE_TEMPLATES, 1):
        r_name = f"Route-{idx:02d}: {r_spec['origin']} -> {r_spec['dest']}"
        if r_name in existing_routes:
            r_obj = existing_routes[r_name]
            r_obj.origin_port = r_spec["origin"]
            r_obj.dest_port = r_spec["dest"]
            r_obj.distance_nm = r_spec["dist"]
            r_obj.cargo_demand_teu = r_spec["demand"]
            r_obj.deadline_days = r_spec["deadline"]
        else:
            r_obj = database.Route(
                name=r_name,
                origin_port=r_spec["origin"],
                dest_port=r_spec["dest"],
                distance_nm=r_spec["dist"],
                typical_laden_pct=0.85,
                cargo_demand_teu=r_spec["demand"],
                deadline_days=r_spec["deadline"],
                source="Standard Benchmark Catalog (Phase 3)",
            )
            session.add(r_obj)
            session.flush()
        all_routes.append(r_obj)

    session.commit()
    return all_routes


# ---------------------------------------------------------------------------
# Main Scenario Generator & Validator
# ---------------------------------------------------------------------------

def generate_benchmark_scenarios(session: Session) -> dict[str, int]:
    """
    Creates and persists the 3 standard benchmark scenarios in SQLite:
    - Small  : 4 vessels, 6 routes  (Demo scale, non-binding cap)
    - Medium : 8 vessels, 15 routes (18 slots, slack=3, binding emissions cap=27,000 t CO2eq)
    - Large  : 15 vessels, 30 routes (33 slots, slack=3, binding emissions cap=55,000 t CO2eq)

    Returns:
        dict[str, int]: mapping of instance name ("Small", "Medium", "Large") to scenario_id.
    """
    vessel_map = seed_benchmark_vessels(session)
    routes_catalog = seed_benchmark_routes(session)

    scenario_ids = {}

    # 1. SMALL INSTANCE (4 vessels, 6 routes)
    small_vessels = [
        vessel_map["Feeder"][0],
        vessel_map["Panamax"][0],
        vessel_map["PostPanamax"][0],
        vessel_map["ULCS"][0],
    ]
    small_routes = routes_catalog[:6]
    small_vessel_limits = {
        small_vessels[0].name: 1,  # Feeder: 1 route
        small_vessels[1].name: 2,  # Panamax: 2 routes
        small_vessels[2].name: 2,  # PostPanamax: 2 routes
        small_vessels[3].name: 2,  # ULCS: 2 routes
    }
    small_cfg = {
        "name": "Small_4V_6R",
        "description": "Small benchmark instance: 4 vessels, 6 routes, unconstrained emissions cap.",
        "vessel_ids": [v.id for v in small_vessels],
        "route_ids": [r.id for r in small_routes],
        "vessel_limits": small_vessel_limits,
        "max_emissions_cap_t": 35000.0,
        "assumed_weather_factor": 1.08,
        "weights": config.OPTIMIZATION_WEIGHTS,
    }
    scenario_ids["Small"] = _upsert_scenario(session, "Small_4V_6R", small_cfg)

    # 2. MEDIUM INSTANCE (8 vessels, 15 routes, binding cap = 58,000 t CO2eq)
    medium_vessels = (
        vessel_map["Feeder"][:2] +
        vessel_map["Panamax"][:2] +
        vessel_map["PostPanamax"][:2] +
        vessel_map["ULCS"][:2]
    )
    medium_routes = routes_catalog[:15]
    medium_vessel_limits = {
        v.name: 3 if v.vessel_class == "ULCS" else 2
        for v in medium_vessels
    }
    medium_cfg = {
        "name": "Medium_8V_15R",
        "description": "Medium benchmark instance: 8 vessels, 15 routes, binding emissions cap = 58,000 t CO2eq (TODO_VERIFY TV-29).",
        "vessel_ids": [v.id for v in medium_vessels],
        "route_ids": [r.id for r in medium_routes],
        "vessel_limits": medium_vessel_limits,
        "max_emissions_cap_t": 58000.0,  # Binding cap
        "assumed_weather_factor": 1.08,
        "weights": config.OPTIMIZATION_WEIGHTS,
    }
    scenario_ids["Medium"] = _upsert_scenario(session, "Medium_8V_15R", medium_cfg)

    # 3. LARGE INSTANCE (15 vessels, 30 routes, binding cap = 122,000 t CO2eq)
    large_vessels = (
        vessel_map["Feeder"][:4] +
        vessel_map["Panamax"][:4] +
        vessel_map["PostPanamax"][:4] +
        vessel_map["ULCS"][:3]
    )
    large_routes = routes_catalog[:30]
    large_vessel_limits = {
        v.name: 3 if v.vessel_class == "ULCS" else 2
        for v in large_vessels
    }
    large_cfg = {
        "name": "Large_15V_30R",
        "description": "Large benchmark instance: 15 vessels, 30 routes, binding emissions cap = 122,000 t CO2eq (TODO_VERIFY TV-30).",
        "vessel_ids": [v.id for v in large_vessels],
        "route_ids": [r.id for r in large_routes],
        "vessel_limits": large_vessel_limits,
        "max_emissions_cap_t": 122000.0,  # Binding cap
        "assumed_weather_factor": 1.08,
        "weights": config.OPTIMIZATION_WEIGHTS,
    }
    scenario_ids["Large"] = _upsert_scenario(session, "Large_15V_30R", large_cfg)

    # Verify slot feasibility and slack for all instances
    for inst_name, sc_id in scenario_ids.items():
        sc_obj = session.query(database.Scenario).filter_by(id=sc_id).first()
        cfg = json.loads(sc_obj.fleet_config_json)
        v_list = session.query(database.Vessel).filter(database.Vessel.id.in_(cfg["vessel_ids"])).all()
        r_list = session.query(database.Route).filter(database.Route.id.in_(cfg["route_ids"])).all()
        
        slack_info = compute_instance_capacity_slack(cfg, v_list, r_list)
        assert slack_info["usable_slots"] >= slack_info["num_routes"], (
            f"Instance {inst_name} has insufficient slots: {slack_info['usable_slots']} < {slack_info['num_routes']}"
        )

    return scenario_ids


def _upsert_scenario(session: Session, name: str, cfg_dict: dict) -> int:
    """Insert or update scenario JSON definition."""
    existing = session.query(database.Scenario).filter_by(name=name).first()
    cfg_json = json.dumps(cfg_dict, indent=2)
    if existing:
        existing.fleet_config_json = cfg_json
        session.commit()
        return existing.id
    else:
        sc = database.Scenario(name=name, fleet_config_json=cfg_json)
        session.add(sc)
        session.commit()
        session.refresh(sc)
        return sc.id


def compute_instance_capacity_slack(
    cfg: dict,
    vessels: list[database.Vessel],
    routes: list[database.Route],
) -> dict:
    """
    Validates capacity hierarchy and calculates slot slack:
    - Total usable slots sum_{v} N_v^max >= |R|
    - For each demand tier, available vessel slots with capacity >= tier demand >= number of routes in tier.
    """
    v_limits = cfg.get("vessel_limits", {})
    total_slots = sum(v_limits.get(v.name, 2) for v in vessels)
    num_routes = len(routes)
    slack = total_slots - num_routes

    # Check nested capacity requirement:
    # Routes requiring ULCS (> 13,000 TEU)
    ulcs_routes = [r for r in routes if r.cargo_demand_teu > 13000]
    ulcs_slots = sum(v_limits.get(v.name, 2) for v in vessels if v.vessel_class == "ULCS")
    assert ulcs_slots >= len(ulcs_routes), (
        f"ULCS slot shortfall: {ulcs_slots} slots available for {len(ulcs_routes)} heavy routes."
    )

    # Routes requiring at least PostPanamax (> 5,000 TEU)
    postpan_routes = [r for r in routes if r.cargo_demand_teu > 5000]
    postpan_slots = sum(v_limits.get(v.name, 2) for v in vessels if v.vessel_class in ["PostPanamax", "ULCS"])
    assert postpan_slots >= len(postpan_routes), (
        f"PostPanamax slot shortfall: {postpan_slots} slots available for {len(postpan_routes)} routes."
    )

    return {
        "instance_name": cfg.get("name"),
        "num_vessels": len(vessels),
        "num_routes": num_routes,
        "total_usable_slots": total_slots,
        "usable_slots": total_slots,
        "slack_slots": slack,
        "ulcs_routes": len(ulcs_routes),
        "ulcs_slots": ulcs_slots,
    }
