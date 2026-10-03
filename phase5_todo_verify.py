"""Additional Phase 5 assumptions kept outside protected config.py."""

PHASE5_TODO_VERIFY_ITEMS = [
    "TV-33 Phase 5 emission-cap rules: Small_v2 95%, Medium_v2 92%, Large_v2 90%, and Mega 88% of unconstrained emissions; verify rationale and feasibility.",
    "TV-34 Phase 5 vessel quotas: max(2, ceil(route_count / vessel_count) + 1) per vessel; verify operational availability basis.",
    "TV-35 Phase 5 objective weights w1/w2/w3; verify units, interpretation, and whether w2 is a shadow carbon price.",
    "TV-36 ECA-minimizing route option distance factor and ECA fraction; verify route-specific geometry and regulatory-zone data.",
    "TV-37 Berth-hour assumptions by route; verify against port call and vessel turnaround records.",
    "TV-38 Port shore-power availability flags; verify port and terminal-specific infrastructure availability.",
    "TV-39 Port grid emissions factors; verify year, location, and electricity accounting boundary.",
    "TV-40 Port electricity tariffs; verify currency, tariff class, demand charges, and effective date.",
    "TV-41 UNVERIFIED: results/phase5_v2_benchmark.csv generated prior to MILP regression fix; verify against updated runs.",
    "TV-42 UNVERIFIED: results/phase5_v2_cap_sweep.csv generated prior to MILP regression fix; verify against updated runs.",
]


def register_phase5_todo_items(config_module) -> list[str]:
    """Merge the Phase 5 assumptions into the shared API-visible registry."""
    registered = set(config_module.TODO_VERIFY_ITEMS)
    config_module.TODO_VERIFY_ITEMS.extend(
        item for item in PHASE5_TODO_VERIFY_ITEMS if item not in registered
    )
    return config_module.TODO_VERIFY_ITEMS
