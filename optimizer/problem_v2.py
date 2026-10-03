"""
QFleet Phase 5 — Version 2 Fleet Optimization Problem Formulation
==================================================================
Extends Phase 2 problem formulation without modifying problem.py:
1. Decision 4-tuple: (vessel, speed, fuel, route_option)
   - 2-3 candidate route options per voyage (distance, weather_factor, ECA fraction)
2. Shore power (cold ironing) option at berth:
   - Evaluates at-berth energy demand, port electricity tariff, and grid carbon intensity
   - Compares shore power vs auxiliary engine fuel consumption
3. Mega-scale instance support (100+ voyages, 30 vessels)
4. Full isolation: never overwrites or alters existing scenarios or DB records.
"""

from __future__ import annotations

import sys
import time
import json
import pathlib
import numpy as np
import pandas as pd
from typing import Any

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from optimizer.problem import get_surrogate_model

# ---------------------------------------------------------------------------
# Route Options & Shore Power Catalogues (Phase 5 Version 2)
# ---------------------------------------------------------------------------

# Port Shore Power Profiles (Tariff $/kWh, Grid Factor g CO2/kWh, Availability)
PORT_SHORE_PROFILES: dict[str, dict[str, Any]] = {
    "Rotterdam":   {"available": True,  "tariff_usd_kwh": 0.18, "grid_co2_g_kwh": 350.0, "source": "TODO_VERIFY TV-31: Port of Rotterdam cold ironing tariff/grid proxy"},
    "Hamburg":     {"available": True,  "tariff_usd_kwh": 0.22, "grid_co2_g_kwh": 320.0, "source": "TODO_VERIFY TV-31: Hamburg shore power rate proxy"},
    "Los Angeles": {"available": True,  "tariff_usd_kwh": 0.20, "grid_co2_g_kwh": 280.0, "source": "TODO_VERIFY TV-31: Port of LA CARB shore power mandate proxy"},
    "Singapore":   {"available": False, "tariff_usd_kwh": 0.16, "grid_co2_g_kwh": 400.0, "source": "TODO_VERIFY TV-31: Singapore shore power pilot proxy"},
    "Shanghai":    {"available": True,  "tariff_usd_kwh": 0.15, "grid_co2_g_kwh": 580.0, "source": "TODO_VERIFY TV-31: Shanghai port electricity tariff proxy"},
    "Shenzhen":    {"available": True,  "tariff_usd_kwh": 0.15, "grid_co2_g_kwh": 560.0, "source": "TODO_VERIFY TV-31: Shenzhen shore power proxy"},
    "Busan":       {"available": False, "tariff_usd_kwh": 0.17, "grid_co2_g_kwh": 450.0, "source": "TODO_VERIFY TV-31: Busan grid proxy"},
    "Tokyo":       {"available": False, "tariff_usd_kwh": 0.24, "grid_co2_g_kwh": 480.0, "source": "TODO_VERIFY TV-31: Tokyo port power proxy"},
    "Seattle":     {"available": True,  "tariff_usd_kwh": 0.14, "grid_co2_g_kwh": 120.0, "source": "TODO_VERIFY TV-31: Seattle hydro grid proxy"},
}

DEFAULT_SHORE_PROFILE = {"available": False, "tariff_usd_kwh": 0.18, "grid_co2_g_kwh": 400.0, "source": "TODO_VERIFY TV-31"}

# Vessel Auxiliary Power Demands (kW at berth)
VESSEL_AUX_POWER_KW: dict[str, float] = {
    "Feeder":       800.0,    # kW (TODO_VERIFY TV-32)
    "Panamax":     1800.0,    # kW (TODO_VERIFY TV-32)
    "PostPanamax": 2800.0,    # kW (TODO_VERIFY TV-32)
    "ULCS":        4000.0,    # kW (TODO_VERIFY TV-32)
}


_GLOBAL_LOOKUP_CACHE: dict[Any, dict] = {}


class FleetOptimizationProblemV2:
    """
    Phase 5 / Version 2 Fleet Optimization Problem.
    Decision 4-tuple per route: (vessel_idx, speed_idx, fuel_idx, route_option_idx).
    Supports candidate route geometries, weather-routing, ECA-minimizing, and shore power.
    """

    def __init__(
        self,
        routes: list[dict],
        vessels: list[dict],
        fuels: list[dict],
        weights: dict[str, float] | None = None,
        emissions_cap_t: float | None = None,
        vessel_limits: dict[str, int] | None = None,
        speed_bins: list[float] | None = None,
        instance_name: str = "Custom_V2",
    ):
        self.instance_name = instance_name
        self.routes = routes
        self.vessels = vessels
        self.fuels = fuels
        self.weights = weights or config.OPTIMIZATION_WEIGHTS.copy()
        self.emissions_cap_t = emissions_cap_t
        self.vessel_limits = vessel_limits or config.DEFAULT_FLEET_VESSEL_LIMITS.copy()

        self.num_routes = len(routes)
        self.num_vessels = len(vessels)
        self.num_fuels = len(fuels)

        self.surrogate = get_surrogate_model()
        self.speed_bins = list(speed_bins) if speed_bins is not None else [0.7, 0.8, 0.9, 1.0, 1.05]
        
        cache_key = (self.instance_name, self.num_routes, self.num_vessels, self.num_fuels, tuple(self.speed_bins))
        if cache_key in _GLOBAL_LOOKUP_CACHE:
            self._lookup_cache = _GLOBAL_LOOKUP_CACHE[cache_key]
        else:
            self._lookup_cache = {}
            self._precompute_lookup_table()
            _GLOBAL_LOOKUP_CACHE[cache_key] = self._lookup_cache

    def _precompute_lookup_table(self) -> None:
        """Precompute surrogate predictions across (route, option, vessel, speed, fuel)."""
        eval_rows = []
        key_list = []

        for r_idx, route in enumerate(self.routes):
            route_options = route.get("route_options", [
                {"option_id": 0, "name": "Direct", "dist_factor": 1.0, "weather_factor": 1.0, "eca_fraction": 0.15},
                {"option_id": 1, "name": "Weather-Optimized", "dist_factor": 1.04, "weather_factor": 0.92, "eca_fraction": 0.15},
                {"option_id": 2, "name": "ECA-Minimizing", "dist_factor": 1.06, "weather_factor": 1.00, "eca_fraction": 0.05},
            ])

            for opt in route_options:
                opt_id = opt["option_id"]
                dist = route["distance_nm"] * opt["dist_factor"]
                wf = opt["weather_factor"]

                for v_idx, vessel in enumerate(self.vessels):
                    des_spd = float(vessel["design_speed_kn"])
                    load_f = min(1.0, float(route["cargo_demand_teu"]) / float(vessel["capacity_teu"]))

                    for s_factor in self.speed_bins:
                        spd = round(float(s_factor * des_spd), 4)

                        for f_idx, fuel in enumerate(self.fuels):
                            key = (route["id"], opt_id, vessel["id"], spd, fuel["id"])
                            key_list.append(key)
                            eval_rows.append({
                                "vessel_class": vessel["vessel_class"],
                                "fuel_type": fuel["name"],
                                "speed_kn": spd,
                                "load_factor": load_f,
                                "weather_factor": wf,
                                "distance_nm": dist,
                            })

        if not eval_rows:
            return

        df_eval = pd.DataFrame(eval_rows)
        preds = self.surrogate.predict(df_eval)

        for i, key in enumerate(key_list):
            r_id, opt_id, v_id, spd, f_id = key
            fuel_t = float(preds[i])
            route = next(r for r in self.routes if r["id"] == r_id)
            vessel = next(v for v in self.vessels if v["id"] == v_id)
            fuel = next(f for f in self.fuels if f["id"] == f_id)

            route_options = route.get("route_options", [])
            opt = next((o for o in route_options if o["option_id"] == opt_id), {"dist_factor": 1.0, "weather_factor": 1.0})
            dist = route["distance_nm"] * opt["dist_factor"]

            # Transit calculations
            transit_h = dist / spd if spd > 0 else 0.0
            transit_days = transit_h / 24.0
            delay_days = max(0.0, transit_days - route["deadline_days"])

            # Fuel & Sea CO2
            fuel_cost = fuel_t * fuel["price_usd_per_tonne"]
            co2_sea_t = (fuel_t * 1e3 * fuel["lhv_mj_per_kg"] * fuel["co2_wtw_g_per_mj"]) * 1e-6

            # Port / Shore Power calculations (at destination port)
            dest_port = route.get("dest_port", "Rotterdam")
            shore_prof = PORT_SHORE_PROFILES.get(dest_port, DEFAULT_SHORE_PROFILE)
            berth_h = route.get("berth_hours", 36.0)
            aux_kw = VESSEL_AUX_POWER_KW.get(vessel["vessel_class"], 2000.0)
            at_berth_energy_kwh = aux_kw * berth_h

            if shore_prof["available"]:
                # Shore power used
                port_cost = at_berth_energy_kwh * shore_prof["tariff_usd_kwh"]
                co2_port_t = (at_berth_energy_kwh * shore_prof["grid_co2_g_kwh"]) * 1e-6
                used_shore = True
            else:
                # Auxiliary engine burning MGO at berth (approx 200 g/kWh specific fuel oil consumption)
                aux_mgo_t = (at_berth_energy_kwh * 200.0) * 1e-6
                port_cost = aux_mgo_t * config.BUNKER_PRICES_USD_PER_TONNE.get("MGO", 820.0)
                co2_port_t = (aux_mgo_t * 1e3 * config.LHV_MJ_PER_KG["MGO"] * config.CO2_WTW_G_CO2EQ_PER_MJ["MGO"]) * 1e-6
                used_shore = False

            total_fuel_and_port_cost = fuel_cost + port_cost
            total_co2_t = co2_sea_t + co2_port_t

            self._lookup_cache[key] = {
                "fuel_t": fuel_t,
                "fuel_cost": fuel_cost,
                "co2_sea_t": co2_sea_t,
                "port_cost": port_cost,
                "co2_port_t": co2_port_t,
                "total_co2_t": total_co2_t,
                "total_cost": total_fuel_and_port_cost,
                "transit_days": transit_days,
                "delay_days": delay_days,
                "used_shore_power": used_shore,
            }

    def evaluate(self, plan: list[dict]) -> tuple[float, bool, dict[str, Any]]:
        """
        Shared evaluate() for V2:
        Computes cost, emissions, schedule delay, penalty violations, and feasibility.
        """
        w1 = self.weights.get("w1_fuel_cost", 1.0)
        w2 = self.weights.get("w2_co2_emission", 100.0)
        w3 = self.weights.get("w3_schedule_penalty", 100000.0)

        total_cost = 0.0
        total_co2_t = 0.0
        total_delay_days = 0.0
        total_penalty = 0.0

        vessel_counts: dict[int, int] = {}
        violations = []

        for item in plan:
            r_id = item["route_id"]
            opt_id = item.get("route_option_id", 0)
            v_id = item["vessel_id"]
            spd = item["speed_kn"]
            f_id = item["fuel_id"]

            vessel_counts[v_id] = vessel_counts.get(v_id, 0) + 1

            key = (r_id, opt_id, v_id, spd, f_id)
            entry = self._lookup_cache.get(key)
            if entry:
                total_cost += entry["total_cost"]
                total_co2_t += entry["total_co2_t"]
                total_delay_days += entry["delay_days"]
            else:
                total_cost += 1e6

        # Quota constraint check
        for v in self.vessels:
            max_routes = self.vessel_limits.get(v["vessel_class"], self.vessel_limits.get(v["name"], 2))
            assigned = vessel_counts.get(v["id"], 0)
            if assigned > max_routes:
                excess = assigned - max_routes
                pen = excess * config.CONSTRAINT_PENALTIES["vessel_overlap"]
                total_penalty += pen
                violations.append(f"Vessel {v['name']} exceeded quota (+{excess})")

        # Emissions cap check
        if self.emissions_cap_t is not None and total_co2_t > self.emissions_cap_t:
            excess_co2 = total_co2_t - self.emissions_cap_t
            pen = excess_co2 * config.CONSTRAINT_PENALTIES["emissions_cap_per_tonne"]
            total_penalty += pen
            violations.append(f"Emissions cap exceeded (+{excess_co2:.1f}t)")

        is_feasible = (len(violations) == 0)
        base_objective = (w1 * total_cost) + (w2 * total_co2_t) + (w3 * total_delay_days)
        final_objective = base_objective + total_penalty

        return final_objective, is_feasible, {
            "cost_usd": total_cost,
            "co2_wtw_t": total_co2_t,
            "delay_days": total_delay_days,
            "penalty_usd": total_penalty,
            "is_feasible": is_feasible,
            "violations": violations,
        }
