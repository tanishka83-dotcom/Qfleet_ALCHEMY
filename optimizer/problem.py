"""
QFleet Phase 2 — Fleet Optimization Problem Formulation & Evaluation
====================================================================
Implements FleetOptimizationProblem:
- Loads scenarios, vessels, fuels, and routes directly from the SQLite DB.
- Provides seed_demo_scenario() to populate a benchmark scenario in DB.
- Evaluates candidate fleet dispatch solutions using the trained Phase 1
  Physics + XGBoost surrogate fuel model.
- Evaluates multi-objective cost (fuel cost, WTW lifecycle CO2, delay penalties).
- Enforces cargo capacity, schedule deadlines, and fleet emissions constraints.
- Persists optimization runs to the SQLite optimization_runs table.
"""

from __future__ import annotations

import sys
import time
import json
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Any

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
import xgboost as xgb

import config
import db as database
from models.fuel_model import compute_physics_baseline, prepare_features, load_data


# ---------------------------------------------------------------------------
# Surrogate ML Fuel Model (Phase 1 Physics + XGBoost Residual)
# ---------------------------------------------------------------------------

class SurrogateFuelModel:
    """
    Trained Phase 1 Physics + XGBoost residual fuel consumption model.
    Evaluates candidate voyage parameters (vessel_class, fuel_type, speed, load, distance).
    """

    def __init__(self):
        self._model = None
        self._feature_cols = None
        self._train_surrogate()

    def _train_surrogate(self) -> None:
        """Fit Physics+XGBoost model on the Phase 1 voyages dataset."""
        df = load_data()
        X_all, y_all = prepare_features(df)
        self._feature_cols = list(X_all.columns)

        indices = np.arange(len(df))
        train_idx, _ = train_test_split(
            indices,
            train_size=config.TRAIN_TEST_SPLIT,
            random_state=config.MODEL_RANDOM_SEED,
            shuffle=True,
        )
        train_df = df.iloc[train_idx].reset_index(drop=True)
        X_train = X_all.iloc[train_idx].reset_index(drop=True)

        phys_train = compute_physics_baseline(train_df)
        resid_train = train_df["fuel_consumed_t"].values - phys_train

        self._model = xgb.XGBRegressor(**config.XGBOOST_DEFAULT_PARAMS)
        self._model.fit(X_train, resid_train)

    def predict(self, eval_df: pd.DataFrame) -> np.ndarray:
        """
        Vectorised prediction of fuel consumed in tonnes.
        eval_df must contain: vessel_class, fuel_type, speed_kn, load_factor,
                              weather_factor, distance_nm
        """
        # Step 1: Analytic physics prediction
        phys_pred = compute_physics_baseline(eval_df)

        # Step 2: One-hot encoded feature matrix aligned with training columns
        dummies_vc = pd.get_dummies(eval_df["vessel_class"], prefix="vc", drop_first=False)
        dummies_ft = pd.get_dummies(eval_df["fuel_type"], prefix="ft", drop_first=False)

        raw_X = pd.concat([
            eval_df[["speed_kn", "load_factor", "weather_factor", "distance_nm"]].reset_index(drop=True),
            dummies_vc.reset_index(drop=True),
            dummies_ft.reset_index(drop=True)
        ], axis=1)

        # Ensure all training columns are present
        X_aligned = pd.DataFrame(0.0, index=np.arange(len(eval_df)), columns=self._feature_cols)
        for col in raw_X.columns:
            if col in self._feature_cols:
                X_aligned[col] = raw_X[col].values.astype(float)

        # Step 3: Add XGBoost residual correction
        xgb_resid = self._model.predict(X_aligned)
        final_pred = phys_pred + xgb_resid
        return np.maximum(final_pred, 1e-3)


# Singleton surrogate instance for fast evaluation across optimizer loops
_SURROGATE_INSTANCE: SurrogateFuelModel | None = None

def get_surrogate_model() -> SurrogateFuelModel:
    global _SURROGATE_INSTANCE
    if _SURROGATE_INSTANCE is None:
        _SURROGATE_INSTANCE = SurrogateFuelModel()
    return _SURROGATE_INSTANCE


# ---------------------------------------------------------------------------
# Demo Scenario Seeder
# ---------------------------------------------------------------------------

def seed_demo_scenario(session: Session | None = None) -> int:
    """
    Inserts a demo benchmark scenario into the SQLite database if none exists.
    Returns scenario_id.
    """
    engine = database.get_engine()
    database.init_db()

    def _seed(s: Session) -> int:
        existing = s.query(database.Scenario).filter_by(name="Demo_SIH26138_Fleet_Dispatch").first()
        if existing:
            return existing.id

        # Query existing vessels and routes
        vessels = s.query(database.Vessel).all()
        routes = s.query(database.Route).all()

        vessel_ids = [v.id for v in vessels]
        route_ids = [r.id for r in routes]

        fleet_cfg = {
            "name": "Demo_SIH26138_Fleet_Dispatch",
            "description": "Benchmark multi-route container fleet assignment and speed optimization scenario",
            "vessel_ids": vessel_ids,
            "route_ids": route_ids,
            "vessel_limits": config.DEFAULT_FLEET_VESSEL_LIMITS,
            "max_emissions_cap_t": config.DEFAULT_FLEET_EMISSIONS_CAP_T,
            "assumed_weather_factor": 1.08,
            "weights": config.OPTIMIZATION_WEIGHTS,
        }

        scenario = database.Scenario(
            name="Demo_SIH26138_Fleet_Dispatch",
            fleet_config_json=json.dumps(fleet_cfg, indent=2)
        )
        s.add(scenario)
        s.commit()
        s.refresh(scenario)
        return scenario.id

    if session is None:
        with Session(engine) as s:
            return _seed(s)
    else:
        return _seed(session)


# ---------------------------------------------------------------------------
# Fleet Optimization Problem Class
# ---------------------------------------------------------------------------

class FleetOptimizationProblem:
    """
    Problem definition loaded directly from the SQLite database.
    Manages solution encoding/decoding, surrogate evaluation, and DB persistence.
    """

    def __init__(self, scenario_id: int | None = None):
        self.engine = database.get_engine()
        database.init_db()

        # 1. Load Scenario from SQLite DB
        with Session(self.engine) as session:
            if scenario_id is None:
                scenario_id = seed_demo_scenario(session)
            scenario = session.query(database.Scenario).filter_by(id=scenario_id).first()
            if not scenario:
                scenario_id = seed_demo_scenario(session)
                scenario = session.query(database.Scenario).filter_by(id=scenario_id).first()

            self.scenario_id = scenario.id
            self.scenario_name = scenario.name
            self.config_data = json.loads(scenario.fleet_config_json)

            # 2. Load reference objects from DB
            v_ids = self.config_data.get("vessel_ids", [])
            r_ids = self.config_data.get("route_ids", [])

            self.vessels = [
                {
                    "id": v.id,
                    "name": v.name,
                    "vessel_class": v.vessel_class,
                    "capacity_teu": v.teu_capacity,
                    "design_speed_kn": v.design_speed_kn,
                    "ref_daily_fuel_t": v.ref_daily_fuel_t,
                    "min_speed_kn": 0.60 * v.design_speed_kn,
                    "max_speed_kn": 1.00 * v.design_speed_kn,
                }
                for v in session.query(database.Vessel).filter(database.Vessel.id.in_(v_ids)).all()
            ]

            self.fuels = [
                {
                    "id": f.id,
                    "name": f.name,
                    "lhv_mj_per_kg": f.lhv_mj_per_kg,
                    "co2_ttw_g_per_g": f.co2_ttw_g_per_g,
                    "co2_wtw_g_per_mj": f.co2_wtw_g_per_mj,
                    "engine_eff_ratio": f.engine_eff_ratio,
                    "price_usd_per_tonne": config.BUNKER_PRICES_USD_PER_TONNE.get(f.name, 650.0),
                }
                for f in session.query(database.Fuel).all()
            ]

            self.routes = [
                {
                    "id": r.id,
                    "name": r.name,
                    "origin_port": r.origin_port,
                    "dest_port": r.dest_port,
                    "distance_nm": float(r.distance_nm),
                    "typical_laden_pct": float(r.typical_laden_pct),
                    "cargo_demand_teu": int(r.cargo_demand_teu or 4000),
                    "deadline_days": float(r.deadline_days or 20.0),
                }
                for r in session.query(database.Route).filter(database.Route.id.in_(r_ids)).all()
            ]

        self.num_routes = len(self.routes)
        self.num_vessels = len(self.vessels)
        self.num_routes = len(self.routes)
        self.num_vessels = len(self.vessels)
        self.num_fuels = len(self.fuels)
        self.emissions_cap_t = float(self.config_data.get("max_emissions_cap_t", config.DEFAULT_FLEET_EMISSIONS_CAP_T))
        self.assumed_weather = float(self.config_data.get("assumed_weather_factor", 1.08))
        self.weights = self.config_data.get("weights", config.OPTIMIZATION_WEIGHTS)
        self.vessel_limits = self.config_data.get("vessel_limits", config.DEFAULT_FLEET_VESSEL_LIMITS)

        # Initialize surrogate fuel prediction model and precompute 600-entry grid cache
        self.surrogate = get_surrogate_model()
        self.eval_count = 0
        self._lookup_cache: dict[tuple[int, int, float, int], dict] = {}
        self._precompute_grid_cache()

    def _precompute_grid_cache(self) -> None:
        """Precompute the full candidate grid once for instant shared evaluation."""
        t0 = time.perf_counter()
        speed_multipliers = np.linspace(0.60, 1.00, 5)
        eval_rows = []
        meta_keys = []

        for r in self.routes:
            for v in self.vessels:
                load_factor = min(1.0, float(r["cargo_demand_teu"]) / float(v["capacity_teu"]))
                cargo_shortfall = max(0, r["cargo_demand_teu"] - v["capacity_teu"])

                for spd_mult in speed_multipliers:
                    spd = round(float(spd_mult * v["design_speed_kn"]), 4)
                    dur_days = r["distance_nm"] / (spd * 24.0)
                    delay_days = max(0.0, dur_days - r["deadline_days"])

                    for f in self.fuels:
                        eval_rows.append({
                            "vessel_class": v["vessel_class"],
                            "fuel_type": f["name"],
                            "speed_kn": spd,
                            "load_factor": float(load_factor),
                            "weather_factor": float(self.assumed_weather),
                            "distance_nm": float(r["distance_nm"]),
                        })
                        meta_keys.append({
                            "key": (r["id"], v["id"], spd, f["id"]),
                            "fuel_price": f["price_usd_per_tonne"],
                            "lhv": f["lhv_mj_per_kg"],
                            "wtw": f["co2_wtw_g_per_mj"],
                            "duration_days": dur_days,
                            "delay_days": delay_days,
                            "cargo_shortfall": cargo_shortfall,
                        })

        df_grid = pd.DataFrame(eval_rows)
        preds = self.surrogate.predict(df_grid)

        for i, meta in enumerate(meta_keys):
            f_t = float(preds[i])
            fuel_cost = f_t * meta["fuel_price"]
            energy_mj = f_t * 1e3 * meta["lhv"]
            co2_wtw = energy_mj * meta["wtw"] * 1e-6

            self._lookup_cache[meta["key"]] = {
                "fuel_t": f_t,
                "fuel_cost": fuel_cost,
                "co2_wtw_t": co2_wtw,
                "duration_days": meta["duration_days"],
                "delay_days": meta["delay_days"],
                "cargo_shortfall": meta["cargo_shortfall"],
            }
        self.surrogate_build_time_s = time.perf_counter() - t0

    # -----------------------------------------------------------------------
    # Encoding & Decoding
    # -----------------------------------------------------------------------

    def decode_solution(self, raw_solution: list[dict] | np.ndarray) -> list[dict]:
        """
        Convert internal vector / array format to structured route assignments.
        If raw_solution is a numpy array of shape (num_routes, 3):
          - col 0: continuous [0, 1] or angle in [0, pi/2] -> mapped to vessel_idx
          - col 1: continuous [0, 1] or angle in [0, pi/2] -> mapped to speed in [min_speed, max_speed]
          - col 2: continuous [0, 1] or angle in [0, pi/2] -> mapped to fuel_idx
        """
        if isinstance(raw_solution, list) and len(raw_solution) > 0 and isinstance(raw_solution[0], dict):
            return raw_solution

        arr = np.asarray(raw_solution, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape((self.num_routes, -1))

        decoded = []
        for r_idx, route in enumerate(self.routes):
            # Normalise continuous inputs (angles in [0, pi/2] or probabilities in [0, 1])
            v_val = arr[r_idx, 0]
            s_val = arr[r_idx, 1]
            f_val = arr[r_idx, 2]

            # If values are qubit angles in [0, pi/2], use sin^2(theta)
            if np.max(arr) <= np.pi / 2.0 + 1e-5:
                v_prob = np.sin(v_val) ** 2
                s_prob = np.sin(s_val) ** 2
                f_prob = np.sin(f_val) ** 2
            else:
                v_prob = np.clip(v_val, 0.0, 0.9999)
                s_prob = np.clip(s_val, 0.0, 1.0)
                f_prob = np.clip(f_val, 0.0, 0.9999)

            v_idx = int(np.floor(v_prob * self.num_vessels)) % self.num_vessels
            f_idx = int(np.floor(f_prob * self.num_fuels)) % self.num_fuels

            chosen_vessel = self.vessels[v_idx]
            chosen_fuel = self.fuels[f_idx]

            # Speed discretized to 5 exact bins: 60%, 70%, 80%, 90%, 100% of design speed
            speed_multipliers = np.linspace(0.60, 1.00, 5)
            s_bin_idx = int(np.floor(s_prob * 5)) % 5
            speed_kn = round(float(speed_multipliers[s_bin_idx] * chosen_vessel["design_speed_kn"]), 4)

            decoded.append({
                "route_id": route["id"],
                "route_name": route["name"],
                "vessel_id": chosen_vessel["id"],
                "vessel_name": chosen_vessel["name"],
                "vessel_class": chosen_vessel["vessel_class"],
                "speed_kn": float(speed_kn),
                "fuel_id": chosen_fuel["id"],
                "fuel_type": chosen_fuel["name"],
            })

        return decoded

    # -----------------------------------------------------------------------
    # Shared Fast Evaluation via Cached Grid
    # -----------------------------------------------------------------------

    def evaluate(self, raw_solution: list[dict] | np.ndarray, count_eval: bool = True) -> tuple[float, bool, dict]:
        """
        Evaluates a fleet dispatch solution using the shared precomputed surrogate table.
        Returns:
            objective_score : float (lower is better)
            is_feasible     : bool (true if all hard constraints satisfied)
            details         : dict with granular breakdown
        """
        if count_eval:
            self.eval_count += 1
        assignments = self.decode_solution(raw_solution)

        total_fuel_t = 0.0
        total_fuel_cost = 0.0
        total_co2_wtw_t = 0.0
        total_delay_days = 0.0
        cargo_unmet_teu = 0.0
        vessel_counts: dict[int, int] = {}

        w1 = self.weights.get("w1_fuel_cost", 1.0)
        w2 = self.weights.get("w2_co2_emission", 100.0)
        w3 = self.weights.get("w3_schedule_penalty", 100000.0)

        for i, a in enumerate(assignments):
            spd_key = round(float(a["speed_kn"]), 4)
            cache_key = (a["route_id"], a["vessel_id"], spd_key, a["fuel_id"])

            if cache_key in self._lookup_cache:
                entry = self._lookup_cache[cache_key]
                total_fuel_t += entry["fuel_t"]
                total_fuel_cost += entry["fuel_cost"]
                total_co2_wtw_t += entry["co2_wtw_t"]
                total_delay_days += entry["delay_days"]
                cargo_unmet_teu += entry["cargo_shortfall"]
            else:
                # Fallback if key slightly differs in float precision
                route = self.routes[i]
                vessel = next(v for v in self.vessels if v["id"] == a["vessel_id"])
                fuel = next(f for f in self.fuels if f["id"] == a["fuel_id"])
                load_factor = min(1.0, float(route["cargo_demand_teu"]) / float(vessel["capacity_teu"]))

                df_single = pd.DataFrame([{
                    "vessel_class": a["vessel_class"],
                    "fuel_type": a["fuel_type"],
                    "speed_kn": float(a["speed_kn"]),
                    "load_factor": float(load_factor),
                    "weather_factor": float(self.assumed_weather),
                    "distance_nm": float(route["distance_nm"]),
                }])
                f_t = float(self.surrogate.predict(df_single)[0])
                cost_f = f_t * fuel["price_usd_per_tonne"]
                co2_f = f_t * 1e3 * fuel["lhv_mj_per_kg"] * fuel["co2_wtw_g_per_mj"] * 1e-6
                dur_days = route["distance_nm"] / (a["speed_kn"] * 24.0)
                delay_f = max(0.0, dur_days - route["deadline_days"])
                shortfall_f = max(0, route["cargo_demand_teu"] - vessel["capacity_teu"])

                total_fuel_t += f_t
                total_fuel_cost += cost_f
                total_co2_wtw_t += co2_f
                total_delay_days += delay_f
                cargo_unmet_teu += shortfall_f

            vessel_counts[a["vessel_id"]] = vessel_counts.get(a["vessel_id"], 0) + 1

        # Fleet size / vessel exclusivity limits
        vessel_overlap_conflicts = 0
        for v in self.vessels:
            v_id = v["id"]
            count = vessel_counts.get(v_id, 0)
            max_allowed = self.vessel_limits.get(
                v["name"],
                self.vessel_limits.get(v["vessel_class"], self.vessel_limits.get(str(v_id), 2))
            )
            if count > max_allowed:
                vessel_overlap_conflicts += (count - max_allowed)

        # Emissions cap excess
        emissions_excess_t = max(0.0, total_co2_wtw_t - self.emissions_cap_t)

        # Penalties
        pen_cargo = cargo_unmet_teu * config.CONSTRAINT_PENALTIES.get("cargo_capacity_per_teu", 200.0)
        pen_emissions = emissions_excess_t * config.CONSTRAINT_PENALTIES.get("emissions_cap_per_tonne", 500.0)
        pen_overlap = vessel_overlap_conflicts * config.CONSTRAINT_PENALTIES.get("vessel_overlap", 500000.0)

        # Base composite objective
        base_objective = (w1 * total_fuel_cost) + (w2 * total_co2_wtw_t) + (w3 * total_delay_days)
        total_objective = base_objective + pen_cargo + pen_emissions + pen_overlap

        is_feasible = (cargo_unmet_teu == 0) and (emissions_excess_t == 0) and (vessel_overlap_conflicts == 0) and (total_delay_days < 1e-4)

        details = {
            "total_objective": float(total_objective),
            "base_objective": float(base_objective),
            "fuel_cost_usd": float(total_fuel_cost),
            "total_fuel_t": float(total_fuel_t),
            "co2_wtw_t": float(total_co2_wtw_t),
            "total_delay_days": float(total_delay_days),
            "cargo_unmet_teu": float(cargo_unmet_teu),
            "emissions_excess_t": float(emissions_excess_t),
            "vessel_overlap_conflicts": int(vessel_overlap_conflicts),
            "is_feasible": is_feasible,
            "assignments": assignments,
        }

        return total_objective, is_feasible, details

    # -----------------------------------------------------------------------
    # Database Persistence
    # -----------------------------------------------------------------------

    def save_run(
        self,
        algorithm: str,
        seed: int | None,
        solution: list[dict] | np.ndarray,
        objective: float,
        runtime_s: float,
        convergence_history: list[float] | None = None,
    ) -> int:
        """
        Persist optimization results into the SQLite optimization_runs table.
        """
        _, _, details = self.evaluate(solution, count_eval=False)

        converged_iter = None
        if convergence_history and len(convergence_history) > 0:
            best_val = min(convergence_history)
            converged_iter = int(np.argmin(convergence_history))

        run_record = database.OptimizationRun(
            scenario_id=self.scenario_id,
            algorithm=algorithm,
            seed=seed,
            objective=float(objective),
            fuel_t=float(details["total_fuel_t"]),
            co2_t=float(details["co2_wtw_t"]),
            runtime_s=float(runtime_s),
            converged_at_iter=converged_iter,
        )

        with Session(self.engine) as session:
            session.add(run_record)
            session.commit()
            session.refresh(run_record)
            return run_record.id

    def uncoupled_brute_force_optimal(self) -> tuple[list[dict], float, bool, dict]:
        """
        Calculates exact per-route optimum without fleet vessel quotas (uncoupled).
        """
        speed_multipliers = np.linspace(0.60, 1.00, 5)
        w1 = self.weights.get("w1_fuel_cost", 1.0)
        w2 = self.weights.get("w2_co2_emission", 100.0)
        w3 = self.weights.get("w3_schedule_penalty", 100000.0)

        best_plan = []
        for r in self.routes:
            best_r_obj = float("inf")
            best_r_entry = None
            for v in self.vessels:
                if v["capacity_teu"] < r["cargo_demand_teu"]:
                    continue
                for spd_mult in speed_multipliers:
                    spd = round(float(spd_mult * v["design_speed_kn"]), 4)
                    for f in self.fuels:
                        entry = self._lookup_cache[(r["id"], v["id"], spd, f["id"])]
                        r_obj = (w1 * entry["fuel_cost"]) + (w2 * entry["co2_wtw_t"]) + (w3 * entry["delay_days"])
                        if r_obj < best_r_obj:
                            best_r_obj = r_obj
                            best_r_entry = {
                                "route_id": r["id"],
                                "route_name": r["name"],
                                "vessel_id": v["id"],
                                "vessel_name": v["name"],
                                "vessel_class": v["vessel_class"],
                                "speed_kn": spd,
                                "fuel_id": f["id"],
                                "fuel_type": f["name"],
                            }
            best_plan.append(best_r_entry)

        obj, is_feas, details = self.evaluate(best_plan)
        return best_plan, obj, is_feas, details

    def exact_brute_force_optimal(self) -> tuple[list[dict], float, bool, dict]:
        """
        Calculates exact global optimum across all 4^6 = 4,096 vessel assignments
        subject to vessel class route quotas and minimum objective over (speed, fuel).
        """
        import itertools

        speed_multipliers = np.linspace(0.60, 1.00, 5)
        w1 = self.weights.get("w1_fuel_cost", 1.0)
        w2 = self.weights.get("w2_co2_emission", 100.0)
        w3 = self.weights.get("w3_schedule_penalty", 100000.0)

        # Precompute best (speed, fuel) for each (route, vessel)
        best_rv_choice = {}
        for r in self.routes:
            for v in self.vessels:
                if v["capacity_teu"] < r["cargo_demand_teu"]:
                    continue
                best_score = float("inf")
                best_tuple = None
                for spd_mult in speed_multipliers:
                    spd = round(float(spd_mult * v["design_speed_kn"]), 4)
                    for f in self.fuels:
                        entry = self._lookup_cache[(r["id"], v["id"], spd, f["id"])]
                        score = (w1 * entry["fuel_cost"]) + (w2 * entry["co2_wtw_t"]) + (w3 * entry["delay_days"])
                        if score < best_score:
                            best_score = score
                            best_tuple = (spd, f["id"], f["name"], score)
                best_rv_choice[(r["id"], v["id"])] = best_tuple

        best_global_obj = float("inf")
        best_global_plan = None

        for v_combo in itertools.product(self.vessels, repeat=len(self.routes)):
            # Check capacity
            if any(v["capacity_teu"] < r["cargo_demand_teu"] for v, r in zip(v_combo, self.routes)):
                continue

            # Check quotas
            v_counts = {}
            for v in v_combo:
                v_counts[v["id"]] = v_counts.get(v["id"], 0) + 1

            if any(v_counts.get(v["id"], 0) > self.vessel_limits.get(v["name"], 2) for v in self.vessels):
                continue

            # Compute sum of minimum route objectives
            combo_plan = []
            total_obj = 0.0
            for r, v in zip(self.routes, v_combo):
                spd, f_id, f_name, r_score = best_rv_choice[(r["id"], v["id"])]
                combo_plan.append({
                    "route_id": r["id"],
                    "route_name": r["name"],
                    "vessel_id": v["id"],
                    "vessel_name": v["name"],
                    "vessel_class": v["vessel_class"],
                    "speed_kn": spd,
                    "fuel_id": f_id,
                    "fuel_type": f_name,
                })
                total_obj += r_score

            if total_obj < best_global_obj:
                best_global_obj = total_obj
                best_global_plan = combo_plan

        obj, is_feas, details = self.evaluate(best_global_plan)
        return best_global_plan, obj, is_feas, details
