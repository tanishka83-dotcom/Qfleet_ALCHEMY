"""
QFleet Dashboard — Page 6: Plan a Fleet (Interactive Fleet Optimizer)
====================================================================
Interactive dispatch optimizer for custom fleet and voyage configurations:
- Pick voyages/vessels from DB catalogue or upload a custom CSV (<= 50 routes).
- Configurable carbon tax ($/t CO2eq) and fleet emissions cap.
- Multi-algorithm execution: MILP (exact), Greedy (+Decarb), GA, SQA, QI-EA.
- Convergence tracking, naive baseline savings, and feasibility reporting.
- Strictly in-memory execution: NEVER writes to data/qfleet.db.

Quantum-inspired classical algorithms: normal CPU, no qubits or quantum hardware.
"""

from __future__ import annotations

import io
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Any

import config
from dashboard.data_loader import load_vessels, load_routes, load_fuels
from optimizer.problem_v2 import FleetOptimizationProblemV2, PORT_SHORE_PROFILES
from optimizer.baselines_v2 import milp_optimize_v2
from optimizer.algorithms_v2 import (
    greedy_optimize_v2,
    ga_optimize_v2,
    sqa_optimize_v2,
    qiea_optimize_v2,
)


def validate_fleet_csv(file_or_buffer: Any) -> tuple[pd.DataFrame | None, str | None]:
    """
    Validates uploaded voyage CSV schema and route limits (<= 50 routes).
    Returns (DataFrame, None) on success, or (None, error_message) on failure.
    """
    try:
        df = pd.read_csv(file_or_buffer)
        required_cols = {"name", "origin_port", "dest_port", "distance_nm", "cargo_demand_teu", "deadline_days"}
        missing = required_cols - set(df.columns)
        if missing:
            return None, f"Missing required columns: {sorted(list(missing))}"
        if len(df) > 50:
            return None, f"Number of routes ({len(df)}) exceeds maximum allowed limit of 50."
        if len(df) == 0:
            return None, "Uploaded CSV contains no route rows."
        numeric_cols = ["distance_nm", "cargo_demand_teu", "deadline_days"]
        for c in numeric_cols:
            if not pd.to_numeric(df[c], errors="coerce").notnull().all():
                return None, f"Column '{c}' must contain valid numeric values."
        return df, None
    except Exception as e:
        return None, f"Error parsing CSV: {e}"


def render() -> None:
    st.title("⚓ Plan a Fleet — Interactive Optimizer")
    st.caption("Configure custom fleet assignments, test decarbonization caps, and benchmark quantum-inspired classical optimizers in real-time.")

    # -----------------------------------------------------------------------
    # Step 1: Fleet & Route Configuration
    # -----------------------------------------------------------------------
    st.subheader("1. Voyage Demand & Fleet Capacity Setup")

    input_mode = st.radio("Input Source:", ["Select from Database Catalogue", "Upload Custom CSV File"], horizontal=True)

    db_routes = load_routes()
    db_vessels = load_vessels()
    db_fuels = load_fuels()

    vessels_data = []
    routes_data = []

    if input_mode == "Upload Custom CSV File":
        uploaded_file = st.file_uploader("Upload Voyage CSV (max 50 routes):", type=["csv"])
        if uploaded_file is not None:
            custom_df, err_msg = validate_fleet_csv(uploaded_file)
            if err_msg:
                st.error(err_msg)
                return
            routes_data = custom_df.to_dict(orient="records")
            for i, r in enumerate(routes_data):
                r["id"] = i + 1
                r["route_options"] = [
                    {"option_id": 0, "name": "Direct", "dist_factor": 1.0, "weather_factor": 1.0, "eca_fraction": 0.15},
                    {"option_id": 1, "name": "Weather-Optimized", "dist_factor": 1.04, "weather_factor": 0.92, "eca_fraction": 0.15},
                    {"option_id": 2, "name": "ECA-Minimizing", "dist_factor": 1.06, "weather_factor": 1.00, "eca_fraction": 0.05},
                ]
            st.success(f"Successfully loaded {len(routes_data)} custom routes.")
            # Default vessels for uploaded CSV
            num_v = min(4, len(db_vessels))
            for _, v in db_vessels.iloc[:num_v].iterrows():
                vessels_data.append({
                    "id": int(v["id"]), "name": str(v["name"]),
                    "vessel_class": str(v["vessel_class"]),
                    "capacity_teu": int(v["capacity_teu"]),
                    "design_speed_kn": float(v["design_speed_kn"]),
                })
        else:
            st.info("Upload a CSV file or switch to Database Catalogue.")
            return
    else:
        col_r_sel, col_v_sel = st.columns(2)
        with col_r_sel:
            num_routes_sel = st.slider("Number of Routes to Dispatch:", min_value=2, max_value=min(30, len(db_routes)), value=6)
            routes_subset = db_routes.iloc[:num_routes_sel]
            for _, r in routes_subset.iterrows():
                routes_data.append({
                    "id": int(r["id"]),
                    "name": str(r["name"]),
                    "origin_port": str(r["origin_port"]),
                    "dest_port": str(r["dest_port"]),
                    "distance_nm": float(r["distance_nm"]),
                    "cargo_demand_teu": int(r["cargo_demand_teu"]),
                    "deadline_days": float(r["deadline_days"]),
                    "route_options": [
                        {"option_id": 0, "name": "Direct", "dist_factor": 1.0, "weather_factor": 1.0, "eca_fraction": 0.15},
                        {"option_id": 1, "name": "Weather-Optimized", "dist_factor": 1.04, "weather_factor": 0.92, "eca_fraction": 0.15},
                        {"option_id": 2, "name": "ECA-Minimizing", "dist_factor": 1.06, "weather_factor": 1.00, "eca_fraction": 0.05},
                    ]
                })

        with col_v_sel:
            num_vessels_sel = st.slider("Vessels Available in Fleet:", min_value=2, max_value=min(15, len(db_vessels)), value=4)
            vessels_subset = db_vessels.iloc[:num_vessels_sel]
            for _, v in vessels_subset.iterrows():
                vessels_data.append({
                    "id": int(v["id"]), "name": str(v["name"]),
                    "vessel_class": str(v["vessel_class"]),
                    "capacity_teu": int(v["capacity_teu"]),
                    "design_speed_kn": float(v["design_speed_kn"]),
                })

    # Fuels list
    fuels_data = [
        {
            "id": int(f["id"]),
            "name": str(f["name"]),
            "price_usd_per_tonne": float(f["price_usd_per_tonne"]),
            "lhv_mj_per_kg": float(f["lhv_mj_per_kg"]),
            "co2_wtw_g_per_mj": float(f["co2_wtw_g_per_mj"]),
        }
        for _, f in db_fuels.iterrows() if f["name"] in config.SIMULATION_FUELS
    ]

    # -----------------------------------------------------------------------
    # Step 2: Policy & Constraints
    # -----------------------------------------------------------------------
    st.subheader("2. Carbon Pricing & Environmental Caps")
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        carbon_tax = st.number_input("Carbon Tax ($ / t CO₂eq):", min_value=0, max_value=500, value=100, step=10, help="Shadow carbon price w2 (TODO_VERIFY TV-10)")
    with c_col2:
        enable_cap = st.checkbox("Enable Fleet CO₂ Emissions Cap", value=True)
        if enable_cap:
            cap_input = st.number_input("Emissions Cap (Tonnes CO₂eq):", min_value=1000, max_value=500000, value=35000, step=5000)
        else:
            cap_input = None

    # -----------------------------------------------------------------------
    # Step 3: Run Optimization
    # -----------------------------------------------------------------------
    st.markdown("---")
    run_btn = st.button("🚀 Run Fleet Optimization Benchmark", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Executing mathematical and heuristic dispatch algorithms..."):
            weights = {
                "w1_fuel_cost": 1.0,
                "w2_co2_emission": float(carbon_tax),
                "w3_schedule_penalty": 100000.0,
            }

            vessel_limits = {v["name"]: max(2, int(np.ceil(len(routes_data) / len(vessels_data)) + 1)) for v in vessels_data}

            prob = FleetOptimizationProblemV2(
                routes=routes_data,
                vessels=vessels_data,
                fuels=fuels_data,
                weights=weights,
                emissions_cap_t=cap_input if enable_cap else None,
                vessel_limits=vessel_limits,
            )

            # ------ Run all methods ------
            results_list = []

            # 1. Exact MILP (synchronous)
            t0 = time.perf_counter()
            milp_plan, milp_obj, milp_stats = milp_optimize_v2(prob)
            milp_time = time.perf_counter() - t0
            milp_stats["runtime_s"] = milp_time
            milp_stats["eval_count"] = 1
            milp_stats["algorithm"] = "MILP (Exact HiGHS)"
            milp_stats["type"] = "Exact Mathematical Solver"
            milp_stats["plan"] = milp_plan
            milp_stats["objective"] = milp_obj
            results_list.append(milp_stats)

            # 2. Greedy (pure)
            _, greedy_obj, greedy_stats = greedy_optimize_v2(prob, decarb_pass=False, seed=42)
            greedy_stats["algorithm"] = "Greedy (pure)"
            greedy_stats["type"] = "Constructive Heuristic"
            greedy_stats["objective"] = greedy_obj
            results_list.append(greedy_stats)

            # 3. Greedy (+decarb)
            gd_plan, gd_obj, gd_stats = greedy_optimize_v2(prob, decarb_pass=True, seed=42)
            gd_stats["algorithm"] = "Greedy (+decarb)"
            gd_stats["type"] = "Constructive + Repair"
            gd_stats["plan"] = gd_plan
            gd_stats["objective"] = gd_obj
            results_list.append(gd_stats)

            # 4. GA
            _, ga_obj, ga_stats = ga_optimize_v2(prob, pop_size=20, n_generations=40, seed=42)
            ga_stats["algorithm"] = "GA (Genetic Algorithm)"
            ga_stats["type"] = "Metaheuristic"
            ga_stats["objective"] = ga_obj
            results_list.append(ga_stats)

            # 5. SQA
            _, sqa_obj, sqa_stats = sqa_optimize_v2(prob, n_trotter=10, n_steps=50, seed=42)
            sqa_stats["algorithm"] = "SQA (Simulated Quantum Annealing)"
            sqa_stats["type"] = "Quantum-Inspired Classical"
            sqa_stats["objective"] = sqa_obj
            results_list.append(sqa_stats)

            # 6. QI-EA
            _, qiea_obj, qiea_stats = qiea_optimize_v2(prob, pop_size=20, n_generations=40, seed=42)
            qiea_stats["algorithm"] = "QI-EA (Quantum-Inspired EA)"
            qiea_stats["type"] = "Quantum-Inspired Classical"
            qiea_stats["objective"] = qiea_obj
            results_list.append(qiea_stats)

            # 7. Naive Baseline (Fastest speed, HFO, direct route, no coupling)
            naive_plan = []
            for r in routes_data:
                v_cand = [v for v in vessels_data if v["capacity_teu"] >= r["cargo_demand_teu"]]
                v_chosen = v_cand[0] if v_cand else vessels_data[-1]
                naive_plan.append({
                    "route_id": r["id"],
                    "route_option_id": 0,
                    "route_option_name": "Direct",
                    "vessel_id": v_chosen["id"],
                    "vessel_name": v_chosen["name"],
                    "vessel_class": v_chosen["vessel_class"],
                    "speed_kn": round(float(v_chosen["design_speed_kn"] * 1.05), 4),
                    "fuel_id": 1,  # HFO
                    "fuel_type": "HFO",
                })
            naive_obj, naive_feas, naive_det = prob.evaluate(naive_plan)

            # Display Results Overview
            st.success("Optimization Completed!")

            # ------ Method Comparison Table ------
            st.subheader("📊 Method Comparison & Feasibility Status")

            summary_rows = []
            for res in results_list:
                is_f = res.get("is_feasible", False)
                obj_val = res.get("objective", 0)
                gap = "--"
                if is_f and milp_stats.get("is_feasible", False) and milp_obj > 0:
                    if res["algorithm"].startswith("MILP"):
                        gap = "0.00% (Reference)"
                    else:
                        gap = f"{((obj_val - milp_obj) / milp_obj) * 100:+.2f}%"

                # Violated constraint info
                feas_str = "✅ Feasible" if is_f else "❌ Failed: no feasible plan"
                if not is_f:
                    violations = []
                    co2 = res.get("co2_t", 0)
                    if cap_input and co2 > cap_input:
                        violations.append(f"CO₂ cap exceeded by {co2 - cap_input:,.0f} t")
                    if violations:
                        feas_str = f"❌ {'; '.join(violations)}"

                summary_rows.append({
                    "Algorithm": res["algorithm"],
                    "Type": res["type"],
                    "Objective ($)": f"${obj_val:,.2f}" if is_f else "Failed: no feasible plan",
                    "CO₂ WTW (t)": f"{res.get('co2_t', 0):,.1f}" if is_f else "--",
                    "Feasibility": feas_str,
                    "Runtime (s)": f"{res.get('runtime_s', 0):.4f}",
                    "Evaluations": str(res.get("eval_count", "--")),
                    "Gap to MILP": gap,
                })

            # Add naive baseline
            summary_rows.append({
                "Algorithm": "Naive Baseline (HFO, max speed)",
                "Type": "Uncoupled Heuristic",
                "Objective ($)": f"${naive_obj:,.2f}" if naive_feas else "Failed: no feasible plan",
                "CO₂ WTW (t)": f"{naive_det['co2_wtw_t']:,.1f}" if naive_feas else "--",
                "Feasibility": "✅ Feasible" if naive_feas else "❌ Cap/Quota Violated",
                "Runtime (s)": "< 0.001",
                "Evaluations": "1",
                "Gap to MILP": f"{((naive_obj - milp_obj)/milp_obj)*100:+.2f}%" if (naive_feas and milp_stats.get("is_feasible")) else "--",
            })

            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            # ------ Baseline savings KPI ------
            if milp_stats.get("is_feasible", False):
                saving_usd = naive_det["cost_usd"] - milp_stats.get("cost_usd", 0)
                co2_saving_t = naive_det["co2_wtw_t"] - milp_stats.get("co2_t", 0)
                saving_pct = (saving_usd / naive_det["cost_usd"]) * 100 if naive_det["cost_usd"] > 0 else 0

                st.subheader("💰 Savings vs Naive Baseline")
                st.caption("Reference type: MILP (exact) vs Naive (fastest speed, HFO, no coupling)")
                b1, b2, b3 = st.columns(3)
                b1.metric("Financial Savings", f"${saving_usd:,.0f}", delta=f"{saving_pct:.1f}% Reduction")
                b2.metric("Emissions Abated", f"{co2_saving_t:,.1f} t CO₂", delta=f"{(co2_saving_t/naive_det['co2_wtw_t'])*100:.1f}% Cleaner" if naive_det["co2_wtw_t"] > 0 else "--")
                b3.metric("Fleet Status", "Optimized", delta="All constraints met")

            # ------ Dispatch Table (MILP) ------
            if milp_stats.get("is_feasible", False) and milp_plan:
                st.subheader("📋 Optimal Fleet Dispatch Assignment (MILP)")
                disp_rows = []
                for item in milp_plan:
                    r_obj = next((r for r in routes_data if r["id"] == item["route_id"]), None)
                    if r_obj:
                        disp_rows.append({
                            "Voyage": r_obj.get("name", f"Route-{item['route_id']}"),
                            "Origin → Destination": f"{r_obj.get('origin_port', '?')} → {r_obj.get('dest_port', '?')}",
                            "Route Choice": item.get("route_option_name", "Direct"),
                            "Assigned Vessel": item.get("vessel_name", "--"),
                            "Operating Speed": f"{item['speed_kn']:.1f} kn",
                            "Fuel Type": item.get("fuel_type", "--"),
                            "In-Port Power": "🔌 Shore Power" if item.get("used_shore_power") else "⛽ Auxiliary Fuel",
                        })
                st.dataframe(pd.DataFrame(disp_rows), use_container_width=True, hide_index=True)

            # ------ Convergence Chart (placeholder using final objectives) ------
            st.subheader("📈 Algorithm Performance Comparison")
            heuristic_results = [r for r in results_list if not r["algorithm"].startswith("MILP")]
            if heuristic_results:
                perf_data = []
                for r in heuristic_results:
                    perf_data.append({
                        "Algorithm": r["algorithm"],
                        "Objective ($)": r.get("objective", 0) if r.get("is_feasible") else None,
                        "Runtime (s)": r.get("runtime_s", 0),
                        "Evaluations": r.get("eval_count", 0),
                        "Feasible": "✅" if r.get("is_feasible") else "❌ Infeasible",
                    })
                perf_df = pd.DataFrame(perf_data)

                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(
                        perf_df, x="Algorithm", y="Runtime (s)",
                        color="Feasible",
                        title="Runtime by Algorithm",
                        color_discrete_map={"✅": "#10b981", "❌ Infeasible": "#ef4444"},
                    )
                    fig.update_layout(template="plotly_dark", height=350, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    feasible_perf = perf_df[perf_df["Objective ($)"].notna()].copy()
                    if not feasible_perf.empty:
                        fig2 = px.bar(
                            feasible_perf, x="Algorithm", y="Objective ($)",
                            title="Objective Value (feasible runs only)",
                            color="Algorithm",
                        )
                        fig2.update_layout(template="plotly_dark", height=350, showlegend=False)
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.warning("No heuristic found a feasible plan.")

            # ------ Algorithm Disclaimer ------
            st.info(
                "💡 **Quantum-Inspired Algorithm Note**: "
                "QI-EA and SQA are **quantum-inspired classical algorithms** running on a normal classical CPU. "
                "There are no qubits, superposition, or quantum hardware involved. "
                "MILP (exact) is the best-performing solver on the tested sizes."
            )
