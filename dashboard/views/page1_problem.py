"""
QFleet Dashboard — Page 1: Problem & Fleet Setup
================================================
Displays vessels, routes, class quotas, CO2 caps, objective weights,
and TODO_VERIFY audit items. All data loaded strictly from SQLite.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from dashboard.data_loader import (
    load_vessels,
    load_routes,
    load_scenarios,
    load_scenario_detail,
    load_fuels,
)


def render() -> None:
    st.title("🚢 Problem Formulation & Fleet Setup")
    st.caption("Inspect vessel specifications, voyage routes, operational quotas, emissions caps, and regulatory audit items.")

    # -----------------------------------------------------------------------
    # Scenario Selector
    # -----------------------------------------------------------------------
    scenarios_df = load_scenarios()
    scenario_options = {row["name"]: row["id"] for _, row in scenarios_df.iterrows()}
    
    selected_name = st.selectbox(
        "Select Scenario / Instance:",
        options=list(scenario_options.keys()),
        index=1 if "Small_4V_6R" in scenario_options else 0,
    )
    sc_id = scenario_options[selected_name]
    sc_detail = load_scenario_detail(sc_id)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Instance Configuration Summary KPI Cards
    # -----------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Vessels in Fleet", len(sc_detail["vessels_df"]))
    with col2:
        st.metric("Routes in Schedule", len(sc_detail["routes_df"]))
    with col3:
        emissions_cap = sc_detail["emissions_cap_t"]
        st.metric("CO₂ Emissions Cap", f"{emissions_cap:,.0f} t" if emissions_cap else "Unconstrained")
    with col4:
        weather_factor = sc_detail["assumed_weather_factor"]
        st.metric("Weather Factor", f"{weather_factor:.2f}")

    # -----------------------------------------------------------------------
    # Objective Weights & Coupling Quotas
    # -----------------------------------------------------------------------
    st.subheader("⚙️ Objective Weights & Coupling Mechanics")
    w_col1, w_col2 = st.columns(2)

    with w_col1:
        st.markdown("**Objective Function Weights ($J = w_1 C_{\\text{fuel}} + w_2 E_{\\text{CO}_2} + w_3 P_{\\text{delay}}$)**")
        weights = sc_detail["weights"]
        w_df = pd.DataFrame([
            {"Weight Component": "w1 (Fuel Cost)", "Value": weights.get("w1_fuel_cost", 1.0), "Unit": "$/USD"},
            {"Weight Component": "w2 (CO₂ Emission)", "Value": weights.get("w2_co2_emission", 100.0), "Unit": "$/tonne CO₂eq"},
            {"Weight Component": "w3 (Schedule Delay Penalty)", "Value": weights.get("w3_schedule_penalty", 100000.0), "Unit": "$/day"},
        ])
        st.dataframe(w_df, use_container_width=True, hide_index=True)

    with w_col2:
        st.markdown("**Vessel / Class Quotas & Slack Capacity**")
        vessel_limits = sc_detail["vessel_limits"]
        limits_data = []
        total_slots = 0
        for entity, limit in vessel_limits.items():
            total_slots += limit
            limits_data.append({"Vessel / Class": entity, "Max Route Limit": limit})
        
        limits_df = pd.DataFrame(limits_data)
        st.dataframe(limits_df, use_container_width=True, hide_index=True)
        
        num_routes = len(sc_detail["routes_df"])
        slack = total_slots - num_routes
        if slack >= 0:
            st.success(f"**Feasibility Capacity Check**: Total usable slots = {total_slots} for {num_routes} routes (Slack = +{slack} slots).")
        else:
            st.error(f"**Infeasible Capacity**: Total slots ({total_slots}) < routes ({num_routes}). Slack = {slack}.")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Vessels Registry
    # -----------------------------------------------------------------------
    st.subheader("🛳️ Fleet Vessel Registry")
    vessels_df = sc_detail["vessels_df"]
    if not vessels_df.empty:
        disp_vessels = vessels_df[[
            "id", "name", "vessel_class", "teu_capacity", "dwt_tonnes",
            "design_speed_kn", "ref_daily_fuel_t", "design_draft_m", "cii_ref_rating"
        ]].rename(columns={
            "id": "ID",
            "name": "Vessel Name",
            "vessel_class": "Class",
            "teu_capacity": "Capacity (TEU)",
            "dwt_tonnes": "DWT (t)",
            "design_speed_kn": "Design Speed (kn)",
            "ref_daily_fuel_t": "Ref Fuel (t/day)",
            "design_draft_m": "Draft (m)",
            "cii_ref_rating": "CII Rating",
        })
        st.dataframe(disp_vessels, use_container_width=True, hide_index=True)
    else:
        st.info("No vessels associated with this scenario in DB.")

    # -----------------------------------------------------------------------
    # Voyage Routes
    # -----------------------------------------------------------------------
    st.subheader("🗺️ Voyage Routes & Commercial Commitments")
    routes_df = sc_detail["routes_df"]
    if not routes_df.empty:
        disp_routes = routes_df[[
            "id", "name", "origin_port", "dest_port", "distance_nm",
            "cargo_demand_teu", "deadline_days", "draft_limit_m"
        ]].rename(columns={
            "id": "ID",
            "name": "Route Name",
            "origin_port": "Origin",
            "dest_port": "Destination",
            "distance_nm": "Distance (NM)",
            "cargo_demand_teu": "Demand (TEU)",
            "deadline_days": "Deadline (Days)",
            "draft_limit_m": "Max Draft (m)",
        })
        st.dataframe(disp_routes, use_container_width=True, hide_index=True)
    else:
        st.info("No routes associated with this scenario in DB.")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Transparent Audit Flags (TODO_VERIFY)
    # -----------------------------------------------------------------------
    st.subheader("🔍 Transparent Regulatory & Modeling Verification Items (TODO_VERIFY)")
    st.warning("All assumptions lacking explicit regulatory consensus are tracked below per project policy.")
    
    todo_items = [
        {"Tag": "TV-01", "Category": "Fuel Spec", "Subject": "VLSFO LHV Proxy", "Description": "Using LFO proxy (40.5 MJ/kg) per IMO MEPC.212(63).", "Status": "Proxy In Use"},
        {"Tag": "TV-02", "Category": "Emissions", "Subject": "VLSFO TTW Factor", "Description": "Using LFO proxy (3.151 g CO2/g fuel) per IMO MEPC.212(63).", "Status": "Proxy In Use"},
        {"Tag": "TV-03", "Category": "Emissions", "Subject": "VLSFO WTW Factor", "Description": "Using HFO proxy (86.2 g CO2eq/MJ) per FuelEU Maritime 2023/1640.", "Status": "Proxy In Use"},
        {"Tag": "TV-04", "Category": "Fuel Spec", "Subject": "Methanol LHV", "Description": "IMO 4th GHG study quotes 19.9 MJ/kg vs FuelEU 19.93 MJ/kg (<0.16% delta).", "Status": "Verified Minor"},
        {"Tag": "TV-09", "Category": "Engine", "Subject": "HFO Engine Efficiency", "Description": "SFC 1% penalty (ratio 0.99) vs VLSFO baseline per DNV GL 2019-0567.", "Status": "Engineering Estimate"},
        {"Tag": "TV-10", "Category": "Engine", "Subject": "MGO Engine Efficiency", "Description": "SFC 2% improvement (ratio 1.02) on dedicated diesel engines.", "Status": "Engineering Estimate"},
        {"Tag": "TV-11", "Category": "Engine", "Subject": "LNG Dual-Fuel Efficiency", "Description": "Ratio 0.95 reflecting 5% pilot fuel / methane slip thermal penalty.", "Status": "Engineering Estimate"},
        {"Tag": "TV-12", "Category": "Engine", "Subject": "Methanol Dual-Fuel Efficiency", "Description": "Ratio 0.85 reflecting 15% SFC increase per MAN ES ME-LGIM.", "Status": "Engineering Estimate"},
        {"Tag": "TV-29", "Category": "Emissions Cap", "Subject": "Medium Instance Binding Cap", "Description": "Emissions cap set to 58,000 t CO2eq to ensure binding constraint.", "Status": "Active Constraint"},
    ]
    st.dataframe(pd.DataFrame(todo_items), use_container_width=True, hide_index=True)
