"""
QFleet Dashboard — Page 4: Fuels & Carbon Price Sensitivity
===========================================================
Displays:
1. Fuel specifications from SQLite fuels table (HFO, VLSFO, MGO, LNG, METHANOL).
2. Carbon price sensitivity curve ($0 to $200 / t CO2eq).
   STRICTLY LABELED: "Recorded plan re-priced, not re-optimized".
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dashboard.data_loader import (
    load_fuels,
    load_benchmark_csv,
    load_optimization_runs_audit,
    load_scenarios,
)


def render() -> None:
    st.title("🌱 Marine Fuels & Carbon Price Sensitivity")
    st.caption("Inspect regulatory marine fuel properties and analyze the financial impact of carbon pricing on recorded fleet plans.")

    # -----------------------------------------------------------------------
    # Fuel Master Grid (from SQLite)
    # -----------------------------------------------------------------------
    st.subheader("⛽ Maritime Fuel Registry (from SQLite DB)")
    fuels_df = load_fuels()
    
    disp_fuels = fuels_df[[
        "id", "name", "lhv_mj_per_kg", "co2_ttw_g_per_g", "co2_wtw_g_per_mj", "engine_eff_ratio", "source"
    ]].rename(columns={
        "id": "ID",
        "name": "Fuel Name",
        "lhv_mj_per_kg": "LHV (MJ/kg)",
        "co2_ttw_g_per_g": "TTW Factor (g CO₂/g)",
        "co2_wtw_g_per_mj": "WTW Factor (g CO₂eq/MJ)",
        "engine_eff_ratio": "Engine Efficiency Ratio",
        "source": "Regulatory / Engineering Source",
    })
    st.dataframe(disp_fuels, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Carbon Price Sensitivity
    # -----------------------------------------------------------------------
    st.subheader("📈 Carbon Price Sensitivity Analysis")
    st.warning("⚠️ **Note**: **Recorded plan re-priced, not re-optimized**. This analysis rescales the carbon cost term on existing dispatch plans without modifying route assignments or speeds.")

    bench_df = load_benchmark_csv()
    
    col_inst, col_range = st.columns([1, 2])
    with col_inst:
        selected_instance = st.selectbox("Select Instance:", ["Small", "Medium", "Large"], index=1)
    
    with col_range:
        current_carbon_price = st.slider("Carbon Price ($ / t CO₂eq):", min_value=0, max_value=200, value=100, step=10)

    # Filter CSV rows for selected instance
    inst_bench = bench_df[bench_df["instance"] == selected_instance].copy()
    
    # Generate carbon price sweep from $0 to $200 in steps of $10
    carbon_prices = np.linspace(0, 200, 21)
    
    sweep_records = []
    for _, row in inst_bench.iterrows():
        algo = row["algorithm"]
        mean_fuel_cost = row["mean_fuel_cost_usd"]
        mean_co2 = row["mean_co2_wtw_t"]
        
        # Original w3 delay penalty component = mean_objective - fuel_cost - (100 * co2)
        base_delay_penalty = max(0.0, row["mean_objective"] - mean_fuel_cost - (100.0 * mean_co2))
        
        for cp in carbon_prices:
            repriced_total = mean_fuel_cost + (cp * mean_co2) + base_delay_penalty
            sweep_records.append({
                "Algorithm": algo,
                "Carbon Price ($/t)": cp,
                "Re-priced Objective ($)": repriced_total,
                "Carbon Cost Component ($)": cp * mean_co2,
                "Fuel Cost Component ($)": mean_fuel_cost,
            })

    sweep_df = pd.DataFrame(sweep_records)

    # Plot re-priced trajectory
    fig_sweep = px.line(
        sweep_df,
        x="Carbon Price ($/t)",
        y="Re-priced Objective ($)",
        color="Algorithm",
        title=f"Fleet Operational Cost vs Carbon Tax — {selected_instance} Instance (Re-priced Plans)",
        color_discrete_map={
            "MILP_HiGHS": "#10b981",
            "GA": "#3b82f6",
            "Greedy": "#f59e0b",
            "SQA": "#8b5cf6",
            "QI-EA": "#ef4444",
        },
    )
    fig_sweep.add_vline(x=current_carbon_price, line_dash="dash", line_color="white", annotation_text=f"Current: ${current_carbon_price}/t")
    fig_sweep.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_sweep, use_container_width=True)

    # Breakdown table at the selected carbon price
    st.markdown(f"**Cost Breakdown at ${current_carbon_price}/t CO₂eq:**")
    current_df = sweep_df[sweep_df["Carbon Price ($/t)"] == current_carbon_price].copy()
    current_df["Fuel Cost ($)"] = current_df["Fuel Cost Component ($)"].apply(lambda x: f"${x:,.0f}")
    current_df["Carbon Cost ($)"] = current_df["Carbon Cost Component ($)"].apply(lambda x: f"${x:,.0f}")
    current_df["Total Re-priced ($)"] = current_df["Re-priced Objective ($)"].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(
        current_df[["Algorithm", "Fuel Cost ($)", "Carbon Cost ($)", "Total Re-priced ($)"]],
        use_container_width=True,
        hide_index=True,
    )
