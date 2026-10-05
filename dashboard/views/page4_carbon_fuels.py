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
import config
from dashboard.background import configure_plotly_theme
from dashboard.data_loader import (
    load_fuels,
    load_extended_fuel_proxies,
    load_benchmark_csv,
    load_optimization_runs_audit,
    load_scenarios,
)


def reprice_benchmark_records(
    benchmark_df: pd.DataFrame,
    carbon_price: float,
    reference_fuel_price: float,
) -> pd.DataFrame:
    """Reprice stored aggregate costs without changing stored emissions or plans."""
    result = benchmark_df.copy()
    base_price = config.BUNKER_PRICES_USD_PER_TONNE["VLSFO"]
    price_scale = reference_fuel_price / base_price if base_price else 1.0
    result["Repriced fuel cost ($)"] = result["mean_fuel_cost_usd"] * price_scale
    result["Stored emissions (t CO₂eq)"] = result["mean_co2_wtw_t"]
    result["Repriced carbon cost ($)"] = (
        result["Stored emissions (t CO₂eq)"] * carbon_price
    )
    base_carbon_price = config.OPTIMIZATION_WEIGHTS["w2_co2_emission"]
    result["Other objective cost ($)"] = (
        result["mean_objective"]
        - result["mean_fuel_cost_usd"]
        - base_carbon_price * result["mean_co2_wtw_t"]
    ).clip(lower=0)
    result["Repriced total ($)"] = (
        result["Repriced fuel cost ($)"]
        + result["Repriced carbon cost ($)"]
        + result["Other objective cost ($)"]
    )
    return result


def build_fuel_comparison(
    fuels_df: pd.DataFrame,
    selected_fuels: list[str],
    price_scale: float,
) -> pd.DataFrame:
    selected = fuels_df[fuels_df["name"].isin(selected_fuels)].copy()
    selected["display_name"] = selected["name"].replace({
        "LH2_GREEN": "Green hydrogen (LH2, proxy)",
        "AMMONIA_GREEN": "Green ammonia (proxy)",
    })
    selected["price_proxy_usd_per_tonne"] = (
        selected["price_usd_per_tonne"] * price_scale
    )
    selected["cost_proxy_usd_per_gj"] = (
        selected["price_proxy_usd_per_tonne"] / selected["lhv_mj_per_kg"]
    )
    return selected


def _reset_filters() -> None:
    for key in (
        "page4_extended_fuels", "p4_fuels", "p4_instance",
        "p4_reference_price", "p4_carbon_price",
    ):
        st.session_state.pop(key, None)


def render() -> None:
    configure_plotly_theme()
    st.title("Carbon & Fuels")
    st.caption("Compare stored fuel factors and reprice recorded benchmark emissions; plans are not re-optimized.")

    bench_df = load_benchmark_csv()
    instance_options = bench_df["instance"].astype(str).drop_duplicates().tolist()
    raw_fuels = load_fuels().copy()
    raw_fuels["price_usd_per_tonne"] = raw_fuels["name"].map(
        config.BUNKER_PRICES_USD_PER_TONNE
    )
    raw_fuels["is_proxy"] = False

    st.sidebar.subheader("Carbon and fuel filters")
    st.sidebar.button("Reset filters", key="p4_reset_filters", on_click=_reset_filters)
    include_extended = st.sidebar.checkbox(
        "Include hydrogen and ammonia (extended scenario)",
        value=False,
        key="page4_extended_fuels",
    )
    if include_extended:
        proxies = load_extended_fuel_proxies()
        if not proxies.empty:
            proxies = proxies.copy()
            proxies["is_proxy"] = True
            raw_fuels = pd.concat([raw_fuels, proxies], ignore_index=True, sort=False)
        else:
            st.sidebar.warning("Extended fuel proxy data is not initialized.")

    fuel_options = raw_fuels["name"].astype(str).tolist()
    if "p4_fuels" not in st.session_state:
        st.session_state["p4_fuels"] = fuel_options
    else:
        st.session_state["p4_fuels"] = [
            name for name in st.session_state["p4_fuels"] if name in fuel_options
        ]
    st.sidebar.multiselect("Fuels", fuel_options, key="p4_fuels")
    if "p4_instance" not in st.session_state or st.session_state["p4_instance"] not in instance_options:
        st.session_state["p4_instance"] = "Medium" if "Medium" in instance_options else instance_options[0]
    st.sidebar.selectbox("Benchmark instance", instance_options, key="p4_instance")
    st.sidebar.slider(
        "Reference fuel price ($/t, VLSFO-equivalent)", 0.0, 5000.0,
        value=float(config.BUNKER_PRICES_USD_PER_TONNE["VLSFO"]), step=25.0,
        key="p4_reference_price",
    )
    st.sidebar.slider(
        "Carbon price ($/t CO₂eq)", 0, 500, value=100, step=10,
        key="p4_carbon_price",
    )
    current_price = st.session_state["p4_reference_price"]
    carbon_price = st.session_state["p4_carbon_price"]
    base_price = config.BUNKER_PRICES_USD_PER_TONNE["VLSFO"]
    price_scale = current_price / base_price if base_price else 1.0
    fuels_df = build_fuel_comparison(
        raw_fuels, st.session_state["p4_fuels"], price_scale
    )
    selected_instance = st.session_state["p4_instance"]
    inst_bench = bench_df[bench_df["instance"] == selected_instance].copy()
    repriced = reprice_benchmark_records(inst_bench, carbon_price, current_price)

    # -----------------------------------------------------------------------
    # Fuel Master Grid (from SQLite)
    # -----------------------------------------------------------------------
    st.subheader("Fuel factors and price proxies")
    kpis = st.columns(4)
    kpis[0].metric("Selected fuels", len(fuels_df))
    kpis[1].metric("Stored benchmark runs", int(inst_bench["n_runs"].sum()))
    kpis[2].metric("Mean stored emissions", f"{inst_bench['mean_co2_wtw_t'].mean():,.1f} t CO₂eq")
    kpis[3].metric("Median repriced total", f"${repriced['Repriced total ($)'].median():,.0f}")
    best_row = repriced.loc[repriced["Repriced total ($)"].idxmin()]
    st.caption(
        f"At ${current_price:,.0f}/t reference fuel and ${carbon_price:,.0f}/t CO₂eq, "
        f"{best_row['algorithm']} has the lowest repriced total for {selected_instance} "
        f"(${best_row['Repriced total ($)']:,.0f}); emissions remain stored values."
    )

    if fuels_df.empty:
        st.info("No fuels selected. Select at least one fuel to populate fuel comparison charts.")
        fuel_display = pd.DataFrame(columns=["Fuel", "Price proxy ($/t)"])
    else:
        fuels_df["display_name"] = fuels_df["name"].replace({
            "LH2_GREEN": "Green hydrogen (LH2, proxy)",
            "AMMONIA_GREEN": "Green ammonia (proxy)",
        })
        fuel_display = fuels_df[[
            "display_name", "lhv_mj_per_kg", "co2_ttw_g_per_g",
            "co2_wtw_g_per_mj", "price_proxy_usd_per_tonne",
            "cost_proxy_usd_per_gj", "source",
        ]].rename(columns={
            "display_name": "Fuel",
            "lhv_mj_per_kg": "LHV (MJ/kg)",
            "co2_ttw_g_per_g": "TTW CO₂ (g/g)",
            "co2_wtw_g_per_mj": "WTW CO₂eq (g/MJ)",
            "price_proxy_usd_per_tonne": "Price proxy ($/t)",
            "cost_proxy_usd_per_gj": "Energy cost proxy ($/GJ)",
            "source": "Source / verification",
        })
    st.download_button(
        "Download fuel comparison CSV",
        fuel_display.to_csv(index=False).encode("utf-8"),
        file_name="qfleet_p4_fuels.csv",
        mime="text/csv",
        key="p4_fuels_csv",
    )
    st.dataframe(fuel_display, use_container_width=True, hide_index=True, key="p4_fuel_table")

    st.caption(
        "Hydrogen and ammonia use unverified pathway, engine-efficiency, and price proxies. "
        "The fuel-price slider scales aggregate stored fuel cost from a VLSFO-equivalent baseline."
    )
    st.subheader("Energy-adjusted fuel comparison")
    comparison_col, emissions_col = st.columns(2)
    with comparison_col:
        cost_fig = px.bar(
            fuels_df,
            x="display_name",
            y="cost_proxy_usd_per_gj",
            color="is_proxy",
            labels={
                "display_name": "Fuel",
                "cost_proxy_usd_per_gj": "Fuel cost proxy ($/GJ)",
                "is_proxy": "Extended proxy",
            },
            title="Energy-normalized fuel cost (proxy values)",
        )
        st.plotly_chart(cost_fig, use_container_width=True)
    with emissions_col:
        emissions_fig = px.bar(
            fuels_df,
            x="display_name",
            y="co2_wtw_g_per_mj",
            color="is_proxy",
            labels={
                "display_name": "Fuel",
                "co2_wtw_g_per_mj": "WTW emissions (g CO₂eq/MJ)",
                "is_proxy": "Extended proxy",
            },
            title="Well-to-wake emissions by fuel",
        )
        st.plotly_chart(emissions_fig, use_container_width=True)

    reference = raw_fuels[raw_fuels["name"] == "VLSFO"].iloc[0]
    reference_cost_per_gj = (
        reference["price_usd_per_tonne"] * price_scale / reference["lhv_mj_per_kg"]
    )
    breakeven_rows = []
    for _, fuel in fuels_df.iterrows():
        if fuel["name"] == "VLSFO":
            continue
        intensity_reduction = (
            reference["co2_wtw_g_per_mj"] - fuel["co2_wtw_g_per_mj"]
        )
        cost_premium = fuel["cost_proxy_usd_per_gj"] - reference_cost_per_gj
        if intensity_reduction <= 0:
            continue
        breakeven_rows.append({
            "Fuel": fuel["display_name"],
            "Breakeven carbon price ($/t CO₂eq)": max(
                0.0, cost_premium * 1000.0 / intensity_reduction
            ),
        })
    breakeven_fig = px.bar(
        pd.DataFrame(breakeven_rows),
        x="Fuel",
        y="Breakeven carbon price ($/t CO₂eq)",
        title="Carbon price to match VLSFO energy cost (extended fuels are proxies)",
    )
    st.plotly_chart(breakeven_fig, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Carbon Price Sensitivity
    # -----------------------------------------------------------------------
    st.subheader("📈 Carbon Price Sensitivity Analysis")
    st.warning("⚠️ **Note**: **Recorded plan re-priced, not re-optimized**. This analysis rescales the carbon cost term on existing dispatch plans without modifying route assignments or speeds.")

    # Generate carbon price sweep from $0 to $200 in steps of $10
    carbon_prices = np.linspace(0, 200, 21)
    
    sweep_records = []
    for _, row in inst_bench.iterrows():
        algo = row["algorithm"]
        for cp in carbon_prices:
            adjusted = reprice_benchmark_records(
                pd.DataFrame([row]), cp, current_price
            ).iloc[0]
            sweep_records.append({
                "Algorithm": algo,
                "Carbon Price ($/t)": cp,
                "Re-priced Objective ($)": adjusted["Repriced total ($)"],
                "Carbon Cost Component ($)": adjusted["Repriced carbon cost ($)"],
                "Fuel Cost Component ($)": adjusted["Repriced fuel cost ($)"],
                "Stored Emissions (t CO₂eq)": adjusted["Stored emissions (t CO₂eq)"],
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
    fig_sweep.add_vline(x=carbon_price, line_dash="dash", line_color="white", annotation_text=f"Current: ${carbon_price}/t")
    fig_sweep.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_sweep, use_container_width=True)

    pareto_fig = px.scatter(
        repriced,
        x="Stored emissions (t CO₂eq)",
        y="Repriced total ($)",
        color="algorithm",
        hover_name="algorithm",
        hover_data=["mean_objective", "Repriced fuel cost ($)", "Repriced carbon cost ($)"],
        labels={
            "Stored emissions (t CO₂eq)": "Stored mean emissions (t CO₂eq)",
            "Repriced total ($)": "Repriced objective proxy ($)",
            "algorithm": "Algorithm",
        },
        title=f"Cost vs stored CO₂ — {selected_instance}",
    )
    pareto_fig.update_layout(clickmode="event+select", height=380)
    st.plotly_chart(pareto_fig, use_container_width=True, key="p4_cost_co2_pareto")

    # Breakdown table at the selected carbon price
    st.markdown(f"**Cost Breakdown at ${carbon_price}/t CO₂eq:**")
    current_df = repriced[[
        "algorithm", "mean_objective", "Stored emissions (t CO₂eq)",
        "Repriced fuel cost ($)", "Repriced carbon cost ($)", "Repriced total ($)",
    ]].rename(columns={
        "algorithm": "Algorithm",
        "mean_objective": "Stored objective ($)",
    })
    st.download_button(
        "Download repriced benchmark CSV",
        current_df.to_csv(index=False).encode("utf-8"),
        file_name="qfleet_p4_repriced.csv",
        mime="text/csv",
        key="p4_repriced_csv",
    )
    st.dataframe(
        current_df,
        use_container_width=True,
        hide_index=True,
        key="p4_repriced_table",
    )
