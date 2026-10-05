"""
QFleet Dashboard — Page 2: Run Explorer
=======================================
Inspects specific optimization runs from the SQLite database:
- Filters to canonical benchmark run set.
- Reports run exclusion audit (demo + historical/superseded).
- Displays recorded metrics from SQLite (read-only).
- Shows feasibility checks (emissions cap).

Note: Dispatch plans are not stored in the DB; this page displays
a clear message instead of re-running optimizers.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from dashboard.data_loader import (
    load_optimization_runs_audit,
    load_scenario_detail,
    load_scenarios,
)


def render() -> None:
    st.title("🔎 Optimization Run Explorer")
    st.caption("Inspect individual optimization executions, objective metrics, and feasibility constraints.")

    audit = load_optimization_runs_audit()
    canonical_df = audit["canonical_benchmark_df"]

    # -----------------------------------------------------------------------
    # Run Set Audit Header
    # -----------------------------------------------------------------------
    total = audit["total_runs_count"]
    canonical = audit["canonical_benchmark_count"]
    demo_excl = audit["demo_excluded_count"]
    superseded_excl = audit["superseded_excluded_count"]
    excluded = audit["excluded_runs_count"]

    with st.expander(f"📊 Run Set Audit & Filtering Summary ({total} DB Runs)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total DB Runs", total)
        c2.metric("Canonical Benchmark Runs", canonical, delta="Active Filter")
        c3.metric("Excluded Demo Runs (Scen 1)", demo_excl)
        c4.metric("Excluded Historical / Superseded", superseded_excl)
        st.info(
            f"**Audit Rationale**: {canonical} canonical runs represent "
            f"3 benchmark instances × (1 MILP + 1 Greedy + 10 GA + 10 SQA + 10 QI-EA). "
            f"{excluded} runs were excluded: {demo_excl} from the Phase 1 demo "
            f"and {superseded_excl} earlier test iterations / legacy algorithm naming."
        )

    # -----------------------------------------------------------------------
    # Filter Selectors
    # -----------------------------------------------------------------------
    st.subheader("🎯 Select Optimization Run")
    scenarios_df = load_scenarios()
    sc_map = {row["name"]: row["id"] for _, row in scenarios_df.iterrows()}

    col_sc, col_alg, col_seed = st.columns(3)

    with col_sc:
        bench_sc_names = [name for name in ["Small_4V_6R", "Medium_8V_15R", "Large_15V_30R"] if name in sc_map]
        selected_sc_name = st.selectbox("Scenario / Instance:", bench_sc_names, index=0)
        selected_sc_id = sc_map[selected_sc_name]

    sc_runs = canonical_df[canonical_df["scenario_id"] == selected_sc_id]
    available_algos = sorted(sc_runs["algo_normalized"].unique().tolist())

    with col_alg:
        selected_algo = st.selectbox("Algorithm:", available_algos, index=0 if available_algos else 0)

    algo_runs = sc_runs[sc_runs["algo_normalized"] == selected_algo]
    available_seeds = sorted(algo_runs["seed"].unique().tolist())

    with col_seed:
        selected_seed = st.selectbox("Random Seed:", available_seeds, index=0 if available_seeds else 0)

    matched_run = algo_runs[algo_runs["seed"] == selected_seed]
    if matched_run.empty:
        st.warning("No matching canonical run found for selected combination.")
        return

    run_row = matched_run.iloc[0]

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Recorded Run Metrics KPI Bar (Strictly from DB)
    # -----------------------------------------------------------------------
    st.subheader("📈 Recorded Execution Metrics (from SQLite DB)")

    def fmt_compact_currency(val: float) -> str:
        if val >= 1_000_000:
            return f"${val / 1_000_000:.2f}M"
        if val >= 1_000:
            return f"${val / 1_000:.2f}k"
        return f"${val:,.2f}"

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric(
            "Objective J",
            fmt_compact_currency(run_row['objective']),
            help=f"Exact objective value: ${run_row['objective']:,.2f} (USD)"
        )
        st.caption(f"Exact: ${run_row['objective']:,.0f}")
    with m_col2:
        st.metric("Fuel Consumed", f"{run_row['fuel_t']:,.1f} t")
    with m_col3:
        st.metric("CO₂ Emissions", f"{run_row['co2_t']:,.1f} t (WTW)")
    with m_col4:
        st.metric("Runtime", f"{run_row['runtime_s']:.4f} s")
    with m_col5:
        st.metric("Converged Iter.", int(run_row['converged_at_iter']))

    # -----------------------------------------------------------------------
    # Feasibility Evaluation
    # -----------------------------------------------------------------------
    sc_detail = load_scenario_detail(selected_sc_id)
    emissions_cap = sc_detail["emissions_cap_t"]
    co2_emitted = run_row["co2_t"]

    st.subheader("🛡️ Feasibility & Constraint Status")
    f_col1, f_col2 = st.columns(2)

    with f_col1:
        if emissions_cap is not None:
            headroom = emissions_cap - co2_emitted
            usage_ratio = min(1.0, max(0.0, co2_emitted / emissions_cap))
            st.write(f"**CO₂ Emissions Cap Utilization**: {co2_emitted:,.1f} t / {emissions_cap:,.1f} t")
            st.progress(usage_ratio)
            if headroom >= 0:
                st.success(f"✅ Compliant — Headroom: {headroom:,.1f} t remaining ({ (1.0 - usage_ratio)*100:.1f}% buffer)")
            else:
                st.error(f"❌ Cap Exceeded — Violation: +{abs(headroom):,.1f} t over cap")
        else:
            st.info("Emissions cap is unconstrained for this instance.")

    with f_col2:
        st.info(f"**DB Run Timestamp**: {run_row['run_at']} | **Run Record ID**: #{run_row['id']}")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Fleet Voyage Dispatch Plan — read-only
    # -----------------------------------------------------------------------
    st.subheader("📋 Voyage Dispatch Assignment Plan")
    st.info(
        "📦 **Dispatch plan detail is not stored in the database.** "
        "The `optimization_runs` table records aggregate metrics "
        "(objective, fuel, CO₂, runtime) but not the per-route "
        "vessel-assignment vector. Use **Page 6 (Plan a Fleet)** "
        "to run a live optimization and view the full dispatch table."
    )
