"""
QFleet Dashboard — Page 3: Benchmark Results
===========================================
Displays Phase 3 comprehensive benchmark results:
- Dynamic table from data/benchmark_results.csv
- Instance specs derived from benchmark_results.csv and scenario DB
- Reference type and J* from CSV
- Feasibility definition (quotas + binding emissions cap)
- Interactive charts: Gap, Runtime
- Wilcoxon Signed-Rank matrix loaded from results/statistical_tests.csv
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.data_loader import (
    load_benchmark_csv,
    load_statistical_tests,
    load_optimization_runs_audit,
    load_scenario_detail,
    load_scenarios,
)


# ---- instance metadata not stored in CSVs ----
_SEARCH_SPACE_LABELS: dict[str, str] = {
    "Small": "$4^6 = 4{,}096$ vessel combos (100% brute-force covered)",
    "Medium": "$8^{15} \\approx 3.52 \\times 10^{13}$ vessel combos",
    "Large": "$15^{30} \\approx 1.92 \\times 10^{35}$ vessel combos",
}
_EVAL_BUDGETS: dict[str, int] = {
    "Small": 1_000,
    "Medium": 5_000,
    "Large": 10_000,
}
_INSTANCE_SCENARIO_MAP: dict[str, str] = {
    "Small": "Small_4V_6R",
    "Medium": "Medium_8V_15R",
    "Large": "Large_15V_30R",
}


def render() -> None:
    st.title("🏆 Benchmark Results & Statistical Rigor")
    st.caption(
        "Empirical performance comparison of optimization algorithms "
        "across Small, Medium, and Large benchmark instances."
    )

    # -----------------------------------------------------------------------
    # Ingest data
    # -----------------------------------------------------------------------
    bench_df = load_benchmark_csv()
    stat_df = load_statistical_tests()

    # Derive instance-level reference info from CSV
    instances = bench_df.groupby("instance").first().reset_index()

    scenarios_df = load_scenarios()
    sc_name_to_id = {row["name"]: row["id"] for _, row in scenarios_df.iterrows()}

    # -----------------------------------------------------------------------
    # Instance Context & Feasibility Rules
    # -----------------------------------------------------------------------
    with st.expander("📌 Benchmark Instance Specifications & Feasibility Protocol", expanded=True):
        cols = st.columns(len(instances))
        for col, (_, inst_row) in zip(cols, instances.iterrows()):
            inst = inst_row["instance"]
            ref_obj = inst_row["reference_obj"]
            ref_type = inst_row["reference_type"]
            sc_name = _INSTANCE_SCENARIO_MAP.get(inst, "")
            sc_id = sc_name_to_id.get(sc_name)

            cap_str = "N/A"
            if sc_id is not None:
                detail = load_scenario_detail(sc_id)
                cap_val = detail.get("emissions_cap_t")
                if cap_val:
                    cap_str = f"{cap_val:,.0f} t CO₂eq"
                    n_vessels = len(detail.get("vessels_df", []))
                    n_routes = len(detail.get("routes_df", []))
                else:
                    cap_str = "Unconstrained"
                    n_vessels = len(detail.get("vessels_df", []))
                    n_routes = len(detail.get("routes_df", []))
            else:
                n_vessels = "?"
                n_routes = "?"

            with col:
                st.markdown(f"### {inst} ({n_vessels}V, {n_routes}R)")
                st.markdown(f"- **Search Space**: {_SEARCH_SPACE_LABELS.get(inst, 'N/A')}")
                st.markdown(f"- **Emissions Cap**: {cap_str}")
                st.markdown(f"- **Reference Type**: {ref_type}")
                st.markdown(f"- **Reference $J^*$**: ${ref_obj:,.0f}")
                st.markdown(f"- **Eval Budget**: {_EVAL_BUDGETS.get(inst, 'N/A'):,} evaluations")

        st.info(
            "**Formal Feasibility Definition**: A dispatch plan $x$ is feasible **iff**:\n"
            "1. **Coupling Quotas**: Each vessel $v$ does not exceed its maximum route limit $\\sum_r x_{rv} \\le q_v$.\n"
            "2. **Emissions Cap**: Total Well-to-Wake CO₂ emissions satisfy $\\sum_r E_r(x) \\le E_{\\text{max}}$."
        )

    # -----------------------------------------------------------------------
    # Performance Summary Tables
    # -----------------------------------------------------------------------
    st.subheader("📊 Performance Summary Table (from benchmark_results.csv)")

    instance_names = bench_df["instance"].unique().tolist()
    tab_names = [f"{inst} Instance" for inst in instance_names] + ["All Combined"]
    tabs = st.tabs(tab_names)

    def _format_table(df_subset: pd.DataFrame) -> pd.DataFrame:
        formatted = df_subset.copy()
        formatted["Mean Objective ($)"] = formatted["mean_objective"].apply(lambda x: f"${x:,.0f}")
        formatted["Gap vs Ref (%)"] = formatted["gap_to_ref_pct"].apply(lambda x: f"{x:.2f}%")
        formatted["Std ($)"] = formatted["std_objective"].apply(lambda x: f"${x:,.0f}" if x > 0 else "—")
        formatted["Best ($)"] = formatted["best_objective"].apply(lambda x: f"${x:,.0f}")
        formatted["Worst ($)"] = formatted["worst_objective"].apply(lambda x: f"${x:,.0f}")
        formatted["Feasibility Rate"] = formatted["feasibility_rate_pct"].apply(lambda x: f"{x:.0f}%")
        formatted["Mean Runtime (s)"] = formatted["mean_runtime_s"].apply(lambda x: f"{x:.3f} s")
        formatted["Surrogate Build (s)"] = formatted["surrogate_build_time_s"].apply(lambda x: f"{x:.4f} s")

        cols = [
            "algorithm", "Reference Type", "Mean Objective ($)", "Gap vs Ref (%)",
            "Std ($)", "Best ($)", "Worst ($)", "Feasibility Rate", "Mean Runtime (s)", "Surrogate Build (s)"
        ]
        formatted["Reference Type"] = formatted["reference_type"]
        return formatted[cols].rename(columns={"algorithm": "Algorithm"})

    for i, inst in enumerate(instance_names):
        with tabs[i]:
            inst_data = bench_df[bench_df["instance"] == inst]
            ref_row = inst_data.iloc[0]
            st.caption(
                f"{inst} Instance — Reference: {ref_row['reference_type']} "
                f"($J^* = ${ref_row['reference_obj']:,.0f})"
            )
            st.dataframe(_format_table(inst_data), use_container_width=True, hide_index=True)

    with tabs[-1]:
        st.dataframe(_format_table(bench_df), use_container_width=True, hide_index=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Interactive Visualizations
    # -----------------------------------------------------------------------
    st.subheader("📈 Visual Comparison Charts")
    chart_col1, chart_col2 = st.columns(2)

    _algo_colors = {
        "MILP_HiGHS": "#10b981",
        "GA": "#3b82f6",
        "Greedy": "#f59e0b",
        "SQA": "#8b5cf6",
        "QI-EA": "#ef4444",
    }

    with chart_col1:
        st.markdown("**Optimality Gap vs Reference Baseline (%)**")
        fig_gap = px.bar(
            bench_df,
            x="instance", y="gap_to_ref_pct", color="algorithm",
            barmode="group",
            labels={"gap_to_ref_pct": "Optimality Gap (%)", "instance": "Problem Instance", "algorithm": "Algorithm"},
            color_discrete_map=_algo_colors,
        )
        fig_gap.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gap, use_container_width=True)

    with chart_col2:
        st.markdown("**Execution Runtime Scaling (Log Scale)**")
        fig_time = px.bar(
            bench_df,
            x="instance", y="mean_runtime_s", color="algorithm",
            barmode="group", log_y=True,
            labels={"mean_runtime_s": "Runtime (s, log scale)", "instance": "Problem Instance", "algorithm": "Algorithm"},
            color_discrete_map=_algo_colors,
        )
        fig_time.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Wilcoxon Signed-Rank Statistical Significance Matrix (from CSV)
    # -----------------------------------------------------------------------
    st.subheader("🔬 Statistical Hypothesis Testing (Wilcoxon Signed-Rank)")
    st.caption(
        "Paired Wilcoxon signed-rank tests (Pratt zero handling) across 10 random seeds "
        "with Holm-Bonferroni FWER step-down correction."
    )

    if stat_df.empty:
        st.warning("Statistical tests CSV (`results/statistical_tests.csv`) not found.")
    else:
        disp_stat = stat_df.copy()
        disp_stat["Mean Objective Delta ($)"] = disp_stat["mean_diff"].apply(
            lambda x: f"+${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"
        )
        disp_stat["Raw p-value"] = disp_stat["wilcoxon_p"].apply(lambda x: f"{x:.4f}")
        disp_stat["Holm p-value"] = disp_stat["wilcoxon_p_holm"].apply(lambda x: f"{x:.4f}")
        disp_stat["Significant (α=0.05)"] = disp_stat["is_significant_005"].apply(
            lambda x: "✅ Yes" if x else "❌ No"
        )
        st.dataframe(
            disp_stat[["instance", "comparison", "Mean Objective Delta ($)",
                        "Raw p-value", "Holm p-value", "Significant (α=0.05)"]].rename(
                columns={"instance": "Instance", "comparison": "Pairwise Comparison"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.error(
        "**Key Empirical Finding**: Under identical evaluation budgets, "
        "**QI-EA consistently loses to classical GA and SQA across all instances** ($p < 0.01$). "
        "Deterministic MILP (SciPy HiGHS) proves optimal across all instances in $<3$ seconds."
    )
