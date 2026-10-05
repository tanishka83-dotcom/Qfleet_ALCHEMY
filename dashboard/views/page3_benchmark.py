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
from dashboard.background import configure_plotly_theme
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


def add_run_metrics(
    runs: pd.DataFrame,
    benchmark_summary: pd.DataFrame,
    scenario_names: dict[int, str],
) -> pd.DataFrame:
    """Join canonical stored runs to references and derive per-run optimality gap."""
    frame = runs.copy()
    scenario_to_instance = {
        int(scenario_id): instance
        for instance, scenario_name in _INSTANCE_SCENARIO_MAP.items()
        for scenario_id, name in scenario_names.items()
        if name == scenario_name
    }
    frame["instance"] = frame["scenario_id"].map(scenario_to_instance)
    references = benchmark_summary.set_index("instance")["reference_obj"].to_dict()
    frame["reference_obj"] = frame["instance"].map(references)
    frame["gap_to_ref_pct"] = (
        (pd.to_numeric(frame["objective"], errors="coerce") / frame["reference_obj"] - 1.0)
        * 100.0
    )
    return frame


def filter_benchmark_data(
    summary: pd.DataFrame,
    runs: pd.DataFrame,
    algorithms: list[str] | None = None,
    instances: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply algorithm and instance filters to aggregate and per-seed sources."""
    filtered_summary = summary.copy()
    filtered_runs = runs.copy()
    for column, values in (("algorithm", algorithms), ("instance", instances)):
        if values is not None:
            filtered_summary = filtered_summary[
                filtered_summary[column].isin(values)
            ]
    for column, values in (
        ("algo_normalized", algorithms),
        ("instance", instances),
    ):
        if values is not None:
            filtered_runs = filtered_runs[filtered_runs[column].isin(values)]
    return (
        filtered_summary.reset_index(drop=True),
        filtered_runs.reset_index(drop=True),
    )


def _reset_filters() -> None:
    for key in ("p3_algorithms", "p3_instances", "p3_metric"):
        st.session_state.pop(key, None)


def render() -> None:
    configure_plotly_theme()
    st.title("Benchmark Results")
    st.caption("Filter stored benchmark runs and compare objective quality and runtime across instances.")

    # -----------------------------------------------------------------------
    # Ingest data
    # -----------------------------------------------------------------------
    summary_df = load_benchmark_csv()
    stat_df = load_statistical_tests()
    audit = load_optimization_runs_audit()
    scenarios_df = load_scenarios()
    scenario_names = {
        int(row["id"]): str(row["name"])
        for _, row in scenarios_df.iterrows()
    }
    runs_df = add_run_metrics(
        audit["canonical_benchmark_df"], summary_df, scenario_names
    )
    instance_options = [
        instance for instance in ["Small", "Medium", "Large"]
        if instance in set(runs_df["instance"].dropna())
    ]
    algorithm_options = sorted(runs_df["algo_normalized"].dropna().unique().tolist())
    metric_options = {
        "Objective": ("objective", "Objective (USD)", True),
        "Runtime": ("runtime_s", "Runtime (s)", True),
        "Gap": ("gap_to_ref_pct", "Gap to reference (%)", True),
    }
    for key, values in (
        ("p3_instances", instance_options),
        ("p3_algorithms", algorithm_options),
    ):
        if key not in st.session_state:
            st.session_state[key] = values
        else:
            st.session_state[key] = [value for value in st.session_state[key] if value in values]

    st.sidebar.subheader("Benchmark filters")
    st.sidebar.button("Reset filters", key="p3_reset_filters", on_click=_reset_filters)
    st.sidebar.multiselect("Instance", instance_options, key="p3_instances")
    st.sidebar.multiselect("Algorithm", algorithm_options, key="p3_algorithms")
    st.sidebar.selectbox("Metric", list(metric_options), key="p3_metric")
    metric_col, metric_label, ascending = metric_options[st.session_state["p3_metric"]]

    filtered_summary, filtered_runs = filter_benchmark_data(
        summary_df,
        runs_df,
        algorithms=st.session_state["p3_algorithms"],
        instances=st.session_state["p3_instances"],
    )
    if filtered_runs.empty:
        st.metric("Stored runs", 0)
        st.info("No stored benchmark runs match these filters. Update the selection or reset filters.")
        return

    metric_values = pd.to_numeric(filtered_runs[metric_col], errors="coerce")
    best_index = metric_values.idxmin()
    best_run = filtered_runs.loc[best_index]
    kpis = st.columns(4)
    kpis[0].metric("Stored runs", len(filtered_runs))
    kpis[1].metric("Algorithms", filtered_runs["algo_normalized"].nunique())
    kpis[2].metric("Instances", filtered_runs["instance"].nunique())
    kpis[3].metric(f"Best {metric_label}", f"{metric_values.min():,.3f}")
    st.caption(
        f"Best filtered result: {best_run['algo_normalized']} on "
        f"{best_run['instance']} ({metric_values.loc[best_index]:,.3f} {metric_label})."
    )

    ranking = filtered_runs.sort_values(metric_col, ascending=ascending).copy()
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    ranking["Best"] = ranking["Rank"] == 1
    ranking_display = ranking[[
        "Rank", "Best", "instance", "algo_normalized", "seed", "objective",
        "runtime_s", "gap_to_ref_pct",
    ]].rename(columns={
        "instance": "Instance",
        "algo_normalized": "Algorithm",
        "seed": "Seed",
        "objective": "Objective (USD)",
        "runtime_s": "Runtime (s)",
        "gap_to_ref_pct": "Gap (%)",
    })
    ranking_style = ranking_display.style.apply(
        lambda row: [
            "background-color: rgba(34, 211, 238, .18); font-weight: 700"
            if row["Best"] else ""
            for _ in row
        ], axis=1
    )
    st.download_button(
        "Download filtered ranking CSV",
        ranking_display.to_csv(index=False).encode("utf-8"),
        file_name="qfleet_p3_benchmark_ranking.csv",
        mime="text/csv",
        key="p3_ranking_csv",
    )
    st.dataframe(ranking_style, use_container_width=True, hide_index=True, key="p3_ranking_table")

    box_fig = px.box(
        filtered_runs,
        x="algo_normalized",
        y=metric_col,
        color="algo_normalized",
        points="all",
        hover_data=["instance", "seed", "objective", "runtime_s", "gap_to_ref_pct"],
        labels={
            "algo_normalized": "Algorithm",
            metric_col: metric_label,
        },
        title=f"Per-seed {metric_label} distribution",
    )
    box_fig.update_layout(clickmode="event+select", height=400)
    st.plotly_chart(box_fig, use_container_width=True, key="p3_seed_boxplot")

    if not filtered_summary.empty:
        scale_fig = px.line(
            filtered_summary,
            x="instance",
            y="mean_runtime_s",
            color="algorithm",
            markers=True,
            hover_data=["n_runs", "feasibility_rate_pct", "gap_to_ref_pct"],
            labels={
                "instance": "Benchmark instance",
                "mean_runtime_s": "Mean runtime (s)",
                "algorithm": "Algorithm",
            },
            title="Stored mean runtime by instance",
            category_orders={"instance": instance_options},
        )
        scale_fig.update_layout(clickmode="event+select", height=400)
        st.plotly_chart(scale_fig, use_container_width=True, key="p3_scalability")

    # Derive instance-level reference info from CSV
    instances = filtered_summary.groupby("instance").first().reset_index()

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

    instance_names = filtered_summary["instance"].unique().tolist()
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
            inst_data = filtered_summary[filtered_summary["instance"] == inst]
            ref_row = inst_data.iloc[0]
            st.caption(
                f"{inst} Instance — Reference: {ref_row['reference_type']} "
                f"($J^* = ${ref_row['reference_obj']:,.0f})"
            )
            st.dataframe(_format_table(inst_data), use_container_width=True, hide_index=True)

    with tabs[-1]:
        st.dataframe(_format_table(filtered_summary), use_container_width=True, hide_index=True)

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
            filtered_summary,
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
            filtered_summary,
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
