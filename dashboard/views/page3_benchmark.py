"""
QFleet Dashboard — Page 3: Benchmark Results
===========================================
Displays Phase 3 comprehensive benchmark results:
- Dynamic table from data/benchmark_results.csv
- One instance per size (Small, Medium, Large)
- Search-space coverage on Small (4^6 = 4,096 combinations)
- MILP status, gap, and timeout tracking (120s limit)
- Reference type distinction (proven exact optimum J* vs best-known MILP)
- Feasibility definition (quotas + binding emissions cap)
- Interactive charts: Gap, Runtime, Convergence
- Wilcoxon Signed-Rank matrix with Holm-Bonferroni correction
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dashboard.data_loader import load_benchmark_csv, load_optimization_runs_audit


def render() -> None:
    st.title("🏆 Benchmark Results & Statistical Rigor")
    st.caption("Empirical performance comparison of 5 optimization algorithms across Small, Medium, and Large benchmark instances.")

    # -----------------------------------------------------------------------
    # Ingest Summary CSV
    # -----------------------------------------------------------------------
    bench_df = load_benchmark_csv()

    # -----------------------------------------------------------------------
    # Instance Context & Feasibility Rules
    # -----------------------------------------------------------------------
    with st.expander("📌 Benchmark Instance Specifications & Feasibility Protocol", expanded=True):
        col_s, col_m, col_l = st.columns(3)
        with col_s:
            st.markdown("### Small (4V, 6R)")
            st.markdown("- **Search Space**: $4^6 = 4,096$ vessel combos (100% brute-force covered)")
            st.markdown("- **Emissions Cap**: 35,000 t CO₂eq (*Unconstrained*)")
            st.markdown("- **Reference Type**: **Proven Exact Optimum ($J^* = \\$5,851,877$)**")
            st.markdown("- **Eval Budget**: 1,000 evaluations")
        with col_m:
            st.markdown("### Medium (8V, 15R)")
            st.markdown("- **Search Space**: $8^{15} \\approx 3.52 \\times 10^{13}$ vessel combos")
            st.markdown("- **Emissions Cap**: 58,000 t CO₂eq (*Binding Constraint*)")
            st.markdown("- **Reference Type**: **Optimal MILP ($J^* = \\$15,781,434$)**")
            st.markdown("- **Eval Budget**: 5,000 evaluations")
        with col_l:
            st.markdown("### Large (15V, 30R)")
            st.markdown("- **Search Space**: $15^{30} \\approx 1.92 \\times 10^{35}$ vessel combos")
            st.markdown("- **Emissions Cap**: 122,000 t CO₂eq (*Binding Constraint*)")
            st.markdown("- **Reference Type**: **Optimal MILP ($J^* = \\$35,705,171$)**")
            st.markdown("- **Eval Budget**: 10,000 evaluations")

        st.info(
            "**Formal Feasibility Definition**: A dispatch plan $x$ is feasible **iff**:\n"
            "1. **Coupling Quotas**: Each vessel $v$ does not exceed its maximum route limit $\\sum_r x_{rv} \\le q_v$.\n"
            "2. **Emissions Cap**: Total Well-to-Wake CO₂ emissions satisfy $\\sum_r E_r(x) \\le E_{\\text{max}}$."
        )

    # -----------------------------------------------------------------------
    # Filter by Instance
    # -----------------------------------------------------------------------
    st.subheader("📊 Performance Summary Table (from benchmark_results.csv)")
    
    inst_tab1, inst_tab2, inst_tab3, inst_tab_all = st.tabs(["Small Instance", "Medium Instance", "Large Instance", "All Combined"])

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

    with inst_tab1:
        st.caption("Small Instance (4 Vessels, 6 Routes) — Reference: Exact Brute-Force Ground Truth ($J^* = $5,851,877)")
        st.dataframe(_format_table(bench_df[bench_df["instance"] == "Small"]), use_container_width=True, hide_index=True)
        st.caption("*Note: On Small, the brute-force lower bound is unconstrained; with coupling penalties, all heuristics achieve equal valid performance ($7.58M).")

    with inst_tab2:
        st.caption("Medium Instance (8 Vessels, 15 Routes) — Reference: Optimal MILP ($J^* = $15,781,434)")
        st.dataframe(_format_table(bench_df[bench_df["instance"] == "Medium"]), use_container_width=True, hide_index=True)
        st.warning("⚠️ **Greedy and QI-EA are 0% feasible** on Medium due to the binding 58,000 t CO₂ cap. GA achieves 90% feasibility with a 1.79% gap.")

    with inst_tab3:
        st.caption("Large Instance (15 Vessels, 30 Routes) — Reference: Optimal MILP ($J^* = $35,705,171)")
        st.dataframe(_format_table(bench_df[bench_df["instance"] == "Large"]), use_container_width=True, hide_index=True)
        st.warning("⚠️ **At Large scale**, MILP is 100% feasible. GA achieves 10% feasibility with 2.01% gap. SQA and QI-EA fail to satisfy the binding emissions cap.")

    with inst_tab_all:
        st.dataframe(_format_table(bench_df), use_container_width=True, hide_index=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Interactive Visualizations
    # -----------------------------------------------------------------------
    st.subheader("📈 Visual Comparison Charts")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Optimality Gap vs Reference Baseline (%)**")
        fig_gap = px.bar(
            bench_df,
            x="instance",
            y="gap_to_ref_pct",
            color="algorithm",
            barmode="group",
            labels={"gap_to_ref_pct": "Optimality Gap (%)", "instance": "Problem Instance", "algorithm": "Algorithm"},
            color_discrete_map={
                "MILP_HiGHS": "#10b981",
                "GA": "#3b82f6",
                "Greedy": "#f59e0b",
                "SQA": "#8b5cf6",
                "QI-EA": "#ef4444",
            },
        )
        fig_gap.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gap, use_container_width=True)

    with chart_col2:
        st.markdown("**Execution Runtime Scaling (Log Scale)**")
        fig_time = px.bar(
            bench_df,
            x="instance",
            y="mean_runtime_s",
            color="algorithm",
            barmode="group",
            log_y=True,
            labels={"mean_runtime_s": "Runtime (s, log scale)", "instance": "Problem Instance", "algorithm": "Algorithm"},
            color_discrete_map={
                "MILP_HiGHS": "#10b981",
                "GA": "#3b82f6",
                "Greedy": "#f59e0b",
                "SQA": "#8b5cf6",
                "QI-EA": "#ef4444",
            },
        )
        fig_time.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Wilcoxon Signed-Rank Statistical Significance Matrix
    # -----------------------------------------------------------------------
    st.subheader("🔬 Statistical Hypothesis Testing (Wilcoxon Signed-Rank)")
    st.caption("Paired Wilcoxon signed-rank tests (Pratt zero handling) across 10 random seeds with Holm-Bonferroni FWER step-down correction.")

    wilcoxon_data = [
        {"Instance": "Small", "Pairwise Comparison": "QI-EA vs GA", "Mean Objective Delta ($)": "+$316,780", "Raw p-value": "0.0020", "Holm p-value": "0.0059", "Statistically Significant (α=0.05)": "✅ Yes (GA beats QI-EA)"},
        {"Instance": "Small", "Pairwise Comparison": "QI-EA vs SQA", "Mean Objective Delta ($)": "+$312,075", "Raw p-value": "0.0020", "Holm p-value": "0.0059", "Statistically Significant (α=0.05)": "✅ Yes (SQA beats QI-EA)"},
        {"Instance": "Small", "Pairwise Comparison": "GA vs SQA", "Mean Objective Delta ($)": "-$4,704", "Raw p-value": "0.0078", "Holm p-value": "0.0078", "Statistically Significant (α=0.05)": "✅ Yes (GA beats SQA)"},
        {"Instance": "Medium", "Pairwise Comparison": "QI-EA vs GA", "Mean Objective Delta ($)": "+$6,441,305", "Raw p-value": "0.0020", "Holm p-value": "0.0059", "Statistically Significant (α=0.05)": "✅ Yes (GA beats QI-EA)"},
        {"Instance": "Medium", "Pairwise Comparison": "QI-EA vs SQA", "Mean Objective Delta ($)": "+$4,698,413", "Raw p-value": "0.0020", "Holm p-value": "0.0059", "Statistically Significant (α=0.05)": "✅ Yes (SQA beats QI-EA)"},
        {"Instance": "Medium", "Pairwise Comparison": "GA vs SQA", "Mean Objective Delta ($)": "-$1,742,892", "Raw p-value": "0.0039", "Holm p-value": "0.0059", "Statistically Significant (α=0.05)": "✅ Yes (GA beats SQA)"},
        {"Instance": "Large", "Pairwise Comparison": "QI-EA vs GA", "Mean Objective Delta ($)": "+$16,419,018", "Raw p-value": "0.0020", "Holm p-value": "0.0059", "Statistically Significant (α=0.05)": "✅ Yes (GA beats QI-EA)"},
        {"Instance": "Large", "Pairwise Comparison": "QI-EA vs SQA", "Mean Objective Delta ($)": "+$8,427,587", "Raw p-value": "0.0020", "Holm p-value": "0.0059", "Statistically Significant (α=0.05)": "✅ Yes (SQA beats QI-EA)"},
        {"Instance": "Large", "Pairwise Comparison": "GA vs SQA", "Mean Objective Delta ($)": "-$7,991,431", "Raw p-value": "0.0020", "Holm p-value": "0.0059", "Statistically Significant (α=0.05)": "✅ Yes (GA beats SQA)"},
    ]
    st.dataframe(pd.DataFrame(wilcoxon_data), use_container_width=True, hide_index=True)

    st.error(
        "**Key Empirical Finding**: Under identical evaluation budgets (1,000 / 5,000 / 10,000 evals), "
        "**QI-EA consistently loses to classical GA and SQA across all instances** ($p < 0.01$). "
        "Deterministic MILP (SciPy HiGHS) proves optimal across all instances in $<3$ seconds."
    )
