"""
QFleet Phase 3 — Benchmark Analysis & Statistical Reporting
===========================================================
Analyzes benchmark results across Small, Medium, and Large instances:
1. Calculates summary metrics per instance and algorithm:
   - Mean, Std, Best (Min), Worst (Max), Mean Runtime (s), Feasibility (%), Gap vs Reference (%)
2. Performs statistical significance testing:
   - Paired Wilcoxon signed-rank test (with Pratt/Wilcox zero handling).
   - Independent Mann-Whitney U test.
   - Holm-Bonferroni FWER step-down correction.
3. Exports benchmark_results.csv.
4. Generates 3 publication-ready PNG charts:
   - Objective gap comparison across instances.
   - Runtime scaling comparison across instances.
   - Convergence trajectory comparison on Medium instance.
"""

from __future__ import annotations

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
import db as database


def analyze_benchmark_results(
    benchmark_data: dict,
    output_dir: pathlib.Path = config.RESULTS_DIR,
    figures_dir: pathlib.Path | None = None,
) -> dict:
    """
    Computes statistical summaries, hypothesis tests, and visualisations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if figures_dir is None:
        figures_dir = _ROOT / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_runs = pd.DataFrame(benchmark_data["raw_runs"])

    # -----------------------------------------------------------------------
    # 1. Compute Reference Ground Truth / Best-Known per Instance
    # -----------------------------------------------------------------------
    reference_objectives = {}
    reference_labels = {}

    for inst_name, inst_meta in benchmark_data["instances"].items():
        if inst_meta.get("exact_ground_truth") is not None:
            reference_objectives[inst_name] = inst_meta["exact_ground_truth"]["objective"]
            reference_labels[inst_name] = "Exact Brute-Force Ground Truth (J*)"
        else:
            # Check MILP result
            milp_runs = df_runs[(df_runs["instance"] == inst_name) & (df_runs["algorithm"] == "MILP_HiGHS")]
            if len(milp_runs) > 0 and milp_runs.iloc[0]["is_feasible"]:
                reference_objectives[inst_name] = float(milp_runs.iloc[0]["objective"])
                reference_labels[inst_name] = "Optimal MILP (J*)"
            else:
                # Best known feasible solution
                feas_runs = df_runs[(df_runs["instance"] == inst_name) & (df_runs["is_feasible"] == True)]
                if len(feas_runs) > 0:
                    reference_objectives[inst_name] = float(feas_runs["objective"].min())
                    reference_labels[inst_name] = "Best-Known Feasible Solution"
                else:
                    reference_objectives[inst_name] = float(df_runs[df_runs["instance"] == inst_name]["objective"].min())
                    reference_labels[inst_name] = "Best-Known Solution"

    # -----------------------------------------------------------------------
    # 2. Aggregated Summary Statistics
    # -----------------------------------------------------------------------
    summary_rows = []

    for (inst, algo), group in df_runs.groupby(["instance", "algorithm"], sort=False):
        ref_obj = reference_objectives[inst]
        objs = group["objective"].values
        runtimes = group["runtime_s"].values
        feas_rate = (group["is_feasible"].sum() / len(group)) * 100.0

        mean_obj = float(np.mean(objs))
        std_obj = float(np.std(objs, ddof=1)) if len(objs) > 1 else 0.0
        best_obj = float(np.min(objs))
        worst_obj = float(np.max(objs))
        mean_runtime = float(np.mean(runtimes))
        mean_fuel_cost = float(group["fuel_cost_usd"].mean())
        mean_co2 = float(group["co2_t"].mean())

        gap_pct = ((mean_obj - ref_obj) / ref_obj) * 100.0

        summary_rows.append({
            "instance": inst,
            "algorithm": algo,
            "n_runs": len(group),
            "reference_obj": ref_obj,
            "reference_type": reference_labels[inst],
            "mean_objective": mean_obj,
            "std_objective": std_obj,
            "best_objective": best_obj,
            "worst_objective": worst_obj,
            "gap_to_ref_pct": gap_pct,
            "mean_fuel_cost_usd": mean_fuel_cost,
            "mean_co2_wtw_t": mean_co2,
            "mean_runtime_s": mean_runtime,
            "feasibility_rate_pct": feas_rate,
            "surrogate_build_time_s": float(group["surrogate_build_time_s"].iloc[0]),
        })

    df_summary = pd.DataFrame(summary_rows)

    # Export CSVs
    csv_path_results = output_dir / "benchmark_results.csv"
    csv_path_data = config.DATA_DIR / "benchmark_results.csv"
    df_summary.to_csv(csv_path_results, index=False)
    df_summary.to_csv(csv_path_data, index=False)

    # -----------------------------------------------------------------------
    # 3. Statistical Hypothesis Testing across 10 Seeds (Wilcoxon & Mann-Whitney)
    # -----------------------------------------------------------------------
    test_results = []
    comparisons = [("QI-EA", "GA"), ("QI-EA", "SQA"), ("GA", "SQA")]

    for inst in ["Small", "Medium", "Large"]:
        raw_p_values = []
        comp_records = []

        for algo_a, algo_b in comparisons:
            runs_a = df_runs[(df_runs["instance"] == inst) & (df_runs["algorithm"] == algo_a)].sort_values("seed")
            runs_b = df_runs[(df_runs["instance"] == inst) & (df_runs["algorithm"] == algo_b)].sort_values("seed")

            vals_a = runs_a["objective"].values
            vals_b = runs_b["objective"].values

            # 1. Paired Wilcoxon Signed-Rank Test (handles all-zero differences)
            diffs = vals_a - vals_b
            if np.all(diffs == 0):
                w_stat, w_p = 0.0, 1.0
            else:
                try:
                    w_res = stats.wilcoxon(vals_a, vals_b, zero_method="pratt", alternative="two-sided")
                    w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
                except Exception:
                    w_stat, w_p = 0.0, 1.0

            # 2. Independent Mann-Whitney U Test
            try:
                mw_res = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                mw_stat, mw_p = float(mw_res.statistic), float(mw_res.pvalue)
            except Exception:
                mw_stat, mw_p = 0.0, 1.0

            raw_p_values.append(w_p)
            comp_records.append({
                "instance": inst,
                "comparison": f"{algo_a} vs {algo_b}",
                "mean_diff": float(np.mean(diffs)),
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "mannwhitney_stat": mw_stat,
                "mannwhitney_p": mw_p,
            })

        # Apply Holm-Bonferroni step-down correction across the 3 comparisons
        holm_p = _holm_bonferroni_correction(raw_p_values)
        for i, rec in enumerate(comp_records):
            rec["wilcoxon_p_holm"] = holm_p[i]
            rec["is_significant_005"] = bool(holm_p[i] < 0.05)
            test_results.append(rec)

    df_tests = pd.DataFrame(test_results)
    df_tests.to_csv(output_dir / "statistical_tests.csv", index=False)

    # -----------------------------------------------------------------------
    # 4. Generate Visualizations (3 PNG Charts)
    # -----------------------------------------------------------------------
    _plot_objective_gap_comparison(df_summary, figures_dir / "objective_gap_comparison.png")
    _plot_runtime_scaling(df_summary, figures_dir / "runtime_scaling.png")
    if "convergence_histories" in benchmark_data:
        _plot_convergence_medium(benchmark_data["convergence_histories"], figures_dir / "convergence_medium.png")

    return {
        "summary": df_summary,
        "statistical_tests": df_tests,
        "raw_runs": df_runs,
        "reference_objectives": reference_objectives,
    }


def _holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Calculates Holm-Bonferroni adjusted p-values."""
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        adj_p = min(1.0, p_val * multiplier)
        adj_p = max(adj_p, running_max)
        running_max = adj_p
        adjusted[orig_idx] = adj_p

    return adjusted


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------

def _plot_objective_gap_comparison(df_summary: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Generates bar chart comparing optimality gaps across instances."""
    plt.figure(figsize=(9, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    instances = ["Small", "Medium", "Large"]
    algorithms = ["MILP_HiGHS", "Greedy", "GA", "SQA", "QI-EA"]
    palette = {"MILP_HiGHS": "#1b9e77", "Greedy": "#d95f02", "GA": "#7570b3", "SQA": "#e7298a", "QI-EA": "#66a61e"}

    x = np.arange(len(instances))
    width = 0.16

    for idx, algo in enumerate(algorithms):
        gaps = []
        for inst in instances:
            sub = df_summary[(df_summary["instance"] == inst) & (df_summary["algorithm"] == algo)]
            gaps.append(float(sub["gap_to_ref_pct"].iloc[0]) if len(sub) > 0 else 0.0)

        plt.bar(x + idx * width - (len(algorithms) - 1) * width / 2, gaps, width, label=algo, color=palette.get(algo, "#333333"), edgecolor="black", linewidth=0.6)

    plt.xlabel("Benchmark Instance", fontsize=11, fontweight="bold")
    plt.ylabel("Gap to Reference Solution (%)", fontsize=11, fontweight="bold")
    plt.title("Optimality Gap Comparison across Fleet Scales", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(x, ["Small (4V, 6R)", "Medium (8V, 15R)", "Large (15V, 30R)"], fontsize=10)
    plt.legend(title="Algorithm", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_runtime_scaling(df_summary: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Generates log-scale runtime scaling curve across problem sizes."""
    plt.figure(figsize=(9, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    instances = ["Small", "Medium", "Large"]
    routes = [6, 15, 30]
    algorithms = ["MILP_HiGHS", "Greedy", "GA", "SQA", "QI-EA"]
    markers = {"MILP_HiGHS": "o", "Greedy": "s", "GA": "^", "SQA": "d", "QI-EA": "v"}
    palette = {"MILP_HiGHS": "#1b9e77", "Greedy": "#d95f02", "GA": "#7570b3", "SQA": "#e7298a", "QI-EA": "#66a61e"}

    for algo in algorithms:
        runtimes = []
        for inst in instances:
            sub = df_summary[(df_summary["instance"] == inst) & (df_summary["algorithm"] == algo)]
            runtimes.append(float(sub["mean_runtime_s"].iloc[0]) if len(sub) > 0 else 1e-4)

        plt.plot(routes, runtimes, marker=markers.get(algo, "o"), linewidth=1.8, markersize=7, label=algo, color=palette.get(algo, "#333333"))

    plt.yscale("log")
    plt.xlabel("Number of Routes (|R|)", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Runtime (seconds, log scale)", fontsize=11, fontweight="bold")
    plt.title("Algorithm Runtime Scaling across Fleet Scales", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(routes, ["6 (Small)", "15 (Medium)", "30 (Large)"], fontsize=10)
    plt.legend(title="Algorithm", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_convergence_medium(histories: dict, out_path: pathlib.Path) -> None:
    """Plots best-so-far convergence histories for the Medium instance."""
    plt.figure(figsize=(9, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    palette = {"GA": "#7570b3", "SQA": "#e7298a", "QI-EA": "#66a61e"}

    for algo, hist in histories.items():
        if hist and len(hist) > 0:
            eval_steps = np.linspace(0, 3000, len(hist))
            plt.plot(eval_steps, [x / 1e6 for x in hist], label=algo, linewidth=2.0, color=palette.get(algo, "#333333"))

    plt.xlabel("Evaluations Budget", fontsize=11, fontweight="bold")
    plt.ylabel("Best-so-Far Objective ($ Millions)", fontsize=11, fontweight="bold")
    plt.title("Convergence Trajectory Comparison (Medium Instance, 8V 15R)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(title="Metaheuristic", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
