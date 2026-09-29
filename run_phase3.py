"""
QFleet Phase 3 — Full Pipeline: benchmark + analysis.
Runs the benchmark harness, then feeds results into analysis.py
to generate CSVs and charts.
"""
import sys
import pathlib

_ROOT = pathlib.Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.run_benchmark import run_full_benchmark
from benchmark.analysis import analyze_benchmark_results

if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING PHASE 3 FULL PIPELINE: BENCHMARK + ANALYSIS")
    print("=" * 80)

    # Step 1: Execute benchmark harness
    benchmark_data = run_full_benchmark()

    # Step 2: Run analysis (CSVs, statistical tests, charts)
    results = analyze_benchmark_results(benchmark_data)

    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE — Summary")
    print("=" * 80)

    df_summary = results["summary"]
    for inst in ["Small", "Medium", "Large"]:
        inst_data = df_summary[df_summary["instance"] == inst]
        print(f"\n--- {inst} Instance ---")
        ref_type = inst_data.iloc[0]["reference_type"]
        ref_obj = inst_data.iloc[0]["reference_obj"]
        print(f"  Reference: {ref_type} = ${ref_obj:,.2f}")
        for _, row in inst_data.iterrows():
            print(f"  {row['algorithm']:12s}  mean={row['mean_objective']:>14,.2f}  gap={row['gap_to_ref_pct']:>7.2f}%  "
                  f"runtime={row['mean_runtime_s']:>7.3f}s  feasible={row['feasibility_rate_pct']:.0f}%")

    print(f"\nStatistical tests saved to: results/statistical_tests.csv")
    print(f"Charts saved to: docs/figures/")
    print("Done.")
