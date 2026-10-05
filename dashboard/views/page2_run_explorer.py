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

import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from dashboard.background import configure_plotly_theme
from dashboard.data_loader import (
    load_optimization_runs_audit,
    load_scenario_detail,
    load_scenarios,
)


def filter_canonical_runs(
    runs: pd.DataFrame,
    scenario_ids: list[int] | None = None,
    algorithms: list[str] | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Filter the canonical stored-run frame; empty selections match nothing."""
    filtered = runs.copy()
    for column, values in (
        ("scenario_id", scenario_ids),
        ("algo_normalized", algorithms),
        ("seed", seeds),
    ):
        if values is not None:
            filtered = filtered[filtered[column].isin(values)]
    return filtered.reset_index(drop=True)


def recorded_convergence_point(run: pd.Series) -> pd.DataFrame:
    """Return the one stored objective/convergence point, never an invented curve."""
    iteration = run.get("converged_at_iter")
    objective = run.get("objective")
    if pd.isna(iteration) or pd.isna(objective):
        return pd.DataFrame(columns=["Iteration", "Objective (USD)"])
    return pd.DataFrame([{
        "Iteration": int(iteration),
        "Objective (USD)": float(objective),
    }])


def _compact_currency(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.2f}k"
    return f"${value:,.2f}"


def _reset_filters() -> None:
    for key in (
        "p2_scenarios", "p2_algorithms", "p2_seeds", "p2_selected_run",
        "p2_compare_left", "p2_compare_right",
    ):
        st.session_state.pop(key, None)


def render() -> None:
    configure_plotly_theme()
    st.title("Optimization Runs")
    st.caption("Explore canonical stored runs. No optimizer is launched from this page.")

    audit = load_optimization_runs_audit()
    canonical = audit["canonical_benchmark_df"].copy()
    scenarios = load_scenarios()
    scenario_names = {
        int(row["id"]): str(row["name"])
        for _, row in scenarios.iterrows()
    }
    if canonical.empty:
        st.info("No canonical optimization runs are stored.")
        return

    scenario_options = sorted(canonical["scenario_id"].dropna().astype(int).unique())
    algorithm_options = sorted(canonical["algo_normalized"].dropna().astype(str).unique())
    seed_options = sorted(canonical["seed"].dropna().astype(int).unique())
    defaults = {
        "p2_scenarios": scenario_options,
        "p2_algorithms": algorithm_options,
        "p2_seeds": seed_options,
    }
    valid_options = {
        "p2_scenarios": set(scenario_options),
        "p2_algorithms": set(algorithm_options),
        "p2_seeds": set(seed_options),
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
        else:
            st.session_state[key] = [
                value for value in st.session_state[key]
                if value in valid_options[key]
            ]

    st.sidebar.subheader("Run filters")
    st.sidebar.button(
        "Reset filters", key="p2_reset_filters", on_click=_reset_filters,
        use_container_width=True,
    )
    st.sidebar.multiselect(
        "Scenario", scenario_options,
        format_func=lambda scenario_id: scenario_names.get(scenario_id, str(scenario_id)),
        key="p2_scenarios",
    )
    st.sidebar.multiselect("Algorithm", algorithm_options, key="p2_algorithms")
    st.sidebar.multiselect("Seed", seed_options, key="p2_seeds")

    filtered = filter_canonical_runs(
        canonical,
        scenario_ids=st.session_state["p2_scenarios"],
        algorithms=st.session_state["p2_algorithms"],
        seeds=st.session_state["p2_seeds"],
    )
    title = "Canonical runs"
    st.subheader(title)
    if filtered.empty:
        st.metric("Runs", 0)
        st.info("No stored runs match these filters. Choose values or reset filters.")
        return

    valid_run_ids = filtered["id"].astype(int).tolist()
    filtered = filtered.sort_values("id", ascending=False).reset_index(drop=True)
    valid_run_ids = filtered["id"].astype(int).tolist()
    if st.session_state.get("p2_selected_run") not in valid_run_ids:
        st.session_state["p2_selected_run"] = valid_run_ids[0]

    run_options = filtered["id"].astype(int).tolist()
    st.sidebar.selectbox(
        "Selected run",
        run_options,
        format_func=lambda run_id: _run_label(
            filtered[filtered["id"] == run_id].iloc[0], scenario_names
        ),
        key="p2_selected_run",
    )
    selected_row = filtered[
        filtered["id"] == st.session_state["p2_selected_run"]
    ].iloc[0]

    objectives = pd.to_numeric(filtered["objective"], errors="coerce")
    runtimes = pd.to_numeric(filtered["runtime_s"], errors="coerce")
    kpis = st.columns(4)
    kpis[0].metric("Filtered runs", len(filtered))
    kpis[1].metric("Best objective", _compact_currency(objectives.min()))
    kpis[2].metric("Median runtime", f"{runtimes.median():.3f} s")
    kpis[3].metric("Selected run", f"#{int(selected_row['id'])}")
    best = filtered.loc[objectives.idxmin()]
    st.caption(
        f"{len(filtered)} canonical runs match; lowest recorded objective is "
        f"{_compact_currency(float(best['objective']))} from "
        f"{best['algo_normalized']} in "
        f"{scenario_names.get(int(best['scenario_id']), best['scenario_id'])}."
    )

    table_columns = [
        "id", "scenario_id", "algo_normalized", "seed", "objective",
        "fuel_t", "co2_t", "runtime_s", "converged_at_iter", "run_at",
    ]
    display = filtered[table_columns].rename(columns={
        "id": "Run ID",
        "scenario_id": "Scenario ID",
        "algo_normalized": "Algorithm",
        "seed": "Seed",
        "objective": "Objective (USD)",
        "fuel_t": "Fuel (t)",
        "co2_t": "WTW CO₂eq (t)",
        "runtime_s": "Runtime (s)",
        "converged_at_iter": "Converged at iteration",
        "run_at": "Recorded at",
    })
    st.download_button(
        "Download filtered runs CSV",
        display.to_csv(index=False).encode("utf-8"),
        file_name="qfleet_p2_runs.csv",
        mime="text/csv",
        key="p2_runs_csv",
    )
    st.dataframe(display, use_container_width=True, hide_index=True, key="p2_runs_table")

    left, right = st.columns(2)
    with left:
        st.subheader("Recorded convergence")
        point = recorded_convergence_point(selected_row)
        if point.empty:
            st.info("This run has no stored convergence point.")
        else:
            fig = go.Figure(go.Scatter(
                x=point["Iteration"],
                y=point["Objective (USD)"],
                mode="markers",
                name="Stored terminal objective",
                marker={"size": 12, "color": "#22d3ee"},
                customdata=[[int(selected_row["id"]), str(selected_row["algorithm"])]],
                hovertemplate=(
                    "Run %{customdata[0]} (%{customdata[1]})<br>"
                    "Iteration %{x}<br>Objective $%{y:,.2f}<extra></extra>"
                ),
            ))
            fig.update_layout(
                xaxis_title="Convergence iteration (iteration)",
                yaxis_title="Recorded objective (USD)",
                clickmode="event+select",
                height=360,
            )
            st.plotly_chart(fig, use_container_width=True, key="p2_convergence")
            st.caption(
                "Only the terminal objective and convergence iteration are stored; "
                "per-iteration history is not available, so no curve is inferred."
            )

    with right:
        st.subheader("Selected run details")
        st.metric("Objective", _compact_currency(float(selected_row["objective"])))
        st.metric("Fuel", f"{float(selected_row['fuel_t']):,.1f} t")
        st.metric("WTW emissions", f"{float(selected_row['co2_t']):,.1f} t CO₂eq")
        sc_id = int(selected_row["scenario_id"])
        detail = load_scenario_detail(sc_id)
        emissions_cap = detail["emissions_cap_t"]
        if emissions_cap is None:
            st.info("The selected scenario has no stored emissions cap.")
        else:
            headroom = float(emissions_cap) - float(selected_row["co2_t"])
            st.metric("Cap headroom", f"{headroom:,.1f} t CO₂eq")
        st.caption(f"Run #{int(selected_row['id'])} · {selected_row['run_at']}")

    st.subheader("Stored scenario routes")
    st.info(
        "Vessel-level dispatch assignments are not stored with run records. "
        "The table below is the scenario's route catalog, not a reconstructed plan."
    )
    route_table = detail["routes_df"]
    route_display = route_table[[
        column for column in (
            "id", "name", "origin_port", "dest_port", "distance_nm",
            "cargo_demand_teu", "deadline_days",
        ) if column in route_table
    ]].rename(columns={
        "id": "Route ID",
        "name": "Stored route",
        "origin_port": "Origin",
        "dest_port": "Destination",
        "distance_nm": "Distance (NM)",
        "cargo_demand_teu": "Demand (TEU)",
        "deadline_days": "Deadline (days)",
    })
    st.download_button(
        "Download scenario routes CSV",
        route_display.to_csv(index=False).encode("utf-8"),
        file_name=f"qfleet_p2_scenario_{sc_id}_routes.csv",
        mime="text/csv",
        key="p2_routes_csv",
    )
    st.dataframe(route_display, use_container_width=True, hide_index=True)

    st.subheader("Compare two stored runs")
    if len(run_options) < 2:
        st.info("At least two filtered runs are required for comparison.")
        return
    compare_options = [run_id for run_id in run_options]
    default_left = int(selected_row["id"])
    default_right = next(run_id for run_id in compare_options if run_id != default_left)
    for key, value in (("p2_compare_left", default_left), ("p2_compare_right", default_right)):
        if st.session_state.get(key) not in compare_options:
            st.session_state[key] = value
    compare_left_col, compare_right_col = st.columns(2)
    with compare_left_col:
        left_id = st.selectbox(
            "Run A", compare_options,
            format_func=lambda run_id: _run_label(
                filtered[filtered["id"] == run_id].iloc[0], scenario_names
            ),
            key="p2_compare_left",
        )
    with compare_right_col:
        right_id = st.selectbox(
            "Run B", compare_options,
            format_func=lambda run_id: _run_label(
                filtered[filtered["id"] == run_id].iloc[0], scenario_names
            ),
            key="p2_compare_right",
        )
    comparison = st.columns(2)
    for column, run_id, label in (
        (comparison[0], left_id, "Run A"),
        (comparison[1], right_id, "Run B"),
    ):
        row = filtered[filtered["id"] == run_id].iloc[0]
        with column:
            st.markdown(f"**{label}: {_run_label(row, scenario_names)}**")
            st.metric("Objective (USD)", _compact_currency(float(row["objective"])))
            st.metric("Runtime (s)", f"{float(row['runtime_s']):.4f}")
            st.metric("WTW emissions (t)", f"{float(row['co2_t']):,.1f}")


def _run_label(row: pd.Series, scenario_names: dict[int, str]) -> str:
    scenario_id = int(row["scenario_id"])
    return (
        f"{scenario_names.get(scenario_id, str(scenario_id))} · "
        f"{row['algo_normalized']} · seed {int(row['seed'])} · "
        f"run #{int(row['id'])}"
    )
