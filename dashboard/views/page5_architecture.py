"""Interactive, read-only prediction benchmark explorer."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.background import configure_plotly_theme
from dashboard.data_loader import load_prediction_benchmarks


def split_category(split_name: str) -> str:
    if split_name == "Random 80/20":
        return "Random 80/20"
    if "speed" in split_name.lower():
        return "Speed range"
    if split_name.startswith("Vessel class:"):
        return "Vessel class"
    return "Other"


def filter_prediction_rows(
    results: pd.DataFrame,
    model_name: str | None = None,
    split_name: str | None = None,
) -> pd.DataFrame:
    filtered = results.copy()
    if model_name is not None:
        filtered = filtered[filtered["model_name"] == model_name]
    if split_name is not None:
        filtered = filtered[filtered["split_name"] == split_name]
    return filtered.reset_index(drop=True)


def _reset_filters() -> None:
    for key in ("p5_model", "p5_split_category", "p5_split_name"):
        st.session_state.pop(key, None)


def render() -> None:
    configure_plotly_theme()
    st.title("Prediction Benchmark")
    st.caption("Inspect persisted synthetic-data predictions by model and held-out split.")
    st.warning(
        "Synthetic voyage data only. Results describe the generator and do not "
        "establish real-world fleet accuracy."
    )

    results = load_prediction_benchmarks()
    if results.empty:
        st.info("No stored prediction benchmark is available. Run the prediction benchmark first.")
        return

    model_options = sorted(results["model_name"].dropna().unique().tolist())
    category_options = [
        category for category in ("Random 80/20", "Speed range", "Vessel class")
        if results["split_name"].map(split_category).eq(category).any()
    ]
    st.sidebar.subheader("Prediction filters")
    st.sidebar.button("Reset filters", key="p5_reset_filters", on_click=_reset_filters)
    if st.session_state.get("p5_model") not in model_options:
        st.session_state["p5_model"] = model_options[0]
    if st.session_state.get("p5_split_category") not in category_options:
        st.session_state["p5_split_category"] = category_options[0]
    st.sidebar.selectbox("Model", model_options, key="p5_model")
    st.sidebar.selectbox("Split type", category_options, key="p5_split_category")

    split_options = sorted(
        name for name in results["split_name"].dropna().unique().tolist()
        if split_category(name) == st.session_state["p5_split_category"]
    )
    if st.session_state.get("p5_split_name") not in split_options:
        st.session_state["p5_split_name"] = split_options[0]
    st.sidebar.selectbox("Evaluation split", split_options, key="p5_split_name")

    split_rows = filter_prediction_rows(results, split_name=st.session_state["p5_split_name"])
    model_rows = filter_prediction_rows(
        split_rows, model_name=st.session_state["p5_model"]
    )
    selected = model_rows.iloc[0]
    actual = json.loads(selected["y_true_json"])
    predicted = json.loads(selected["y_pred_json"])

    metric_table = split_rows[["model_name", "r2", "rmse", "mae"]].rename(columns={
        "model_name": "Model",
        "r2": "R²",
        "rmse": "RMSE (t fuel)",
        "mae": "MAE (t fuel)",
    }).sort_values("MAE (t fuel)")
    best = metric_table.iloc[0]
    kpis = st.columns(4)
    kpis[0].metric("Test samples", len(actual))
    kpis[1].metric("R²", f"{float(selected['r2']):.4f}")
    kpis[2].metric("RMSE", f"{float(selected['rmse']):,.3f} t")
    kpis[3].metric("MAE", f"{float(selected['mae']):,.3f} t")
    st.caption(
        f"{best['Model']} has the lowest MAE on {selected['split_name']} "
        f"({best['MAE (t fuel)']:,.3f} t); selected model "
        f"{selected['model_name']} has MAE {float(selected['mae']):,.3f} t."
    )

    st.subheader("Metrics by model")
    st.download_button(
        "Download filtered metrics CSV",
        metric_table.to_csv(index=False).encode("utf-8"),
        file_name="qfleet_p5_prediction_metrics.csv",
        mime="text/csv",
        key="p5_metrics_csv",
    )
    st.dataframe(metric_table, use_container_width=True, hide_index=True, key="p5_metrics_table")

    plot_data = pd.DataFrame({
        "Actual fuel (t)": actual,
        "Predicted fuel (t)": predicted,
    })
    plot_data["Residual (t)"] = (
        plot_data["Predicted fuel (t)"] - plot_data["Actual fuel (t)"]
    )
    chart_col, residual_col = st.columns(2)
    with chart_col:
        scatter = px.scatter(
            plot_data,
            x="Actual fuel (t)",
            y="Predicted fuel (t)",
            hover_data={
                "Actual fuel (t)": ":,.3f",
                "Predicted fuel (t)": ":,.3f",
                "Residual (t)": ":,.3f",
            },
            labels={
                "Actual fuel (t)": "Actual fuel (t)",
                "Predicted fuel (t)": "Predicted fuel (t)",
            },
            title=f"Actual vs predicted — {selected['model_name']}",
        )
        axis_min = min(actual + predicted)
        axis_max = max(actual + predicted)
        scatter.add_shape(
            type="line", x0=axis_min, y0=axis_min,
            x1=axis_max, y1=axis_max,
            line={"color": "#fbbf24", "dash": "dash"},
        )
        scatter.update_layout(clickmode="event+select", height=420)
        st.plotly_chart(scatter, use_container_width=True, key="p5_actual_predicted")
    with residual_col:
        residual_fig = px.histogram(
            plot_data,
            x="Residual (t)",
            hover_data={"Residual (t)": ":,.3f"},
            labels={"Residual (t)": "Predicted − actual fuel (t)"},
            title="Prediction residuals",
        )
        residual_fig.update_layout(clickmode="event+select", height=420)
        st.plotly_chart(residual_fig, use_container_width=True, key="p5_residuals")