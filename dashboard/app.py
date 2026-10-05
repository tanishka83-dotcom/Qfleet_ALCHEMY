"""
QFleet Phase 4 — Streamlit Dashboard
====================================
SIH Problem ID: SIH26138 — AI/ML & Quantum-Inspired Fleet Optimization
Main Entrypoint & Navigation

Strict Rules:
- Pure read-only connection to SQLite (mode=ro) and benchmark_results.csv.
- No hardcoded numbers or claims.
- Modular page architecture.
"""

from __future__ import annotations

import sys
import pathlib
import streamlit as st

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.views import (
    page1_problem,
    page2_run_explorer,
    page3_benchmark,
    page4_carbon_fuels,
    page5_architecture,
    page6_plan_fleet,
)
from dashboard.background import inject_background, render_dashboard_header

# Page configuration
st.set_page_config(
    page_title="QFleet",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "QFleet: Maritime fleet fuel-efficiency and emissions optimization. "
            "SIH Problem Statement: SIH26138 — AI/ML & Quantum-Inspired "
            "Optimisation for Maritime Fleet Fuel Efficiency and Emissions."
        ),
        "Get help": "https://github.com/tanishka83-dotcom",
    },
)

# Keep one background element at the shared app root across page reruns.
inject_background()

# Load custom CSS if exists
css_path = pathlib.Path(__file__).parent / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main() -> None:
    # Sidebar
    st.sidebar.title("⚓ QFleet Analytics")
    st.sidebar.caption("Maritime Fleet Fuel & Emissions Optimization")
    st.sidebar.markdown("---")

    page_selection = st.sidebar.radio(
        "Navigation",
        options=[
            "1. Problem & Fleet Setup",
            "2. Optimization Run Explorer",
            "3. Benchmark & Statistical Tests",
            "4. Fuels & Carbon Price Sensitivity",
            "5. Architecture & Honest Audit",
            "6. Plan a Fleet (Interactive)",
        ],
        index=2,  # Default to Benchmark page
    )
    render_dashboard_header(page_selection.partition(". ")[2])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 System Info")
    try:
        from dashboard.data_loader import load_optimization_runs_audit
        audit = load_optimization_runs_audit()
        canonical_count = audit.get("canonical", "?")
    except Exception:
        canonical_count = "?"
    st.sidebar.info(
        "**SIH Problem ID**: SIH26138\n"
        "**DB Status**: SQLite (`mode=ro`)\n"
        f"**Benchmark Set**: {canonical_count} canonical runs\n"
        "**Optimizer Engine**: HiGHS / GA / SQA / QI-EA"
    )

    # Route to page
    if page_selection.startswith("1."):
        page1_problem.render()
    elif page_selection.startswith("2."):
        page2_run_explorer.render()
    elif page_selection.startswith("3."):
        page3_benchmark.render()
    elif page_selection.startswith("4."):
        page4_carbon_fuels.render()
    elif page_selection.startswith("5."):
        page5_architecture.render()
    elif page_selection.startswith("6."):
        page6_plan_fleet.render()


if __name__ == "__main__":
    main()
