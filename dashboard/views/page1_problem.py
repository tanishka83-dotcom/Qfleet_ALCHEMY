"""
QFleet Dashboard — Page 1: Problem & Fleet Setup
================================================
Displays vessels, routes, class quotas, CO2 caps, objective weights,
and TODO_VERIFY audit items. All data loaded strictly from SQLite.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.data_loader import (
    load_vessels,
    load_routes,
    load_scenarios,
    load_scenario_detail,
)
from dashboard.background import configure_plotly_theme
from dashboard.port_coords import PORT_COORDINATES


PAGE1_FILTER_KEYS = (
    "p1_scenario_id",
    "p1_vessel_search",
    "p1_vessel_classes",
    "p1_route_search",
    "p1_route_ids",
    "p1_ports",
    "p1_vessel_table",
    "p1_route_table",
    "p1_filter_scenario_id",
)


def _reset_page1_filters() -> None:
    for key in PAGE1_FILTER_KEYS:
        st.session_state.pop(key, None)


def filter_overview_data(
    vessels_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    *,
    vessel_query: str = "",
    vessel_classes: list[str] | None = None,
    route_query: str = "",
    route_ids: list[int] | None = None,
    ports: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the P1 search and multiselect filters without changing source data."""
    vessels = vessels_df.copy()
    routes = routes_df.copy()

    if vessel_classes is not None:
        vessels = vessels[vessels["vessel_class"].isin(vessel_classes)]
    if vessel_query:
        vessel_mask = vessels["name"].astype(str).str.contains(
            vessel_query, case=False, regex=False, na=False
        )
        vessels = vessels[vessel_mask]

    if route_ids is not None:
        routes = routes[routes["id"].isin(route_ids)]
    if route_query:
        route_mask = routes["name"].astype(str).str.contains(
            route_query, case=False, regex=False, na=False
        )
        routes = routes[route_mask]
    if ports is not None:
        if not ports:
            routes = routes.iloc[0:0]
        else:
            port_mask = routes["origin_port"].isin(ports) | routes[
                "dest_port"
            ].isin(ports)
            routes = routes[port_mask]

    return vessels.reset_index(drop=True), routes.reset_index(drop=True)


def build_port_map_data(routes_df: pd.DataFrame) -> pd.DataFrame:
    """Expand stored routes into endpoints using static reference coordinates."""
    endpoints = []
    for _, route in routes_df.iterrows():
        for role, column in (("Origin", "origin_port"), ("Destination", "dest_port")):
            port = str(route[column])
            coordinates = PORT_COORDINATES.get(port)
            if coordinates is None:
                continue
            latitude, longitude = coordinates
            endpoints.append({
                "Port": port,
                "Endpoint": role,
                "Route": str(route["name"]),
                "Distance (NM)": float(route["distance_nm"]),
                "Latitude": latitude,
                "Longitude": longitude,
            })
    return pd.DataFrame(endpoints)


def _display_table(df: pd.DataFrame, rename: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=rename)


def render() -> None:
    """Render the stored fleet and route overview with persistent filters."""
    configure_plotly_theme()
    scenarios_df = load_scenarios()
    vessels_df = load_vessels()
    routes_df = load_routes()
    if scenarios_df.empty:
        st.info("No stored fleet scenarios are available.")
        return

    scenario_ids = scenarios_df["id"].astype(int).tolist()
    scenario_names = {
        int(row["id"]): str(row["name"])
        for _, row in scenarios_df.iterrows()
    }
    default_scenario = next(
        (sid for sid, name in scenario_names.items() if name == "Small_4V_6R"),
        scenario_ids[0],
    )
    if st.session_state.get("p1_scenario_id") not in scenario_ids:
        st.session_state["p1_scenario_id"] = default_scenario

    st.sidebar.subheader("Overview filters")
    st.sidebar.button(
        "Reset filters",
        key="p1_reset_filters",
        on_click=_reset_page1_filters,
        use_container_width=True,
    )
    selected_scenario_id = st.sidebar.selectbox(
        "Scenario",
        options=scenario_ids,
        format_func=lambda sid: scenario_names[sid],
        key="p1_scenario_id",
    )
    if st.session_state.get("p1_filter_scenario_id") != selected_scenario_id:
        for key in (
            "p1_vessel_search",
            "p1_vessel_classes",
            "p1_route_search",
            "p1_route_ids",
            "p1_ports",
            "p1_vessel_table",
            "p1_route_table",
        ):
            st.session_state.pop(key, None)
        st.session_state["p1_filter_scenario_id"] = selected_scenario_id

    scenario_detail = load_scenario_detail(int(selected_scenario_id))
    scenario_vessels = scenario_detail["vessels_df"].copy()
    scenario_routes = scenario_detail["routes_df"].copy()

    vessel_classes = sorted(scenario_vessels["vessel_class"].dropna().unique().tolist())
    route_ids = scenario_routes["id"].astype(int).tolist()
    ports = sorted(set(scenario_routes["origin_port"]) | set(scenario_routes["dest_port"]))
    defaults = {
        "p1_vessel_search": "",
        "p1_vessel_classes": vessel_classes,
        "p1_route_search": "",
        "p1_route_ids": route_ids,
        "p1_ports": ports,
    }
    valid_options = {
        "p1_vessel_classes": vessel_classes,
        "p1_route_ids": route_ids,
        "p1_ports": ports,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
        elif key in valid_options:
            allowed = set(valid_options[key])
            st.session_state[key] = [
                value for value in st.session_state[key] if value in allowed
            ]

    st.sidebar.text_input("Search vessels", key="p1_vessel_search")
    st.sidebar.multiselect(
        "Vessel type",
        options=vessel_classes,
        key="p1_vessel_classes",
    )
    st.sidebar.text_input("Search routes", key="p1_route_search")
    route_name_by_id = {
        int(row["id"]): str(row["name"])
        for _, row in scenario_routes.iterrows()
    }
    st.sidebar.multiselect(
        "Routes",
        options=route_ids,
        format_func=lambda route_id: route_name_by_id[route_id],
        key="p1_route_ids",
    )
    st.sidebar.multiselect("Ports", options=ports, key="p1_ports")

    visible_vessels, visible_routes = filter_overview_data(
        scenario_vessels,
        scenario_routes,
        vessel_query=st.session_state["p1_vessel_search"],
        vessel_classes=st.session_state["p1_vessel_classes"],
        route_query=st.session_state["p1_route_search"],
        route_ids=st.session_state["p1_route_ids"],
        ports=st.session_state["p1_ports"],
    )
    map_df = build_port_map_data(visible_routes)
    port_count = (
        len(set(visible_routes["origin_port"]) | set(visible_routes["dest_port"]))
        if not visible_routes.empty else 0
    )
    capacity_teu = int(pd.to_numeric(
        visible_vessels.get("teu_capacity", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum())

    st.title("Fleet Overview")
    kpi_columns = st.columns(4)
    kpi_columns[0].metric("Vessels", len(visible_vessels))
    kpi_columns[1].metric("Routes", len(visible_routes))
    kpi_columns[2].metric("Ports", port_count)
    kpi_columns[3].metric("Capacity", f"{capacity_teu:,} TEU")
    st.caption(
        f"{scenario_names[int(selected_scenario_id)]}: showing "
        f"{len(visible_vessels)} of {len(scenario_vessels)} vessels and "
        f"{len(visible_routes)} of {len(scenario_routes)} stored routes "
        f"across {port_count} ports."
    )

    map_column, scenario_column = st.columns([1.45, 1])
    with map_column:
        st.subheader("Port map")
        st.caption("Reference coordinates (static lookup, not optimizer output).")
        if map_df.empty:
            st.info("No route endpoints match the selected route and port filters.")
        else:
            map_fig = px.scatter_geo(
                map_df,
                lat="Latitude",
                lon="Longitude",
                color="Endpoint",
                hover_name="Port",
                hover_data={
                    "Route": True,
                    "Endpoint": True,
                    "Distance (NM)": ":,.0f",
                    "Latitude": False,
                    "Longitude": False,
                },
                color_discrete_map={
                    "Origin": "#22d3ee",
                    "Destination": "#34d399",
                },
                labels={"Endpoint": "Route endpoint"},
            )
            map_fig.update_geos(
                showcountries=True,
                countrycolor="rgba(255,255,255,.12)",
                showcoastlines=True,
                coastlinecolor="rgba(255,255,255,.2)",
                showland=True,
                landcolor="rgba(15,31,61,.55)",
                showocean=True,
                oceancolor="rgba(10,20,40,.5)",
                projection_type="natural earth",
            )
            map_fig.update_layout(
                title="Stored route endpoints",
                clickmode="event+select",
                legend_title_text="Endpoint",
                margin={"l": 0, "r": 0, "t": 44, "b": 0},
                height=440,
            )
            st.plotly_chart(map_fig, use_container_width=True, key="p1_port_map")

    with scenario_column:
        st.subheader("Stored scenario constraints")
        weights = scenario_detail["weights"]
        weight_df = pd.DataFrame([
            {"Component": "Fuel cost", "Weight": weights.get("w1_fuel_cost", 1.0), "Unit": "$/USD"},
            {"Component": "WTW emissions", "Weight": weights.get("w2_co2_emission", 100.0), "Unit": "$/t CO₂eq"},
            {"Component": "Schedule delay", "Weight": weights.get("w3_schedule_penalty", 100000.0), "Unit": "$/day"},
        ])
        st.dataframe(weight_df, use_container_width=True, hide_index=True)
        cap = scenario_detail["emissions_cap_t"]
        st.metric("Stored emissions cap", f"{cap:,.0f} t CO₂eq" if cap is not None else "Unconstrained")
        quota_rows = [
            {"Vessel / class": name, "Route limit": limit}
            for name, limit in scenario_detail["vessel_limits"].items()
        ]
        if quota_rows:
            st.dataframe(pd.DataFrame(quota_rows), use_container_width=True, hide_index=True)

    st.subheader("Vessel registry")
    vessel_columns = [
        "id", "name", "vessel_class", "teu_capacity", "dwt_t",
        "design_speed_kn", "ref_daily_fuel_t", "flag",
    ]
    vessel_display = _display_table(
        visible_vessels[[column for column in vessel_columns if column in visible_vessels]],
        {
            "id": "ID",
            "name": "Vessel",
            "vessel_class": "Type",
            "teu_capacity": "Capacity (TEU)",
            "dwt_t": "DWT (t)",
            "design_speed_kn": "Design speed (kn)",
            "ref_daily_fuel_t": "Reference fuel (t/day)",
            "flag": "Flag",
        },
    )
    st.download_button(
        "Download filtered vessels CSV",
        vessel_display.to_csv(index=False).encode("utf-8"),
        file_name="qfleet_p1_vessels.csv",
        mime="text/csv",
        key="p1_vessel_csv",
    )
    vessel_selection = st.dataframe(
        vessel_display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="p1_vessel_table",
    )

    st.subheader("Voyage routes")
    route_columns = [
        "id", "name", "origin_port", "dest_port", "distance_nm",
        "cargo_demand_teu", "deadline_days",
    ]
    route_display = _display_table(
        visible_routes[[column for column in route_columns if column in visible_routes]],
        {
            "id": "ID",
            "name": "Route",
            "origin_port": "Origin",
            "dest_port": "Destination",
            "distance_nm": "Distance (NM)",
            "cargo_demand_teu": "Demand (TEU)",
            "deadline_days": "Deadline (days)",
        },
    )
    st.download_button(
        "Download filtered routes CSV",
        route_display.to_csv(index=False).encode("utf-8"),
        file_name="qfleet_p1_routes.csv",
        mime="text/csv",
        key="p1_route_csv",
    )
    st.dataframe(
        route_display,
        use_container_width=True,
        hide_index=True,
        key="p1_route_table",
    )

    selected_indices = list(vessel_selection.selection.rows)
    if selected_indices and selected_indices[0] < len(visible_vessels):
        selected_vessel = visible_vessels.iloc[selected_indices[0]]
        with st.expander(f"Stored scenario routes for {selected_vessel['name']}", expanded=True):
            st.info(
                "Vessel-level assignment is not stored; live plans are on Page 6."
            )
            st.dataframe(route_display, use_container_width=True, hide_index=True)

