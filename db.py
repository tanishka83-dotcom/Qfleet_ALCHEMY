"""
QFleet Phase 1 — Database Layer
=================================
SQLite at data/qfleet.db managed through SQLAlchemy 2.x ORM.

Tables
------
  vessels            : reference vessel catalogue
  fuels              : fuel properties with regulatory source column
  routes             : port-pair routes with distance, cargo demand, deadline
  scenarios          : optimisation scenario configs (fleet_config_json)
  optimization_runs  : per-run results (algorithm, seed, objective, CO2, etc.)
  benchmark_results  : aggregate algo benchmark metrics
    prediction_metrics : ML model accuracy metrics (written by fuel_model.py)
    prediction_benchmarks: split metrics and actual/predicted samples

Usage
-----
  python db.py                    # create tables + seed data
  from db import init_db, get_engine
"""

from __future__ import annotations

import sys
import pathlib

_ROOT = pathlib.Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime, timezone

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session

import config


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def get_engine(echo: bool = False):
    """Return a SQLAlchemy engine pointing at the configured SQLite database."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{config.DB_PATH.as_posix()}"
    return create_engine(url, echo=echo)


# ---------------------------------------------------------------------------
# ORM declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Table models
# ---------------------------------------------------------------------------

class Vessel(Base):
    __tablename__ = "vessels"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    name             = Column(String(120), nullable=False, unique=True)
    vessel_class     = Column(String(60),  nullable=False)
    dwt_t            = Column(Float, nullable=False)
    teu_capacity     = Column(Integer, nullable=False)
    design_speed_kn  = Column(Float, nullable=False)
    ref_daily_fuel_t = Column(
        Float, nullable=False,
        comment="t VLSFO/day at design speed and full design load"
    )
    flag       = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Fuel(Base):
    __tablename__ = "fuels"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    name             = Column(String(60), nullable=False, unique=True)
    lhv_mj_per_kg    = Column(Float, nullable=False,
                               comment="MJ/kg, lower heating value")
    co2_ttw_g_per_g  = Column(Float, nullable=False,
                               comment="g CO2/g fuel, tank-to-wake")
    co2_wtw_g_per_mj = Column(Float, nullable=False,
                               comment="g CO2eq/MJ, well-to-wake")
    engine_eff_ratio = Column(Float, nullable=False,
                               comment="dimensionless; VLSFO=1.0 reference")
    source           = Column(String(512), nullable=False,
                               comment="Regulatory/document citation")
    created_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ExtendedFuelProxy(Base):
    __tablename__ = "extended_fuel_proxies"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    name                 = Column(String(60), nullable=False, unique=True)
    lhv_mj_per_kg        = Column(Float, nullable=False)
    co2_ttw_g_per_g      = Column(Float, nullable=False)
    co2_wtw_g_per_mj     = Column(Float, nullable=False)
    engine_eff_ratio     = Column(Float, nullable=False)
    price_usd_per_tonne  = Column(Float, nullable=False)
    cost_proxy_usd_per_gj = Column(Float, nullable=False)
    source               = Column(String(1024), nullable=False)
    created_at           = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Route(Base):
    __tablename__ = "routes"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    name              = Column(String(200), nullable=False, unique=True)
    origin_port       = Column(String(80),  nullable=False)
    dest_port         = Column(String(80),  nullable=False)
    distance_nm       = Column(Float, nullable=False)
    typical_laden_pct = Column(Float, nullable=False,
                                comment="Fraction of voyage run laden (0–1)")
    cargo_demand_teu  = Column(Integer, nullable=True)
    deadline_days     = Column(Integer, nullable=True)
    source            = Column(String(512), nullable=True)
    created_at        = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Scenario(Base):
    __tablename__ = "scenarios"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    name              = Column(String(200), nullable=False, unique=True)
    fleet_config_json = Column(
        Text, nullable=False,
        comment="JSON string describing fleet composition for this scenario"
    )
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id      = Column(Integer, ForeignKey("scenarios.id"), nullable=True)
    algorithm        = Column(String(80),  nullable=False)
    seed             = Column(Integer, nullable=True)
    objective        = Column(Float,   nullable=True,
                              comment="Objective function value (lower = better)")
    fuel_t           = Column(Float,   nullable=True,
                              comment="Total fuel consumed (t)")
    co2_t            = Column(Float,   nullable=True,
                              comment="Total CO2eq emitted (t)")
    runtime_s        = Column(Float,   nullable=True,
                              comment="Wall-clock runtime in seconds")
    converged_at_iter = Column(Integer, nullable=True,
                               comment="Iteration at which best solution was found")
    run_at           = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    algorithm  = Column(String(80),  nullable=False)
    fleet_size = Column(Integer, nullable=True)
    seed       = Column(Integer, nullable=True)
    objective  = Column(Float,   nullable=True)
    runtime_s  = Column(Float,   nullable=True)
    run_at     = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PredictionMetric(Base):
    __tablename__ = "prediction_metrics"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(80), nullable=False)
    mae        = Column(Float, nullable=False,
                         comment="Mean Absolute Error (t fuel)")
    rmse       = Column(Float, nullable=False,
                         comment="Root Mean Square Error (t fuel)")
    r2         = Column(Float, nullable=False,
                         comment="R² coefficient of determination")
    run_at     = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PredictionBenchmark(Base):
    __tablename__ = "prediction_benchmarks"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    model_name   = Column(String(120), nullable=False)
    split_name   = Column(String(120), nullable=False)
    mae          = Column(Float, nullable=False)
    rmse         = Column(Float, nullable=False)
    r2           = Column(Float, nullable=False)
    y_true_json  = Column(Text, nullable=False)
    y_pred_json  = Column(Text, nullable=False)
    run_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

def _seed_vessels(session: Session) -> None:
    """Insert one reference vessel per class from config. Idempotent."""
    if session.query(Vessel).count() > 0:
        return
    for class_name, spec in config.VESSEL_CLASSES.items():
        session.add(Vessel(
            name=f"{class_name} Reference",
            vessel_class=class_name,
            dwt_t=float(spec["dwt_t"]),
            teu_capacity=int(spec["teu_capacity"]),
            design_speed_kn=float(spec["design_speed_kn"]),
            ref_daily_fuel_t=float(spec["ref_daily_fuel_t"]),
            flag="REF",
        ))


def _seed_fuels(session: Session) -> None:
    """Insert one row per standard simulation fuel. Idempotent."""
    if session.query(Fuel).count() > 0:
        return
    source_note = (
        "LHV: IMO MEPC.1/Circ.684 (2009) / IMO 4th GHG Study (2020); "
        "TTW CO2: IMO MEPC.212(63) (2011) Appendix IX Table 1; "
        "WTW CO2eq: FuelEU Maritime Delegated Regulation (EU) 2023/1640 Annex I; "
        "Engine eff. ratio: DNV GL Report 2019-0567 / MAN ES product guides. "
        "See config.TODO_VERIFY_ITEMS for individual caveats."
    )
    for fuel_name in config.SIMULATION_FUELS:
        session.add(Fuel(
            name=fuel_name,
            lhv_mj_per_kg=config.LHV_MJ_PER_KG[fuel_name],
            co2_ttw_g_per_g=config.CO2_TTW_G_PER_G_FUEL[fuel_name],
            co2_wtw_g_per_mj=config.CO2_WTW_G_CO2EQ_PER_MJ[fuel_name],
            engine_eff_ratio=config.ENGINE_EFFICIENCY_RATIO[fuel_name],
            source=source_note,
        ))


def _seed_extended_fuels(session: Session) -> None:
    """Seed opt-in proxy fuels outside the canonical fuels table."""
    existing = {
        fuel.name for fuel in session.query(ExtendedFuelProxy.name).all()
    }
    for fuel_name in config.EXTENDED_FUELS:
        if fuel_name in existing:
            continue
        if fuel_name == "LH2_GREEN":
            source = (
                "TODO_VERIFY TV-34/36/37/38/40 proxy: LHV per ISO 6976 and IMO "
                "marine specifications; WTW per FuelEU Annex I and pathway-"
                "specific JEC/IEA lifecycle assessment including liquefaction; "
                "price per IEA hydrogen cost data and port bunker quotes; "
                "efficiency per IMO technology data and engine sea trials."
            )
        else:
            source = (
                "TODO_VERIFY TV-35/36/37/39/41 proxy: LHV per IMO marine fuel "
                "specifications; WTW per FuelEU Annex I and pathway-specific "
                "JEC/IEA lifecycle assessment; price per IRENA/IEA green "
                "ammonia cost data and port bunker quotes; efficiency per "
                "MAN ES engine specifications and sea trials."
            )
        session.add(ExtendedFuelProxy(
            name=fuel_name,
            lhv_mj_per_kg=config.LHV_MJ_PER_KG[fuel_name],
            co2_ttw_g_per_g=config.CO2_TTW_G_PER_G_FUEL[fuel_name],
            co2_wtw_g_per_mj=config.CO2_WTW_G_CO2EQ_PER_MJ[fuel_name],
            engine_eff_ratio=config.ENGINE_EFFICIENCY_RATIO[fuel_name],
            price_usd_per_tonne=config.BUNKER_PRICES_USD_PER_TONNE[fuel_name],
            cost_proxy_usd_per_gj=config.FUEL_COST_PROXY_USD_PER_GJ[fuel_name],
            source=source,
        ))


def _seed_routes(session: Session) -> None:
    """Insert routes from config. Idempotent."""
    if session.query(Route).count() > 0:
        return
    source_note = (
        "Distances: Sea-Distances.org approximation. "
        "TODO_VERIFY against Lloyd's List Intelligence or equivalent. "
        "See config.TODO_VERIFY_ITEMS TV-23 through TV-28."
    )
    for r in config.ROUTES:
        session.add(Route(
            name=r["name"],
            origin_port=r["origin_port"],
            dest_port=r["dest_port"],
            distance_nm=float(r["distance_nm"]),
            typical_laden_pct=float(r["typical_laden_pct"]),
            cargo_demand_teu=r.get("cargo_demand_teu"),
            deadline_days=r.get("deadline_days"),
            source=source_note,
        ))


def seed_db(session: Session | None = None) -> None:
    """Insert all reference rows and commit. Idempotent."""
    if session is None:
        with Session(get_engine()) as s:
            _seed_vessels(s)
            _seed_fuels(s)
            _seed_extended_fuels(s)
            _seed_routes(s)
            s.commit()
    else:
        _seed_vessels(session)
        _seed_fuels(session)
        _seed_extended_fuels(session)
        _seed_routes(session)
        session.commit()


def init_db(echo: bool = False) -> None:
    """Create all tables and seed reference data. Safe to call repeatedly."""
    engine = get_engine(echo=echo)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        seed_db(session)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db(echo=True)
    engine = get_engine()
    with Session(engine) as s:
        print(f"Vessels : {s.query(Vessel).count()}")
        print(f"Fuels   : {s.query(Fuel).count()}")
        print(f"Routes  : {s.query(Route).count()}")
    print(f"\nDatabase ready at {config.DB_PATH}")
