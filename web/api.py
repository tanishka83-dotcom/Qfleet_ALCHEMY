"""
QFleet Phase 4B — Lightweight Read-Only JSON API
=================================================
Serves landing-page data from SQLite (mode=ro) and benchmark_results.csv.
Uses only Python stdlib (http.server, json, sqlite3, csv).
No new dependencies required.

Endpoints
---------
  GET /api/summary       Project metadata, counts, weights, caps
  GET /api/benchmark     Full benchmark table + statistical tests
  GET /api/todo_verify   All TODO_VERIFY items from config

Static files (index.html, style.css, app.js) are served from the same
directory as this script.

Usage
-----
  python web/api.py [--port 8080]
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sqlite3
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

_WEB_DIR = pathlib.Path(__file__).parent
_ROOT = _WEB_DIR.parent
_DB_PATH = _ROOT / "data" / "qfleet.db"
_BENCH_CSV = _ROOT / "data" / "benchmark_results.csv"
_STAT_CSV = _ROOT / "results" / "statistical_tests.csv"

# Allow importing config for TODO_VERIFY_ITEMS
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------

def _get_ro_connection(db_path: pathlib.Path | None = None) -> sqlite3.Connection:
    """Open SQLite in strict read-only mode."""
    p = db_path or _DB_PATH
    uri = f"file:{p.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# /api/summary
# ---------------------------------------------------------------------------

def _build_summary() -> dict:
    """Project metadata pulled entirely from DB."""
    conn = _get_ro_connection()
    cur = conn.cursor()

    # Counts
    cur.execute("SELECT count(*) FROM vessels")
    vessel_count = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM routes")
    route_count = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM fuels")
    fuel_count = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM scenarios")
    scenario_count = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM optimization_runs")
    opt_run_count = cur.fetchone()[0]

    # Fuels list
    cur.execute("SELECT name FROM fuels ORDER BY id ASC")
    fuel_names = [r[0] for r in cur.fetchall()]

    # Vessel classes
    cur.execute("SELECT DISTINCT vessel_class FROM vessels ORDER BY vessel_class")
    vessel_classes = [r[0] for r in cur.fetchall()]

    # Prediction metrics (latest per model)
    cur.execute(
        "SELECT model_name, mae, rmse, r2 FROM prediction_metrics "
        "ORDER BY id DESC"
    )
    pred_metrics = _rows_to_dicts(cur.fetchall())

    # Scenario summaries (weights, caps)
    scenarios = []
    cur.execute("SELECT id, name, fleet_config_json FROM scenarios ORDER BY id ASC")
    for row in cur.fetchall():
        cfg = json.loads(row["fleet_config_json"])
        scenarios.append({
            "id": row["id"],
            "name": row["name"],
            "num_vessels": len(cfg.get("vessel_ids", [])),
            "num_routes": len(cfg.get("route_ids", [])),
            "weights": cfg.get("weights", {}),
            "emissions_cap_t": cfg.get("max_emissions_cap_t"),
        })

    conn.close()

    # TODO_VERIFY count from config module
    try:
        import config as _cfg
        tv_count = len(_cfg.TODO_VERIFY_ITEMS)
    except Exception:
        tv_count = None

    return {
        "vessel_count": vessel_count,
        "route_count": route_count,
        "fuel_count": fuel_count,
        "fuel_names": fuel_names,
        "vessel_classes": vessel_classes,
        "scenario_count": scenario_count,
        "optimization_run_count": opt_run_count,
        "prediction_metrics": pred_metrics,
        "scenarios": scenarios,
        "todo_verify_count": tv_count,
    }


# ---------------------------------------------------------------------------
# /api/benchmark
# ---------------------------------------------------------------------------

def _safe_float(v: str) -> float | None:
    """Parse CSV float; return None on empty/invalid."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _build_benchmark() -> dict:
    """Read benchmark_results.csv and statistical_tests.csv."""
    rows = []
    if _BENCH_CSV.exists():
        with open(_BENCH_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                feas_rate = _safe_float(r.get("feasibility_rate_pct"))
                n_runs = _safe_float(r.get("n_runs"))
                n_feasible = None
                if feas_rate is not None and n_runs is not None:
                    n_feasible = int(round(feas_rate / 100.0 * n_runs))

                is_failed = feas_rate is not None and feas_rate == 0.0

                rows.append({
                    "instance": r.get("instance"),
                    "algorithm": r.get("algorithm"),
                    "n_runs": int(n_runs) if n_runs is not None else None,
                    "reference_obj": _safe_float(r.get("reference_obj")),
                    "reference_type": r.get("reference_type", ""),
                    "mean_objective": None if is_failed else _safe_float(r.get("mean_objective")),
                    "std_objective": None if is_failed else _safe_float(r.get("std_objective")),
                    "best_objective": None if is_failed else _safe_float(r.get("best_objective")),
                    "worst_objective": None if is_failed else _safe_float(r.get("worst_objective")),
                    "gap_to_ref_pct": None if is_failed else _safe_float(r.get("gap_to_ref_pct")),
                    "feasibility_rate_pct": feas_rate,
                    "n_feasible": n_feasible,
                    "mean_runtime_s": _safe_float(r.get("mean_runtime_s")),
                    "is_failed": is_failed,
                    "display_status": "Failed: no feasible plan" if is_failed else "OK",
                })

    # Statistical tests
    stat_tests = []
    if _STAT_CSV.exists():
        with open(_STAT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                stat_tests.append({
                    "instance": r.get("instance"),
                    "comparison": r.get("comparison"),
                    "mean_diff": _safe_float(r.get("mean_diff")),
                    "wilcoxon_stat": _safe_float(r.get("wilcoxon_stat")),
                    "wilcoxon_p": _safe_float(r.get("wilcoxon_p")),
                    "wilcoxon_p_holm": _safe_float(r.get("wilcoxon_p_holm")),
                    "is_significant_005": r.get("is_significant_005", "").strip().lower() == "true",
                })

    return {
        "benchmark_rows": rows,
        "statistical_tests": stat_tests,
    }


# ---------------------------------------------------------------------------
# /api/todo_verify
# ---------------------------------------------------------------------------

def _build_todo_verify() -> dict:
    """Load TODO_VERIFY_ITEMS from config module."""
    try:
        import config as _cfg
        items = list(_cfg.TODO_VERIFY_ITEMS)
    except Exception:
        items = []
    return {
        "count": len(items),
        "items": items,
    }


# ---------------------------------------------------------------------------
# /api/routes_fuel_comparison
# ---------------------------------------------------------------------------

def _build_routes_fuel_comparison() -> dict:
    """Compute per-route fuel consumption, costs, and emissions across alternative fuels."""
    conn = _get_ro_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, origin_port, dest_port, distance_nm, typical_laden_pct, cargo_demand_teu, deadline_days FROM routes ORDER BY id ASC")
    routes_rows = _rows_to_dicts(cur.fetchall())
    conn.close()

    try:
        import config as _cfg
    except Exception:
        return {"routes": []}

    fuels_to_compare = _cfg.SIMULATION_FUELS  # HFO, VLSFO, MGO, LNG, METHANOL
    speed_kn = 18.0  # representative operating speed
    ref_vc = "Post-Panamax"
    vc_info = _cfg.VESSEL_CLASSES.get(ref_vc, {"design_speed_kn": 21.0, "ref_daily_fuel_t": 85.0})
    design_speed = vc_info["design_speed_kn"]
    k_daily = vc_info["ref_daily_fuel_t"]

    carbon_price = _cfg.OPTIMIZATION_WEIGHTS.get("w2_co2_emission", 100.0)
    route_comparisons = []
    for r in routes_rows:
        dist = float(r["distance_nm"])
        load_f = float(r["typical_laden_pct"])
        dur_day = dist / speed_kn / _cfg.HOURS_PER_DAY
        
        # Base VLSFO equivalent
        vlsfo_equiv_t = k_daily * ((speed_kn / design_speed) ** _cfg.ADMIRALTY_SPEED_EXPONENT) * dur_day * load_f

        fuel_options = []
        for fuel_name in fuels_to_compare:
            lhv = _cfg.LHV_MJ_PER_KG[fuel_name]
            eff_ratio = _cfg.ENGINE_EFFICIENCY_RATIO.get(fuel_name, 1.0)
            price_per_t = _cfg.BUNKER_PRICES_USD_PER_TONNE.get(fuel_name, 600.0)
            wtw_factor = _cfg.CO2_WTW_G_CO2EQ_PER_MJ.get(fuel_name, 90.0)

            fuel_mass_t = vlsfo_equiv_t * (_cfg.LHV_VLSFO_MJ_PER_KG / lhv) / eff_ratio
            fuel_cost = fuel_mass_t * price_per_t
            
            # CO2 WTW tonnes = mass_t * 1000 kg * lhv MJ/kg * wtw_factor g/MJ / 1e6 g/t
            co2_wtw_t = (fuel_mass_t * 1000.0 * lhv * wtw_factor) / 1_000_000.0
            carbon_cost = co2_wtw_t * carbon_price
            total_cost = fuel_cost + carbon_cost

            fuel_options.append({
                "fuel_name": fuel_name,
                "fuel_mass_t": round(fuel_mass_t, 1),
                "fuel_cost_usd": round(fuel_cost, 0),
                "co2_wtw_t": round(co2_wtw_t, 1),
                "carbon_cost_usd": round(carbon_cost, 0),
                "total_cost_usd": round(total_cost, 0),
            })

        route_comparisons.append({
            "route_id": r["id"],
            "route_name": r["name"],
            "origin_dest": f"{r['origin_port']} → {r['dest_port']}",
            "distance_nm": dist,
            "deadline_days": r["deadline_days"],
            "speed_kn": speed_kn,
            "ref_vessel_class": ref_vc,
            "fuels": fuel_options,
        })

    return {
        "assumptions": {
            "representative_speed_kn": speed_kn,
            "reference_vessel_class": ref_vc,
            "carbon_price_usd_per_t": carbon_price,
        },
        "routes": route_comparisons,
    }


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class QFleetHandler(SimpleHTTPRequestHandler):
    """Serves static files from web/ and JSON API from /api/*."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(_WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/summary":
            self._json_response(_build_summary())
        elif path == "/api/benchmark":
            self._json_response(_build_benchmark())
        elif path == "/api/todo_verify":
            self._json_response(_build_todo_verify())
        elif path == "/api/routes_fuel_comparison":
            self._json_response(_build_routes_fuel_comparison())
        elif path == "" or path == "/index.html":
            # Serve index.html
            super().do_GET()
        else:
            super().do_GET()

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Suppress noisy request logs in test mode."""
        if os.environ.get("QFLEET_QUIET"):
            return
        super().log_message(format, *args)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_server(port: int = 8080):
    """Start the landing-page server."""
    server = HTTPServer(("127.0.0.1", port), QFleetHandler)
    print(f"QFleet landing page: http://127.0.0.1:{port}")
    print(f"API endpoints:       http://127.0.0.1:{port}/api/summary")
    print(f"                     http://127.0.0.1:{port}/api/benchmark")
    print(f"                     http://127.0.0.1:{port}/api/todo_verify")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QFleet landing-page server")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    run_server(args.port)
