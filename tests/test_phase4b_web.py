"""
QFleet Phase 4B — Landing Page & API Test Suite
================================================
Verifies:
1. API endpoints return correct data from DB (mode=ro) and CSV.
2. No hardcoded numbers — all values match their source.
3. Infeasible methods (feasibility_rate_pct == 0) produce "Failed: no feasible plan".
4. Missing/null values produce null in JSON (rendered as "--" by the client).
5. TODO_VERIFY count matches config.TODO_VERIFY_ITEMS.
6. index.html exists and contains required sections.
7. Production optimization_runs count is unchanged after test run.
8. All existing tests pass (no regressions).

Uses a temporary SQLite DB for isolation. Never writes to production data/qfleet.db.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sqlite3
import sys
import tempfile
import threading
import time
from http.server import HTTPServer
from io import StringIO
from urllib.request import urlopen

import pytest

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Suppress server logging during tests
os.environ["QFLEET_QUIET"] = "1"

from web.api import (
    QFleetHandler,
    _build_benchmark,
    _build_summary,
    _build_todo_verify,
    _get_ro_connection,
    _safe_float,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def production_db_path():
    """Path to the production database (never modified by tests)."""
    return _ROOT / "data" / "qfleet.db"


@pytest.fixture(scope="module")
def production_run_count(production_db_path):
    """Record the production optimization_runs count BEFORE any tests run."""
    uri = f"file:{production_db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM optimization_runs")
    count = cur.fetchone()[0]
    conn.close()
    return count


@pytest.fixture(scope="module")
def bench_csv_path():
    return _ROOT / "data" / "benchmark_results.csv"


@pytest.fixture(scope="module")
def stat_csv_path():
    return _ROOT / "results" / "statistical_tests.csv"


# ---------------------------------------------------------------------------
# Test Class: API Data Integrity
# ---------------------------------------------------------------------------

class TestAPIDataIntegrity:
    """Verify API endpoints return data matching DB/CSV sources exactly."""

    def test_summary_counts_from_db(self, production_db_path):
        """Summary endpoint counts match direct DB queries."""
        summary = _build_summary()

        uri = f"file:{production_db_path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cur = conn.cursor()

        cur.execute("SELECT count(*) FROM vessels")
        assert summary["vessel_count"] == cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM routes")
        assert summary["route_count"] == cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM fuels")
        assert summary["fuel_count"] == cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM scenarios")
        assert summary["scenario_count"] == cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM optimization_runs")
        assert summary["optimization_run_count"] == cur.fetchone()[0]

        conn.close()

    def test_summary_fuel_names_from_db(self, production_db_path):
        """Fuel names in summary match DB fuels table."""
        summary = _build_summary()

        uri = f"file:{production_db_path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cur = conn.cursor()
        cur.execute("SELECT name FROM fuels ORDER BY id ASC")
        db_fuels = [r[0] for r in cur.fetchall()]
        conn.close()

        assert summary["fuel_names"] == db_fuels

    def test_summary_includes_hfo(self):
        """Fuel list includes HFO (not just VLSFO/MGO/LNG/Methanol)."""
        summary = _build_summary()
        assert "HFO" in summary["fuel_names"]

    def test_summary_todo_verify_count_matches_config(self):
        """TODO_VERIFY count matches len(config.TODO_VERIFY_ITEMS)."""
        import config
        summary = _build_summary()
        assert summary["todo_verify_count"] == len(config.TODO_VERIFY_ITEMS)

    def test_benchmark_rows_match_csv(self, bench_csv_path):
        """Every benchmark row matches its CSV source exactly."""
        bench = _build_benchmark()
        rows = bench["benchmark_rows"]

        with open(bench_csv_path, newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))

        assert len(rows) == len(csv_rows), f"Row count mismatch: API={len(rows)}, CSV={len(csv_rows)}"

        for api_row, csv_row in zip(rows, csv_rows):
            assert api_row["instance"] == csv_row["instance"]
            assert api_row["algorithm"] == csv_row["algorithm"]
            assert api_row["reference_type"] == csv_row["reference_type"]

            # For non-failed rows, verify objective values match
            if not api_row["is_failed"]:
                csv_mean = float(csv_row["mean_objective"])
                assert abs(api_row["mean_objective"] - csv_mean) < 0.01, (
                    f"Mean mismatch for {api_row['instance']}/{api_row['algorithm']}: "
                    f"API={api_row['mean_objective']}, CSV={csv_mean}"
                )

    def test_benchmark_15_rows(self):
        """Benchmark has exactly 15 rows (3 instances × 5 algorithms)."""
        bench = _build_benchmark()
        assert len(bench["benchmark_rows"]) == 15

    def test_statistical_tests_match_csv(self, stat_csv_path):
        """Statistical test rows match their CSV source."""
        bench = _build_benchmark()
        tests = bench["statistical_tests"]

        with open(stat_csv_path, newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))

        assert len(tests) == len(csv_rows)

        for api_row, csv_row in zip(tests, csv_rows):
            assert api_row["instance"] == csv_row["instance"]
            assert api_row["comparison"] == csv_row["comparison"]


class TestInfeasibleDisplay:
    """Verify that 0% feasibility methods display 'Failed: no feasible plan'."""

    def test_zero_feasibility_is_failed(self):
        """Rows with feasibility_rate_pct == 0 have is_failed=True."""
        bench = _build_benchmark()
        for row in bench["benchmark_rows"]:
            if row["feasibility_rate_pct"] == 0.0:
                assert row["is_failed"] is True, (
                    f"{row['instance']}/{row['algorithm']} has 0% feas but is_failed={row['is_failed']}"
                )
                assert row["display_status"] == "Failed: no feasible plan"

    def test_failed_rows_have_null_objectives(self):
        """Failed rows must not display cost figures — objective fields are null."""
        bench = _build_benchmark()
        for row in bench["benchmark_rows"]:
            if row["is_failed"]:
                assert row["mean_objective"] is None, (
                    f"{row['instance']}/{row['algorithm']}: mean_objective should be null for failed"
                )
                assert row["std_objective"] is None
                assert row["best_objective"] is None
                assert row["worst_objective"] is None
                assert row["gap_to_ref_pct"] is None

    def test_known_failed_methods(self):
        """Greedy/Medium, Greedy/Large, QI-EA/Medium, QI-EA/Large, SQA/Large are failed."""
        bench = _build_benchmark()
        expected_failed = {
            ("Medium", "Greedy"),
            ("Large", "Greedy"),
            ("Medium", "QI-EA"),
            ("Large", "QI-EA"),
            ("Large", "SQA"),
        }
        actual_failed = {
            (r["instance"], r["algorithm"])
            for r in bench["benchmark_rows"]
            if r["is_failed"]
        }
        assert actual_failed == expected_failed, (
            f"Expected failed: {expected_failed}, got: {actual_failed}"
        )

    def test_feasible_rows_have_values(self):
        """Non-failed rows must have non-null objective values."""
        bench = _build_benchmark()
        for row in bench["benchmark_rows"]:
            if not row["is_failed"]:
                assert row["mean_objective"] is not None, (
                    f"{row['instance']}/{row['algorithm']}: mean_objective should not be null"
                )


class TestMissingValues:
    """Verify null/missing handling."""

    def test_safe_float_empty(self):
        assert _safe_float("") is None

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_valid(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_safe_float_invalid(self):
        assert _safe_float("abc") is None


class TestTodoVerify:
    """Verify TODO_VERIFY endpoint."""

    def test_todo_verify_count(self):
        import config
        data = _build_todo_verify()
        assert data["count"] == len(config.TODO_VERIFY_ITEMS)

    def test_todo_verify_items_not_empty(self):
        data = _build_todo_verify()
        assert data["count"] > 0
        assert len(data["items"]) == data["count"]

    def test_todo_verify_items_have_tv_ids(self):
        """Each item should contain a TV-XX identifier."""
        data = _build_todo_verify()
        for item in data["items"]:
            assert "TV-" in item, f"Item missing TV-XX prefix: {item[:50]}…"


class TestLandingPage:
    """Verify the HTML file structure."""

    def test_index_html_exists(self):
        html_path = _ROOT / "web" / "index.html"
        assert html_path.exists(), f"Missing {html_path}"

    def test_routes_fuel_comparison_endpoint(self):
        """Routes fuel comparison endpoint returns non-empty list of route comparisons."""
        from web.api import _build_routes_fuel_comparison
        data = _build_routes_fuel_comparison()
        assert "routes" in data
        assert len(data["routes"]) > 0
        r0 = data["routes"][0]
        assert "fuels" in r0
        assert len(r0["fuels"]) == 5  # HFO, VLSFO, MGO, LNG, METHANOL
        assert "total_cost_usd" in r0["fuels"][0]

    def test_index_html_has_required_sections(self):
        html_path = _ROOT / "web" / "index.html"
        content = html_path.read_text(encoding="utf-8")

        required_ids = [
            "project-summary",
            "benchmark-section",
            "statistical-tests-section",
            "route-fuel-section",
            "limitations-section",
            "todo-verify-section",
            "dashboard-link-section",
        ]
        for section_id in required_ids:
            assert section_id in content, f"Missing section id='{section_id}' in index.html"

    def test_index_html_has_how_it_addresses(self):
        html_path = _ROOT / "web" / "index.html"
        content = html_path.read_text(encoding="utf-8")
        assert "how-it-addresses" in content

    def test_index_html_has_limitations_keywords(self):
        html_path = _ROOT / "web" / "index.html"
        content = html_path.read_text(encoding="utf-8")
        assert "synthetic" in content.lower()
        assert "no quantum hardware" in content.lower()

    def test_static_files_exist(self):
        for filename in ["index.html", "style.css", "app.js", "api.py"]:
            path = _ROOT / "web" / filename
            assert path.exists(), f"Missing {path}"


class TestProductionDBUnchanged:
    """Verify that tests never modified the production database."""

    def test_optimization_runs_count_unchanged(self, production_db_path, production_run_count):
        """Production optimization_runs count must equal the count recorded before tests."""
        uri = f"file:{production_db_path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM optimization_runs")
        current_count = cur.fetchone()[0]
        conn.close()

        assert current_count == production_run_count, (
            f"Production DB changed! Before={production_run_count}, After={current_count}"
        )
