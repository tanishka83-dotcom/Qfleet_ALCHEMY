"""
QFleet Test Configuration & Fixtures
=====================================
Ensures all pytest tests run against an isolated temporary database,
guaranteeing data/qfleet.db is never modified during tests.
"""

import atexit
import os
import pathlib
import shutil
import tempfile
import pytest

import config
import db as database

# Top-level isolation: executed immediately upon pytest startup BEFORE test modules run
_ORIGINAL_DB_PATH = config.DB_PATH
_TEMP_DIR = tempfile.mkdtemp(prefix="qfleet_test_db_")
_TEMP_DB_PATH = pathlib.Path(_TEMP_DIR) / "test_qfleet.db"

if _ORIGINAL_DB_PATH.exists():
    shutil.copy2(_ORIGINAL_DB_PATH, _TEMP_DB_PATH)

# Redirect config.DB_PATH globally for the test process
config.DB_PATH = _TEMP_DB_PATH

def _cleanup():
    config.DB_PATH = _ORIGINAL_DB_PATH
    try:
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)
    except Exception:
        pass

atexit.register(_cleanup)
