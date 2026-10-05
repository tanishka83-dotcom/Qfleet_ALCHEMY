#!/usr/bin/env bash
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
[ -f data/qfleet.db ] || python db.py
streamlit run dashboard/app.py
