if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (-not (Test-Path data\qfleet.db)) { python db.py }
streamlit run dashboard/app.py
