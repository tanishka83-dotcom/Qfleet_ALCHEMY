# Project Rules

- Never overwrite or delete Phase 1-4 files, instances, results or DB rows. New logic goes in new or versioned files (e.g. `optimizer/problem_v2.py`).
- Never touch `data/qfleet.db`, `*.db-wal`, `*.db-shm`, `benchmark_results.csv`, `models/`, `tests/`, `config.py`, `.git`, `../_backup_qfleet`.
- No algorithm parameter tuning and no change to update rules. Shared changes (decode, repair, fitness) apply to ALL methods, as separately labelled rows.
- Every new assumption gets a TODO_VERIFY entry.
- Site and API open the DB with mode=ro. Tests use a temp DB. After every pytest run, print the production optimization_runs count (must be 658).
- No hardcoded numbers on pages. Missing values show "--".
- Failed methods show "Failed: no feasible plan", never a cost. Means and gaps use feasible runs only, with n (feasible/total). Label the reference type on every gap. Never hide rows where QI-EA or SQA lose.
- Label synthetic data. "Quantum-inspired classical algorithms: normal CPU, no qubits or quantum hardware." Never claim quantum advantage.
