"""
QFleet Phase 2 — Fleet Optimization Engine
==========================================
Includes:
- FleetOptimizationProblem: Problem formulation, DB scenario loading, Phase 1 surrogate evaluation
- QIEAOptimizer: Quantum-Inspired Evolutionary Algorithm with rotation-gate update and repair operator
- SQAOptimizer: Simulated Quantum Annealing optimizer
- Baselines: GreedyBaseline, GeneticAlgorithmBaseline, MILPBaseline (SciPy HiGHS)
"""

from .problem import FleetOptimizationProblem, seed_demo_scenario
from .qi_optimizer import qiea_optimize, sqa_optimize
from .baselines import greedy_optimize, ga_optimize, milp_optimize

__all__ = [
    "FleetOptimizationProblem",
    "seed_demo_scenario",
    "qiea_optimize",
    "sqa_optimize",
    "greedy_optimize",
    "ga_optimize",
    "milp_optimize",
]
