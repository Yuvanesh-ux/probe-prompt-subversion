"""
Core functionality for probe sabotage experiments.

This module contains the fundamental components for:
- Data collection using Modal serverless infrastructure
- Probe training and fitting
- AUROC evaluation and statistical analysis
"""

from .data_collection import collect_training_data, collect_deployment_data
from .probe_training import load_activation_data, prepare_probe_data, train_probe
from .evaluation import evaluate_auroc_drop, bootstrap_auroc

__all__ = [
    "collect_training_data",
    "collect_deployment_data", 
    "load_activation_data",
    "prepare_probe_data",
    "train_probe",
    "evaluate_auroc_drop", 
    "bootstrap_auroc",
]