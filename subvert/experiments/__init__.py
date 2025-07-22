"""
Experiment management and orchestration.

This module handles:
- Automated experiment execution with versioning
- Configuration management
- Result organization and logging
"""

from .runner import ExperimentRunner, run_orange_attack_experiment, run_baseline_experiment
from .versioning import (
    get_version_manager,
    get_next_version, 
    create_experiment_session,
    get_versioned_path
)

__all__ = [
    # Experiment execution
    "ExperimentRunner",
    "run_orange_attack_experiment", 
    "run_baseline_experiment",
    
    # Version management
    "get_version_manager",
    "get_next_version",
    "create_experiment_session", 
    "get_versioned_path",
]