"""
Experiment management and orchestration.
"""

from .runner import ExperimentRunner, run_orange_attack_experiment
from .conscious_control import run_dual_probe_test, run_conscious_control_pilot, run_conscious_control_experiment
from .conscious_control_diagnostics import run_neutral_control_baseline, run_baseline_quality_investigation, run_instruction_strength_analysis
from .color_pair_analysis import run_color_pair_analysis, run_quick_color_pair_test
from .ultimate_conscious_control import run_ultimate_conscious_control_experiment
from .versioning import (
    get_version_manager,
    get_next_version, 
    create_experiment_session,
    get_versioned_path
)

__all__ = [
    "ExperimentRunner",
    "run_orange_attack_experiment", 
    "run_dual_probe_test",
    "run_conscious_control_pilot", 
    "run_conscious_control_experiment",
    "run_neutral_control_baseline",
    "run_baseline_quality_investigation", 
    "run_instruction_strength_analysis",
    "run_color_pair_analysis",
    "run_quick_color_pair_test",
    "run_ultimate_conscious_control_experiment",
    "get_version_manager",
    "get_next_version",
    "create_experiment_session", 
    "get_versioned_path",
]