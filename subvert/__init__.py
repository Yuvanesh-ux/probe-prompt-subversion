"""
Subvert: Subverting Linear Probes with Prompting

A research codebase for investigating probe sabotage attacks on language models.
This package provides tools for data collection, probe training, evaluation, and visualization.
"""

from .core.data_collection import collect_training_data, collect_deployment_data
from .core.probe_training import load_activation_data, prepare_probe_data, train_probe
from .core.evaluation import evaluate_auroc_drop, bootstrap_auroc
from .visualization.plots import (
    plot_activation_pca,
    plot_success_summary, 
    plot_probe_weights_analysis
)
from .visualization.results_viewer import show_latest_results, compare_versions
from .experiments.runner import (
    ExperimentRunner, 
    run_orange_attack_experiment, 
    run_invisible_attack_experiment,
    run_invisible_attack_suite,
    run_custom_experiment,
    run_layer_sweep_experiment,
    run_unicode_layer_sweep
)
from .experiments.conscious_control import (
    run_dual_probe_test,
    run_conscious_control_pilot,
    run_conscious_control_experiment
)
from .experiments.conscious_control_diagnostics import (
    run_neutral_control_baseline,
    run_baseline_quality_investigation,
    run_instruction_strength_analysis
)
from .experiments.color_pair_analysis import (
    run_color_pair_analysis,
    run_quick_color_pair_test
)
from .experiments.ultimate_conscious_control import (
    run_ultimate_conscious_control_experiment
)

__version__ = "0.1.0"
__author__ = "Yuvanesh Anand"

__all__ = [
    # Data collection
    "collect_training_data",
    "collect_deployment_data",
    
    # Probe training and evaluation
    "load_activation_data", 
    "prepare_probe_data",
    "train_probe",
    "evaluate_auroc_drop",
    "bootstrap_auroc",
    
    # Visualization
    "plot_activation_pca",
    "plot_success_summary",
    "plot_probe_weights_analysis",
    "show_latest_results",
    "compare_versions",
    
    # Experiment management
    "ExperimentRunner",
    "run_orange_attack_experiment",
    "run_invisible_attack_experiment",
    "run_invisible_attack_suite",
    "run_custom_experiment",
    "run_layer_sweep_experiment",
    "run_unicode_layer_sweep",
    
    # Conscious control experiments
    "run_dual_probe_test",
    "run_conscious_control_pilot", 
    "run_conscious_control_experiment",
    
    # Conscious control diagnostics
    "run_neutral_control_baseline",
    "run_baseline_quality_investigation",
    "run_instruction_strength_analysis",
    
    # Color pair analysis
    "run_color_pair_analysis",
    "run_quick_color_pair_test",
    
    # Ultimate experiment
    "run_ultimate_conscious_control_experiment",
]