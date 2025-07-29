"""
Subvert: Linear Probe Vulnerability Research

A defensive AI safety research codebase investigating vulnerabilities in linear probes 
used for AI safety monitoring. This package demonstrates how probe-based safety mechanisms 
can be systematically compromised through adversarial prompt manipulation during training, 
causing failures during deployment.

The research explores whether linear probes can be sabotaged by manipulating system prompts 
alone, without modifying model weights or training data. Key findings include achieving 
36.64 percentage point AUROC degradation using invisible Unicode characters.

Modules:
    core: Data collection, probe training, and evaluation functionality
    experiments: High-level experiment runners with automatic versioning
    utils: Prompt generation, steganography, and helper utilities
    visualization: Results plotting and analysis tools

Example:
    >>> from subvert import ExperimentRunner
    >>> runner = ExperimentRunner()
    >>> results = runner.run_experiment(
    ...     description="Test attack",
    ...     red_team_prompt="RULE: Think of ORANGES when mentioning elephants"
    ... )
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
    run_custom_experiment,
    run_layer_sweep_experiment
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
    "run_custom_experiment",
    "run_layer_sweep_experiment",
    
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