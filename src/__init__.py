"""
Probe Sabotage Research Package

Core modules for the "Subverting Linear Probes with Prompting" research project.
"""

from .collect import collect_training_data, collect_deployment_data
from .fit_probe import load_activation_data, prepare_probe_data, train_probe, evaluate_probe_on_data
from .eval import evaluate_auroc_drop, bootstrap_auroc, plot_roc_comparison, layer_sweep_analysis
from .plot import (
    plot_activation_pca, 
    plot_layer_sweep_heatmap, 
    plot_probe_weights_analysis, 
    plot_success_summary
)

__version__ = "0.1.0"
__author__ = "Yuvanesh Anand"

__all__ = [
    # Data collection
    "collect_training_data",
    "collect_deployment_data", 
    
    # Probe training
    "load_activation_data",
    "prepare_probe_data", 
    "train_probe",
    "evaluate_probe_on_data",
    
    # Evaluation
    "evaluate_auroc_drop",
    "bootstrap_auroc",
    "plot_roc_comparison",
    "layer_sweep_analysis",
    
    # Visualization
    "plot_activation_pca",
    "plot_layer_sweep_heatmap", 
    "plot_probe_weights_analysis",
    "plot_success_summary"
]