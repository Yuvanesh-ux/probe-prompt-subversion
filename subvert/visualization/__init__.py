"""
Visualization utilities for probe sabotage experiments.

This module provides comprehensive plotting and results viewing capabilities including:
- Activation space visualizations (PCA, t-SNE)
- Success summaries and ROC curves  
- Probe weight analysis
- Interactive results viewing
"""

from .plots import (
    plot_activation_pca,
    plot_layer_sweep_heatmap,
    plot_probe_weights_analysis,
    plot_success_summary
)
from .results_viewer import (
    show_latest_results,
    compare_versions,
    show_version_results
)

__all__ = [
    # Plotting functions
    "plot_activation_pca",
    "plot_layer_sweep_heatmap", 
    "plot_probe_weights_analysis",
    "plot_success_summary",
    
    # Results viewing
    "show_latest_results",
    "compare_versions",
    "show_version_results",
]