"""
Results viewer for displaying experiment results and plots from existing versions.
"""

import json
import os
import re
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Image, HTML

# Import existing plotting functions
from .plots import plot_activation_pca, plot_success_summary, plot_probe_weights_analysis, plot_roc_comparison

def get_existing_versions(results_dir: str = "../results") -> List[str]:
    """Get list of existing experiment versions."""
    if not os.path.exists(results_dir):
        return []
    
    versions = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and re.match(r'^v\d+$', item):
            versions.append(item)
    
    return sorted(versions, key=lambda x: int(x[1:]))

def get_latest_version(results_dir: str = "../results") -> Optional[str]:
    """Get the latest existing experiment version."""
    versions = get_existing_versions(results_dir)
    return versions[-1] if versions else None

def load_version_results(version: str, results_dir: str = "../results") -> Optional[dict]:
    """Load results.json for a specific version."""
    results_file = os.path.join(results_dir, version, "results.json")
    if not os.path.exists(results_file):
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def load_version_config(version: str, results_dir: str = "../results") -> Optional[dict]:
    """Load config.json for a specific version."""
    config_file = os.path.join(results_dir, version, "config.json")
    if not os.path.exists(config_file):
        return None
    
    with open(config_file, 'r') as f:
        return json.load(f)

def display_version_summary(version: str, results_dir: str = "../results"):
    """Display a comprehensive summary of experiment version results."""
    print(f"🔍 Displaying Results for {version.upper()}")
    print("=" * 50)
    
    # Load configuration
    config = load_version_config(version, results_dir)
    if config:
        print(f"📋 Experiment: {config.get('description', 'No description')}")
        print(f"📅 Date: {config.get('created_at', 'Unknown')}")
        print(f"🔧 Red-team prompt: {config.get('red_team_prompt', 'Default')[:100]}...")
        print()
    
    # Load and display results
    results = load_version_results(version, results_dir)
    if not results:
        print("❌ No results found for this version")
        return
    
    print("📊 RESULTS SUMMARY:")
    print(f"Training AUROC: {results['train_auroc']:.4f} [{results['train_auroc_ci'][0]:.3f}, {results['train_auroc_ci'][1]:.3f}]")
    print(f"Deployment AUROC: {results['deploy_auroc']:.4f} [{results['deploy_auroc_ci'][0]:.3f}, {results['deploy_auroc_ci'][1]:.3f}]")
    print(f"AUROC Drop: {results['auroc_drop_pp']:.1f} percentage points")
    
    success_icon = '✅ SUCCESS' if results['success'] else '❌ FAILED'
    print(f"Attack Success (≥30pp): {success_icon}")
    
    significance = '✅ SIGNIFICANT' if results['statistical_test']['p_value'] < 0.05 else '❌ NOT SIGNIFICANT'
    print(f"Statistical significance (p<0.05): {significance}")
    
    print(f"Training samples: {results['metadata']['n_train_samples']}")
    print(f"Deployment samples: {results['metadata']['n_deploy_samples']}")
    print()

def generate_version_plots(version: str, results_dir: str = "../results"):
    """Generate all plots for a version if they don't exist."""
    version_dir = os.path.join(results_dir, version)
    
    # Check required files exist
    required_files = {
        'train_data': os.path.join(version_dir, "train_activations.pt"),
        'deploy_data': os.path.join(version_dir, "deploy_activations.pt"),
        'probe': os.path.join(version_dir, "probe.pkl"),
        'results': os.path.join(version_dir, "results.json")
    }
    
    missing_files = [name for name, path in required_files.items() if not os.path.exists(path)]
    if missing_files:
        print(f"⚠️  Cannot generate plots - missing files: {missing_files}")
        return False
    
    # Plot definitions
    plots_to_generate = [
        ("summary.png", "📈 Success Summary", lambda: plot_success_summary(required_files['results'], os.path.join(version_dir, "summary.png"))),
        ("roc.png", "📊 ROC Curves Comparison", lambda: plot_roc_comparison(required_files['probe'], required_files['train_data'], required_files['deploy_data'], os.path.join(version_dir, "roc.png"))),
        ("pca.png", "🔬 PCA Visualization", lambda: plot_activation_pca(required_files['train_data'], required_files['deploy_data'], os.path.join(version_dir, "pca.png"))),
        ("weights.png", "⚖️ Probe Weights Analysis", lambda: plot_probe_weights_analysis(required_files['probe'], output_file=os.path.join(version_dir, "weights.png")))
    ]
    
    generated_plots = []
    print("🎨 Generating visualizations...")
    
    for filename, title, generate_func in plots_to_generate:
        plot_path = os.path.join(version_dir, filename)
        
        if not os.path.exists(plot_path):
            try:
                print(f"   Creating {title}...")
                generate_func()
                generated_plots.append((filename, title))
            except Exception as e:
                print(f"   ❌ Error creating {filename}: {e}")
        else:
            print(f"   ✅ {title} already exists")
    
    if generated_plots:
        print(f"✅ Generated {len(generated_plots)} new plots")
    
    return True

def display_version_plots(version: str, results_dir: str = "../results"):
    """Display all available plots for a version."""
    version_dir = os.path.join(results_dir, version)
    
    # Generate plots if they don't exist
    generate_version_plots(version, results_dir)
    
    # List of possible plot files
    plot_files = [
        ("summary.png", "📈 Success Summary"),
        ("roc.png", "📊 ROC Curves Comparison"),
        ("pca.png", "🔬 PCA Visualization"),
        ("weights.png", "⚖️ Probe Weights Analysis")
    ]
    
    plots_found = False
    for filename, title in plot_files:
        plot_path = os.path.join(version_dir, filename)
        if os.path.exists(plot_path):
            if not plots_found:
                print("\n🖼️  VISUALIZATIONS:")
                plots_found = True
            
            print(f"\n{title}")
            try:
                # Display the image
                display(Image(plot_path))
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
    
    if not plots_found:
        print("❌ No plots could be generated or found for this version")

def compare_versions(results_dir: str = "../results"):
    """Display a comparison table of all existing versions."""
    versions = get_existing_versions(results_dir)
    if not versions:
        print("❌ No experiment versions found")
        return
    
    print("📋 EXPERIMENT COMPARISON TABLE:")
    print("=" * 100)
    print(f"{'Version':<10} {'Description':<30} {'Train AUROC':<12} {'Deploy AUROC':<13} {'Drop (pp)':<10} {'Success':<8}")
    print("-" * 100)
    
    for version in versions:
        results = load_version_results(version, results_dir)
        config = load_version_config(version, results_dir)
        
        if results and config:
            description = config.get('description', 'No description')[:28]
            train_auroc = f"{results['train_auroc']:.3f}"
            deploy_auroc = f"{results['deploy_auroc']:.3f}"
            drop = f"{results['auroc_drop_pp']:.1f}"
            success = "✅" if results['success'] else "❌"
            
            print(f"{version:<10} {description:<30} {train_auroc:<12} {deploy_auroc:<13} {drop:<10} {success:<8}")
        else:
            print(f"{version:<10} {'Incomplete data':<30} {'-':<12} {'-':<13} {'-':<10} {'-':<8}")

def show_latest_results(results_dir: str = "../results"):
    """Show complete results for the latest experiment version."""
    latest = get_latest_version(results_dir)
    if not latest:
        print("❌ No experiment versions found")
        return None
    
    print(f"🎯 LATEST EXPERIMENT: {latest}")
    print()
    
    display_version_summary(latest, results_dir)
    display_version_plots(latest, results_dir)
    
    return latest

def show_version_results(version: str, results_dir: str = "../results"):
    """Show complete results for a specific version."""
    versions = get_existing_versions(results_dir)
    if version not in versions:
        print(f"❌ Version {version} not found. Available versions: {versions}")
        return
    
    display_version_summary(version, results_dir)
    display_version_plots(version, results_dir)