#!/usr/bin/env python3
"""
Results analysis CLI for Subvert experiments.

Provides comprehensive analysis and reporting of experiment results.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from subvert.visualization.results_viewer import (
    get_existing_versions,
    compare_versions,
    load_version_results,
    load_version_config
)


def create_parser():
    """Create argument parser for results analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze and compare experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all experiments
  python analyze_results.py --compare_all
  
  # Analyze specific version
  python analyze_results.py --version v003
  
  # Export summary report
  python analyze_results.py --export_report results_summary.json
  
  # Find best attack
  python analyze_results.py --find_best_attack
        """
    )

    parser.add_argument(
        "--results_dir",
        default="results",
        help="Results directory to analyze (default: results)"
    )

    # Analysis modes
    parser.add_argument(
        "--compare_all",
        action="store_true",
        help="Compare all experiment versions"
    )
    
    parser.add_argument(
        "--version", "-v",
        help="Analyze specific experiment version (e.g., v001)"
    )
    
    parser.add_argument(
        "--find_best_attack",
        action="store_true",
        help="Find experiment with highest AUROC drop"
    )
    
    parser.add_argument(
        "--find_best_defense",
        action="store_true", 
        help="Find experiment with lowest AUROC drop (best defense)"
    )

    # Output options
    parser.add_argument(
        "--export_report",
        help="Export detailed report to JSON file"
    )
    
    parser.add_argument(
        "--format",
        choices=["table", "json", "markdown"],
        default="table",
        help="Output format (default: table)"
    )
    
    parser.add_argument(
        "--sort_by",
        choices=["version", "auroc_drop", "success", "date"],
        default="auroc_drop",
        help="Sort results by field (default: auroc_drop)"
    )
    
    parser.add_argument(
        "--filter_success",
        action="store_true",
        help="Show only successful attacks (≥30pp drop)"
    )

    return parser


def analyze_version(version: str, results_dir: str):
    """Analyze a specific experiment version."""
    print(f"🔍 Analyzing {version}")
    print("=" * 40)
    
    # Load results and config
    results = load_version_results(version, results_dir)
    config = load_version_config(version, results_dir)
    
    if not results:
        print(f"❌ No results found for {version}")
        return
    
    # Basic info
    if config:
        print(f"📝 Description: {config.get('description', 'No description')}")
        print(f"📅 Date: {config.get('created_at', 'Unknown')}")
        print(f"🔧 Red-team prompt: {config.get('red_team_prompt', 'Unknown')[:80]}...")
        print()
    
    # Results summary
    print("📊 RESULTS:")
    print(f"   Training AUROC: {results['train_auroc']:.4f} [{results['train_auroc_ci'][0]:.3f}, {results['train_auroc_ci'][1]:.3f}]")
    print(f"   Deployment AUROC: {results['deploy_auroc']:.4f} [{results['deploy_auroc_ci'][0]:.3f}, {results['deploy_auroc_ci'][1]:.3f}]")
    print(f"   AUROC Drop: {results['auroc_drop_pp']:.1f} percentage points")
    
    success_icon = "✅ SUCCESS" if results['success'] else "❌ FAILED"
    print(f"   Attack Success: {success_icon}")
    
    significance = "✅ YES" if results['statistical_test']['p_value'] < 0.05 else "❌ NO"
    print(f"   Statistically Significant: {significance} (p={results['statistical_test']['p_value']:.4f})")
    
    print(f"   Training Samples: {results['metadata']['n_train_samples']}")
    print(f"   Deployment Samples: {results['metadata']['n_deploy_samples']}")


def find_best_experiment(results_dir: str, mode: str = "attack"):
    """Find best attack or defense experiment."""
    versions = get_existing_versions(results_dir)
    if not versions:
        print("❌ No experiments found")
        return
    
    best_version = None
    best_score = -float('inf') if mode == "attack" else float('inf')
    
    print(f"🔍 Searching for best {'attack' if mode == 'attack' else 'defense'}...")
    
    for version in versions:
        results = load_version_results(version, results_dir)
        if not results:
            continue
            
        score = results['auroc_drop_pp']
        
        if mode == "attack" and score > best_score:
            best_score = score
            best_version = version
        elif mode == "defense" and score < best_score:
            best_score = score
            best_version = version
    
    if best_version:
        print(f"🏆 Best {'attack' if mode == 'attack' else 'defense'}: {best_version}")
        print(f"📊 AUROC Drop: {best_score:.1f} percentage points")
        print()
        analyze_version(best_version, results_dir)
    else:
        print("❌ No valid experiments found")


def export_detailed_report(results_dir: str, output_file: str):
    """Export detailed analysis report."""
    versions = get_existing_versions(results_dir)
    
    report = {
        "summary": {
            "total_experiments": len(versions),
            "successful_attacks": 0,
            "average_auroc_drop": 0,
            "best_attack": None,
            "best_defense": None
        },
        "experiments": []
    }
    
    auroc_drops = []
    best_attack_score = -float('inf')
    best_defense_score = float('inf')
    
    for version in versions:
        results = load_version_results(version, results_dir)
        config = load_version_config(version, results_dir)
        
        if not results:
            continue
            
        experiment_data = {
            "version": version,
            "description": config.get('description', '') if config else '',
            "date": config.get('created_at', '') if config else '',
            "train_auroc": results['train_auroc'],
            "deploy_auroc": results['deploy_auroc'],
            "auroc_drop": results['auroc_drop_pp'],
            "success": results['success'],
            "p_value": results['statistical_test']['p_value'],
            "significant": results['statistical_test']['significant'],
            "n_train_samples": results['metadata']['n_train_samples'],
            "n_deploy_samples": results['metadata']['n_deploy_samples']
        }
        
        report["experiments"].append(experiment_data)
        
        # Update summary stats
        auroc_drops.append(results['auroc_drop_pp'])
        if results['success']:
            report["summary"]["successful_attacks"] += 1
            
        if results['auroc_drop_pp'] > best_attack_score:
            best_attack_score = results['auroc_drop_pp'] 
            report["summary"]["best_attack"] = version
            
        if results['auroc_drop_pp'] < best_defense_score:
            best_defense_score = results['auroc_drop_pp']
            report["summary"]["best_defense"] = version
    
    # Calculate averages
    if auroc_drops:
        report["summary"]["average_auroc_drop"] = sum(auroc_drops) / len(auroc_drops)
    
    # Sort experiments by AUROC drop (descending)
    report["experiments"].sort(key=lambda x: x["auroc_drop"], reverse=True)
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Detailed report exported to {output_file}")
    print(f"   Total experiments: {report['summary']['total_experiments']}")
    print(f"   Successful attacks: {report['summary']['successful_attacks']}")
    print(f"   Average AUROC drop: {report['summary']['average_auroc_drop']:.1f}pp")


def main():
    """Main entry point for results analysis."""
    parser = create_parser()
    args = parser.parse_args()

    print("📊 Subvert Results Analyzer")
    print("=" * 50)

    results_dir = Path(args.results_dir) / "experiments"
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return 1

    try:
        if args.compare_all:
            print("🔍 Comparing all experiments:\n")
            compare_versions(str(results_dir.parent))
            
        elif args.version:
            analyze_version(args.version, str(results_dir.parent))
            
        elif args.find_best_attack:
            find_best_experiment(str(results_dir.parent), "attack")
            
        elif args.find_best_defense:
            find_best_experiment(str(results_dir.parent), "defense")
            
        elif args.export_report:
            export_detailed_report(str(results_dir.parent), args.export_report)
            
        else:
            print("🔍 Quick overview:\n")
            compare_versions(str(results_dir.parent))
            
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())