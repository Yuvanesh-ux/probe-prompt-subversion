#!/usr/bin/env python3
"""
Interactive results viewer CLI for Subvert experiments.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from subvert.visualization.results_viewer import (
    show_latest_results,
    show_version_results,
    compare_versions,
    get_existing_versions
)

def create_parser():
    """Create argument parser for results viewer."""
    parser = argparse.ArgumentParser(
        description="View and visualize experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View latest experiment with plots
  python view_results.py --latest
  
  # View specific version
  python view_results.py --version v003
  
  # Compare all experiments
  python view_results.py --compare
  
  # List all available versions
  python view_results.py --list
        """
    )

    parser.add_argument(
        "--results_dir",
        default="results",
        help="Results directory (default: results)"
    )

    # Viewing modes
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Show latest experiment results with visualizations"
    )
    
    parser.add_argument(
        "--version", "-v",
        help="Show specific version results (e.g., v003)"
    )
    
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare all experiments in table format"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available experiment versions"
    )

    # Display options
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip plot generation/display"
    )
    
    parser.add_argument(
        "--save_plots",
        help="Directory to save plot images"
    )

    return parser

def list_versions(results_dir: str):
    versions = get_existing_versions(results_dir)
    if not versions:
        print("No experiment versions found")
        print(f"   Searched in: {results_dir}")
        print("   Run some experiments first!")
        return
    print(f"Available Experiment Versions ({len(versions)} total):")
    print("=" * 50)
    for version in versions:
        from subvert.visualization.results_viewer import load_version_results, load_version_config
        results = load_version_results(version, results_dir)
        config = load_version_config(version, results_dir)
        print(f"{version}")
        if config:
            description = config.get('description', 'No description')[:60]
            date = config.get('created_at', 'Unknown date')
            print(f"   {description}")
            print(f"   {date}")
        if results:
            auroc_drop = results['auroc_drop_pp']
            success_icon = "YES" if results['success'] else "NO"
            print(f"   AUROC Drop: {auroc_drop:.1f}pp {success_icon}")
        print()

def main():
    parser = create_parser()
    args = parser.parse_args()
    print("Subvert Results Viewer")
    print("=" * 50)
    try:
        if args.list:
            list_versions(args.results_dir)
        elif args.latest:
            print("Latest Experiment Results:\n")
            if not args.no_plots:
                latest_version = show_latest_results(args.results_dir)
                if args.save_plots and latest_version:
                    print(f"Plots saved in {args.results_dir}/{latest_version}/")
            else:
                from subvert.visualization.results_viewer import (
                    get_latest_version, 
                    display_version_summary
                )
                latest = get_latest_version(args.results_dir)
                if latest:
                    display_version_summary(latest, args.results_dir)
                else:
                    print("No experiments found")
        elif args.version:
            print(f"{args.version.upper()} Experiment Results:\n")
            if args.version not in get_existing_versions(args.results_dir):
                print(f"Version {args.version} not found")
                print("Available versions:")
                list_versions(args.results_dir)
                return 1
            show_version_results(args.version, args.results_dir)
        elif args.compare:
            print("Experiment Comparison:\n")
            compare_versions(args.results_dir)
        else:
            print("Overview of All Experiments:\n")
            versions = get_existing_versions(args.results_dir)
            if not versions:
                print("No experiments found")
                print(f"   Searched in: {args.results_dir}")
                print("\nTo get started:")
                print("   1. Run: python scripts/run_experiment.py --experiment quick_test")
                print("   2. Then: python scripts/view_results.py --latest")
                return 1
            compare_versions(args.results_dir)
            print(f"\nNext steps:")
            print(f"   • View latest: python scripts/view_results.py --latest")
            print(f"   • View specific: python scripts/view_results.py --version {versions[-1]}")
            print(f"   • Run new experiment: python scripts/run_experiment.py")
        return 0
    except KeyboardInterrupt:
        print("\nViewer interrupted by user")
        return 130
    except Exception as e:
        print(f"Viewing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())