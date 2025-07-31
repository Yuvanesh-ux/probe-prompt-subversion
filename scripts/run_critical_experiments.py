"""
Run critical methodological experiments for position effects and bidirectional control.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from subvert.experiments.position_semantic_experiments import (
    run_position_semantic_experiment, 
    run_position_semantic_suite
)
from subvert.experiments.enhancement_experiments import (
    run_enhancement_experiment,
    run_enhancement_suite
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run critical methodological experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Critical methodological tests:
  
Position Effects: Test if effects are positional bias vs semantic control
Bidirectional Control: Test for enhancement and suppression capabilities

Examples:
  # Test position vs semantic effects for blue/red
  python run_critical_experiments.py --experiment position --colors blue red
  
  # Test bidirectional control for multiple color pairs
  python run_critical_experiments.py --experiment bidirectional --suite
  
  # Run both critical experiments
  python run_critical_experiments.py --experiment both --suite
        """
    )

    parser.add_argument(
        "--experiment", "-e",
        choices=["position", "bidirectional", "both"],
        required=True,
        help="Which test to run: position effects, bidirectional control, or both"
    )

    parser.add_argument(
        "--colors",
        nargs=2,
        default=["blue", "red"],
        help="Color pair to test (default: blue red)"
    )

    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run on multiple color pairs (blue/red, green/purple)"
    )

    parser.add_argument(
        "--n_training",
        type=int,
        default=2000,
        help="Training samples for probe (default: 2000)"
    )

    parser.add_argument(
        "--n_condition", 
        type=int,
        default=500,
        help="Test samples per condition (default: 500)"
    )

    parser.add_argument(
        "--layer",
        default="blocks.31.hook_resid_post",
        help="Transformer layer to analyze (default: blocks.31.hook_resid_post)"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show configuration without running experiments"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Critical Methodological Experiments")
    logger.info("=" * 50)
    
    color_pair = tuple(args.colors)
    
    if args.suite:
        color_pairs = [('blue', 'red'), ('green', 'purple')]
        logger.info(f"Running suite on color pairs: {color_pairs}")
    else:
        logger.info(f"Testing single color pair: {color_pair}")

    logger.info(f"Training samples: {args.n_training}")
    logger.info(f"Test samples per condition: {args.n_condition}")
    logger.info(f"Analysis layer: {args.layer}")

    if args.dry_run:
        logger.info("\nDry run mode - configuration verified!")
        return 0

    logger.info("\n" + "=" * 50)

    try:
        if args.experiment == "position":
            logger.info("TESTING POSITION vs SEMANTIC EFFECTS")
            if args.suite:
                results = run_position_semantic_suite(
                    color_pairs=color_pairs,
                    n_training=args.n_training,
                    n_condition=args.n_condition,
                    layer=args.layer
                )
                logger.info(f"\nOverall finding: {results['suite_summary']['overall_conclusion']}")
            else:
                results = run_position_semantic_experiment(
                    color_pair=color_pair,
                    n_training=args.n_training,
                    n_condition=args.n_condition,
                    layer=args.layer
                )
                logger.info(f"\nFinding: {results['conclusion']['primary_mechanism']}")

        elif args.experiment == "bidirectional":
            logger.info("TESTING BIDIRECTIONAL CONTROL")
            if args.suite:
                results = run_enhancement_suite(
                    color_pairs=color_pairs,
                    n_training=args.n_training,
                    n_condition=args.n_condition,
                    layer=args.layer
                )
                logger.info(f"\nBidirectional control rate: {results['suite_summary']['overall_bidirectional_rate']:.1%}")
            else:
                results = run_enhancement_experiment(
                    color_pair=color_pair,
                    n_training=args.n_training,
                    n_condition=args.n_condition,
                    layer=args.layer
                )
                logger.info(f"\nBidirectional control: {results['conclusion']['bidirectional_control']}")

        elif args.experiment == "both":
            logger.info("RUNNING BOTH CRITICAL TESTS")
            
            logger.info("\n" + "-" * 30)
            logger.info("TEST 1: Position vs Semantic Effects")
            logger.info("-" * 30)
            
            if args.suite:
                position_results = run_position_semantic_suite(
                    color_pairs=color_pairs,
                    n_training=args.n_training,
                    n_condition=args.n_condition,
                    layer=args.layer
                )
                pos_finding = position_results['suite_summary']['overall_conclusion']
            else:
                position_results = run_position_semantic_experiment(
                    color_pair=color_pair,
                    n_training=args.n_training,
                    n_condition=args.n_condition,
                    layer=args.layer
                )
                pos_finding = position_results['conclusion']['primary_mechanism']

            logger.info("\n" + "-" * 30)
            logger.info("TEST 2: Bidirectional Control")
            logger.info("-" * 30)
            
            if args.suite:
                enhancement_results = run_enhancement_suite(
                    color_pairs=color_pairs,
                    n_training=args.n_training,
                    n_condition=args.n_condition,
                    layer=args.layer
                )
                enh_rate = enhancement_results['suite_summary']['overall_bidirectional_rate']
                enh_finding = f"{enh_rate:.1%} bidirectional control"
            else:
                enhancement_results = run_enhancement_experiment(
                    color_pair=color_pair,
                    n_training=args.n_training,
                    n_condition=args.n_condition,
                    layer=args.layer
                )
                enh_finding = enhancement_results['conclusion']['primary_finding']

            logger.info("\n" + "=" * 50)
            logger.info("METHODOLOGICAL FINDINGS")
            logger.info("=" * 50)
            logger.info(f"Position Effects: {pos_finding}")
            logger.info(f"Bidirectional Control: {enh_finding}")

        logger.info("\n" + "=" * 50)
        logger.info("CRITICAL EXPERIMENTS COMPLETED")
        logger.info("=" * 50)
        
        return 0

    except KeyboardInterrupt:
        logger.info("\nExperiments interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\nExperiments failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())