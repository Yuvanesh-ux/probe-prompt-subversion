#!/usr/bin/env python3
"""
Prompt generation CLI for Subvert experiments.

Generates balanced datasets of prompts for probe sabotage experiments.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports  
sys.path.append(str(Path(__file__).parent.parent))

from subvert.utils.prompts import generate_balanced_prompts


def create_parser():
    """Create argument parser for prompt generation."""
    parser = argparse.ArgumentParser(
        description="Generate balanced prompt datasets for experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 2000 balanced prompts
  python generate_prompts.py --n_prompts 2000 --output prompts/scaled_prompts_2000.json
  
  # Quick test set
  python generate_prompts.py --n_prompts 100 --output prompts/test_prompts.json
  
  # Custom concept
  python generate_prompts.py --concept tigers --n_prompts 1000
        """
    )

    parser.add_argument(
        "--n_prompts", "-n",
        type=int,
        default=2000,
        help="Total number of prompts to generate (default: 2000)"
    )
    
    parser.add_argument(
        "--concept",
        default="elephants",
        help="Target concept for prompts (default: elephants)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (default: auto-generated)"
    )
    
    parser.add_argument(
        "--balance_ratio",
        type=float,
        default=0.5,
        help="Ratio of concept-mentioning prompts (default: 0.5 for 50/50 split)"
    )
    
    parser.add_argument(
        "--templates_file",
        help="Path to custom prompt templates (JSON format)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser


def main():
    """Main entry point for prompt generation."""
    parser = create_parser()
    args = parser.parse_args()

    print("🎯 Subvert Prompt Generator")
    print("=" * 40)

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_dir = Path("prompts")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"scaled_prompts_{args.n_prompts}.json"

    # Configuration summary
    print(f"📊 Total prompts: {args.n_prompts}")
    print(f"🎯 Concept: {args.concept}")
    print(f"⚖️  Balance ratio: {args.balance_ratio:.1%} concept-mentioning")
    print(f"🎲 Random seed: {args.seed}")
    print(f"📁 Output: {output_file}")
    
    if args.verbose:
        concept_count = int(args.n_prompts * args.balance_ratio)
        neutral_count = args.n_prompts - concept_count
        print(f"   └─ {concept_count} {args.concept}-mentioning prompts")
        print(f"   └─ {neutral_count} neutral prompts")

    print("\n🏗️  Generating prompts...")

    try:
        # Generate prompts
        prompts = generate_balanced_prompts(
            n_prompts=args.n_prompts,
            concept=args.concept,
            balance_ratio=args.balance_ratio,
            templates_file=args.templates_file,
            random_seed=args.seed
        )

        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save prompts
        with open(output_file, 'w') as f:
            json.dump(prompts, f, indent=2)

        print(f"✅ Generated {len(prompts)} prompts")
        print(f"💾 Saved to {output_file}")

        # Show samples if verbose
        if args.verbose:
            print("\n📝 Sample prompts:")
            for i, prompt in enumerate(prompts[:5]):
                print(f"   {i+1}. {prompt}")
            if len(prompts) > 5:
                print(f"   ... and {len(prompts) - 5} more")

        return 0

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())