"""
Color pair asymmetry analysis - test different color combinations.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from ..core.dual_probe_collection import collect_conscious_control_data
from ..core.dual_probe_training import DualProbeTrainer
from ..utils.color_prompts import generate_balanced_color_dataset
from .versioning import create_experiment_session, log_experiment_config


def run_color_pair_analysis(color_pairs: List[Tuple[str, str]], 
                           n_training: int = 800,
                           n_condition: int = 150) -> Dict:
    """
    Test instruction asymmetry across different color pairs.
    
    Args:
        color_pairs: List of (color1, color2) tuples to test
        n_training: Number of training prompts per pair
        n_condition: Number of prompts per condition
        
    Returns:
        Analysis results for all color pairs
    """
    print(f"Testing instruction asymmetry across {len(color_pairs)} color pairs...")
    
    # Get version
    version, version_dir = create_experiment_session()
    version_dir = Path(version_dir)
    
    # Create color_pairs subdirectory
    cp_results_dir = version_dir.parent / "color_pairs" 
    cp_results_dir.mkdir(exist_ok=True)
    cp_version_dir = cp_results_dir / f"v{version[1:]:0>3}"
    cp_version_dir.mkdir(exist_ok=True)
    version_dir = cp_version_dir
    
    print(f"Experiment version: {version}")
    
    results = {
        'version': int(version[1:]),
        'version_dir': str(version_dir),
        'color_pairs': {},
        'summary': {}
    }
    
    for color1, color2 in color_pairs:
        print(f"\n=== Testing {color1.upper()} vs {color2.upper()} ===")
        
        # Generate training dataset with both colors
        training_dataset = generate_balanced_color_dataset(
            n_total=n_training,
            blue_ratio=0.25,  # Use as color1 ratio
            red_ratio=0.25,   # Use as color2 ratio  
            neutral_ratio=0.5,
            random_seed=42,
            color_overrides={'blue': color1, 'red': color2}
        )
        
        training_prompts_file = version_dir / f"{color1}_{color2}_training_prompts.json"
        with open(training_prompts_file, 'w') as f:
            json.dump(training_dataset['all_prompts'], f)
        
        # Collect baseline training data
        print(f"Training probes for {color1}/{color2}...")
        baseline_train_file = version_dir / f"{color1}_{color2}_baseline_train.pt"
        collect_conscious_control_data(
            prompts_file=str(training_prompts_file),
            output_file=str(baseline_train_file),
            system_prompt="You are a helpful assistant.",
            hook_points=["blocks.31.hook_resid_post"],
            batch_size=100
        )
        
        # Train baseline probes
        trainer = DualProbeTrainer(random_state=42)
        baseline_data = trainer.load_dual_probe_data(str(baseline_train_file))
        baseline_results = trainer.train_dual_probes(baseline_data)
        
        baseline_probe_file = version_dir / f"{color1}_{color2}_baseline_probes.pkl"
        trainer.save_dual_probes(baseline_results, str(baseline_probe_file))
        
        # Generate experimental prompts
        exp_dataset = generate_balanced_color_dataset(
            n_total=n_condition,
            blue_ratio=0.1,
            red_ratio=0.1, 
            neutral_ratio=0.8,
            random_seed=43,
            color_overrides={'blue': color1, 'red': color2}
        )
        
        exp_prompts_file = version_dir / f"{color1}_{color2}_exp_prompts.json"
        with open(exp_prompts_file, 'w') as f:
            json.dump(exp_dataset['all_prompts'], f)
        
        # Test key instruction types
        instructions = [
            ('neutral', 'You are a helpful assistant.'),
            ('mention_color1', f'Think about {color1}.'),
            ('mention_color2', f'Think about {color2}.'),
            ('emphasize_color1', f'Focus on the color {color1}.'),
            ('emphasize_color2', f'Focus on the color {color2}.'),
            ('color1_not_color2', f'Think about {color1} without thinking of {color2}.'),
            ('color2_not_color1', f'Think about {color2} without thinking of {color1}.'),
        ]
        
        pair_results = {
            'color1': color1,
            'color2': color2,
            'baseline_auroc': {
                'color1': float(baseline_results['blue_probe']['test_auroc']),  # color1 uses blue probe
                'color2': float(baseline_results['red_probe']['test_auroc'])    # color2 uses red probe
            },
            'instructions': {}
        }
        
        for instruction_name, system_prompt in instructions:
            print(f"  Testing: {instruction_name}")
            
            exp_file = version_dir / f"{color1}_{color2}_{instruction_name}_activations.pt"
            collect_conscious_control_data(
                prompts_file=str(exp_prompts_file),
                output_file=str(exp_file),
                system_prompt=system_prompt,
                hook_points=["blocks.31.hook_resid_post"],
                batch_size=100
            )
            
            eval_results = trainer.evaluate_dual_probes_on_data(
                probe_file=str(baseline_probe_file),
                data_file=str(exp_file)
            )
            
            pair_results['instructions'][instruction_name] = {
                'system_prompt': system_prompt,
                'color1_auroc': float(eval_results['blue']['auroc']),
                'color2_auroc': float(eval_results['red']['auroc']),
                'mean_differential': float(eval_results['differential']['mean_differential'])
            }
        
        # Calculate asymmetry metrics
        neutral_diff = pair_results['instructions']['neutral']['mean_differential']
        color1_emphasis = pair_results['instructions']['emphasize_color1']['mean_differential'] - neutral_diff
        color2_emphasis = pair_results['instructions']['emphasize_color2']['mean_differential'] - neutral_diff
        color1_dual = pair_results['instructions']['color1_not_color2']['mean_differential'] - neutral_diff
        color2_dual = pair_results['instructions']['color2_not_color1']['mean_differential'] - neutral_diff
        
        # Calculate asymmetry ratio (how much stronger is color2 vs color1)
        asymmetry_ratio = abs(color2_emphasis) / max(abs(color1_emphasis), 0.001)
        
        pair_results['analysis'] = {
            'neutral_baseline': float(neutral_diff),
            'color1_emphasis_corrected': float(color1_emphasis),
            'color2_emphasis_corrected': float(color2_emphasis),
            'color1_dual_corrected': float(color1_dual),
            'color2_dual_corrected': float(color2_dual),
            'asymmetry_ratio': float(asymmetry_ratio),
            'dominant_color': color2 if abs(color2_emphasis) > abs(color1_emphasis) else color1
        }
        
        results['color_pairs'][f"{color1}_{color2}"] = pair_results
        
        print(f"  {color1} emphasis: {color1_emphasis:+.4f}")
        print(f"  {color2} emphasis: {color2_emphasis:+.4f}")
        print(f"  Asymmetry ratio: {asymmetry_ratio:.2f}x")
        print(f"  Dominant color: {pair_results['analysis']['dominant_color']}")
    
    # Cross-pair summary analysis
    asymmetry_ratios = []
    dominant_colors = []
    
    for pair_name, pair_data in results['color_pairs'].items():
        asymmetry_ratios.append(pair_data['analysis']['asymmetry_ratio'])
        dominant_colors.append(pair_data['analysis']['dominant_color'])
    
    results['summary'] = {
        'avg_asymmetry_ratio': float(sum(asymmetry_ratios) / len(asymmetry_ratios)),
        'max_asymmetry_ratio': float(max(asymmetry_ratios)),
        'min_asymmetry_ratio': float(min(asymmetry_ratios)),
        'dominant_color_frequency': {color: dominant_colors.count(color) for color in set(dominant_colors)},
        'red_dominance_frequency': dominant_colors.count('red') if 'red' in dominant_colors else 0,
        'total_pairs_tested': len(color_pairs)
    }
    
    # Save results
    results_file = version_dir / "color_pair_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== COLOR PAIR ANALYSIS SUMMARY ===")
    print(f"Average asymmetry ratio: {results['summary']['avg_asymmetry_ratio']:.2f}x")
    print(f"Range: {results['summary']['min_asymmetry_ratio']:.2f}x - {results['summary']['max_asymmetry_ratio']:.2f}x")
    print(f"Dominant colors: {results['summary']['dominant_color_frequency']}")
    print(f"Results saved to: {results_file}")
    
    return results


def run_quick_color_pair_test(color_pairs: List[Tuple[str, str]], 
                             n_condition: int = 100) -> Dict:
    """
    Quick test using existing baseline probes from red/blue experiment.
    
    This is faster but less accurate since it uses red/blue probes for other colors.
    """
    print(f"Quick test of {len(color_pairs)} color pairs using existing probes...")
    
    # Use existing baseline probes (assumes v050 exists)
    baseline_probe_file = Path("/Users/yuva/Documents/Github/prompt-subversion-/results/conscious_control/v050/baseline_probes.pkl")
    
    if not baseline_probe_file.exists():
        print("Error: Need existing baseline probes. Run full analysis instead.")
        return {}
    
    # Get version
    version, version_dir = create_experiment_session()
    version_dir = Path(version_dir)
    
    cp_results_dir = version_dir.parent / "color_pairs_quick"
    cp_results_dir.mkdir(exist_ok=True)
    cp_version_dir = cp_results_dir / f"v{version[1:]:0>3}"
    cp_version_dir.mkdir(exist_ok=True)
    version_dir = cp_version_dir
    
    trainer = DualProbeTrainer(random_state=42)
    results = {
        'version': int(version[1:]),
        'version_dir': str(version_dir),
        'note': 'Quick test using red/blue probes - results approximate',
        'color_pairs': {}
    }
    
    for color1, color2 in color_pairs:
        print(f"Testing {color1} vs {color2}...")
        
        # Generate experimental prompts
        exp_dataset = generate_balanced_color_dataset(
            n_total=n_condition,
            blue_ratio=0.1,
            red_ratio=0.1,
            neutral_ratio=0.8,
            random_seed=43,
            color_overrides={'blue': color1, 'red': color2}
        )
        
        exp_prompts_file = version_dir / f"{color1}_{color2}_prompts.json"
        with open(exp_prompts_file, 'w') as f:
            json.dump(exp_dataset['all_prompts'], f)
        
        # Test emphasis instructions only (most reliable)
        instructions = [
            ('neutral', 'You are a helpful assistant.'),
            ('emphasize_color1', f'Focus on the color {color1}.'),
            ('emphasize_color2', f'Focus on the color {color2}.'),
        ]
        
        pair_results = {'color1': color1, 'color2': color2, 'instructions': {}}
        
        for instruction_name, system_prompt in instructions:
            exp_file = version_dir / f"{color1}_{color2}_{instruction_name}.pt"
            collect_conscious_control_data(
                prompts_file=str(exp_prompts_file),
                output_file=str(exp_file),
                system_prompt=system_prompt,
                hook_points=["blocks.31.hook_resid_post"],
                batch_size=100
            )
            
            eval_results = trainer.evaluate_dual_probes_on_data(
                probe_file=str(baseline_probe_file),
                data_file=str(exp_file)
            )
            
            pair_results['instructions'][instruction_name] = {
                'mean_differential': float(eval_results['differential']['mean_differential'])
            }
        
        # Calculate corrected effects
        neutral_diff = pair_results['instructions']['neutral']['mean_differential']
        color1_effect = pair_results['instructions']['emphasize_color1']['mean_differential'] - neutral_diff
        color2_effect = pair_results['instructions']['emphasize_color2']['mean_differential'] - neutral_diff
        
        pair_results['analysis'] = {
            'color1_effect': float(color1_effect),
            'color2_effect': float(color2_effect),
            'asymmetry_ratio': float(abs(color2_effect) / max(abs(color1_effect), 0.001)),
            'dominant_color': color2 if abs(color2_effect) > abs(color1_effect) else color1
        }
        
        results['color_pairs'][f"{color1}_{color2}"] = pair_results
        
        print(f"  {color1}: {color1_effect:+.4f}")
        print(f"  {color2}: {color2_effect:+.4f}")
        print(f"  Ratio: {pair_results['analysis']['asymmetry_ratio']:.2f}x")
    
    return results