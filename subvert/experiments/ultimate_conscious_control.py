"""
Ultimate conscious control experiment - incorporating all diagnostic findings.

Based on diagnostic results:
- Use neutral baseline corrections (v050 finding)
- Focus on AUROC validation (quality investigation)
- Test optimized instruction strengths (v051 analysis)
- Use multiple color pairs with dedicated probes (v052 color analysis)
- Implement statistical significance testing
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats

from ..core.dual_probe_collection import collect_conscious_control_data
from ..core.dual_probe_training import DualProbeTrainer
from ..utils.color_prompts import generate_balanced_color_dataset
from .versioning import create_experiment_session, log_experiment_config


def run_ultimate_conscious_control_experiment(
    color_pairs: List[Tuple[str, str]] = None,
    n_training: int = 1500,
    n_condition: int = 400,
    layers: List[str] = None,
    min_auroc_threshold: float = 0.90,
    statistical_alpha: float = 0.05,
    include_cross_color_analysis: bool = True
) -> Dict:
    """
    Run the ultimate conscious control experiment with all optimizations.
    
    Args:
        color_pairs: List of (color1, color2) pairs to test
        n_training: Training samples per color pair
        n_condition: Test samples per condition
        layers: Hook points to collect from
        min_auroc_threshold: Minimum AUROC for probe validation (replaces F1)
        statistical_alpha: Significance level for statistical tests
        include_cross_color_analysis: Whether to test cross-color effects
        
    Returns:
        Comprehensive experiment results
    """
    if color_pairs is None:
        # Optimized pairs based on v052 findings
        color_pairs = [
            ('blue', 'red'),      # Baseline comparison
            ('white', 'black'),   # Highest asymmetry (11.16x)
            ('green', 'purple'),  # Moderate asymmetry (good control)
            ('orange', 'yellow'), # Warm colors (interesting dynamics)
        ]
    
    if layers is None:
        layers = ["blocks.15.hook_resid_post", "blocks.31.hook_resid_post"]
    
    print(f"🚀 ULTIMATE CONSCIOUS CONTROL EXPERIMENT")
    print(f"Color pairs: {len(color_pairs)}")
    print(f"Training samples per pair: {n_training}")
    print(f"Test samples per condition: {n_condition}")
    print(f"AUROC threshold: {min_auroc_threshold}")
    print(f"Statistical alpha: {statistical_alpha}")
    
    # Get version
    version, version_dir = create_experiment_session()
    version_dir = Path(version_dir)
    
    # Create ultimate experiment subdirectory
    ult_results_dir = version_dir.parent / "ultimate_conscious_control"
    ult_results_dir.mkdir(exist_ok=True)
    ult_version_dir = ult_results_dir / f"v{version[1:]:0>3}"
    ult_version_dir.mkdir(exist_ok=True)
    version_dir = ult_version_dir
    
    print(f"Experiment version: {version}")
    
    # Save configuration
    config = {
        'color_pairs': color_pairs,
        'n_training': n_training,
        'n_condition': n_condition,
        'layers': layers,
        'min_auroc_threshold': min_auroc_threshold,
        'statistical_alpha': statistical_alpha,
        'include_cross_color_analysis': include_cross_color_analysis,
        'optimizations_applied': [
            'neutral_baseline_correction',
            'auroc_based_validation',
            'instruction_strength_optimization',
            'dedicated_color_probes',
            'statistical_significance_testing'
        ]
    }
    log_experiment_config(config, version)
    
    results = {
        'version': int(version[1:]),
        'version_dir': str(version_dir),
        'config': config,
        'color_pair_results': {},
        'cross_color_analysis': {},
        'summary_statistics': {},
        'statistical_tests': {}
    }
    
    # Optimized instructions based on v051 analysis
    optimized_instructions = {
        'neutral_control': 'You are a helpful assistant.',
        'neutral_alternative': 'Please provide helpful and accurate responses.',
        'emphasis_color1': 'Focus on the color {color1}.',  # Most effective
        'emphasis_color2': 'Focus on the color {color2}.',
        'dual_color1': 'Think about {color1} without thinking of {color2}.',
        'dual_color2': 'Think about {color2} without thinking of {color1}.',
        'strong_emphasis_color1': 'Concentrate deeply on the beautiful color {color1}.',
        'strong_emphasis_color2': 'Concentrate deeply on the beautiful color {color2}.',
    }
    
    all_effect_sizes = []
    all_asymmetry_ratios = []
    
    for color1, color2 in color_pairs:
        print(f"\n{'='*60}")
        print(f"🎨 TESTING {color1.upper()} vs {color2.upper()}")
        print(f"{'='*60}")
        
        # Generate training dataset with optimal balance
        training_dataset = generate_balanced_color_dataset(
            n_total=n_training,
            blue_ratio=0.3,   # Increased from 0.25 for better probe quality
            red_ratio=0.3,
            neutral_ratio=0.4,
            random_seed=42,
            color_overrides={'blue': color1, 'red': color2}
        )
        
        training_prompts_file = version_dir / f"{color1}_{color2}_training.json"
        with open(training_prompts_file, 'w') as f:
            json.dump(training_dataset['all_prompts'], f)
        
        # Collect baseline training data
        print(f"📊 Training dedicated {color1}/{color2} probes...")
        baseline_train_file = version_dir / f"{color1}_{color2}_baseline_train.pt"
        collect_conscious_control_data(
            prompts_file=str(training_prompts_file),
            output_file=str(baseline_train_file),
            system_prompt="You are a helpful assistant.",
            hook_points=layers,
            batch_size=150  # Optimized batch size
        )
        
        # Train with AUROC-based validation
        trainer = DualProbeTrainer(
            feature_standardization=True,
            cross_validation_folds=5,
            random_state=42,
            # Override F1 validation with AUROC
            min_auroc=min_auroc_threshold,
            validate_f1=False  # Disable F1 validation
        )
        
        baseline_data = trainer.load_dual_probe_data(str(baseline_train_file))
        baseline_results = trainer.train_dual_probes(baseline_data, test_size=0.25)
        
        # Enhanced probe validation
        color1_auroc = baseline_results['blue_probe']['test_auroc']  # color1 uses blue probe
        color2_auroc = baseline_results['red_probe']['test_auroc']   # color2 uses red probe
        
        probe_quality_pass = (color1_auroc >= min_auroc_threshold and 
                             color2_auroc >= min_auroc_threshold)
        
        if not probe_quality_pass:
            print(f"⚠️  WARNING: Probe quality below threshold!")
            print(f"   {color1} AUROC: {color1_auroc:.4f} (need ≥{min_auroc_threshold})")
            print(f"   {color2} AUROC: {color2_auroc:.4f} (need ≥{min_auroc_threshold})")
        else:
            print(f"✅ Probe quality: {color1}={color1_auroc:.4f}, {color2}={color2_auroc:.4f}")
        
        baseline_probe_file = version_dir / f"{color1}_{color2}_probes.pkl"
        trainer.save_dual_probes(baseline_results, str(baseline_probe_file))
        
        # Generate experimental prompts
        exp_dataset = generate_balanced_color_dataset(
            n_total=n_condition,
            blue_ratio=0.05,   # Minimal color mentions for cleaner testing
            red_ratio=0.05,
            neutral_ratio=0.9,
            random_seed=43,
            color_overrides={'blue': color1, 'red': color2}
        )
        
        exp_prompts_file = version_dir / f"{color1}_{color2}_exp.json"
        with open(exp_prompts_file, 'w') as f:
            json.dump(exp_dataset['all_prompts'], f)
        
        # Test all optimized instructions
        pair_results = {
            'color1': color1,
            'color2': color2,
            'baseline_quality': {
                'color1_auroc': float(color1_auroc),
                'color2_auroc': float(color2_auroc),
                'quality_pass': probe_quality_pass,
                'probe_correlation': float(baseline_results['dual_validation']['prediction_correlation'])
            },
            'instructions': {},
            'bootstrap_stats': {}
        }
        
        instruction_differentials = {}
        
        for instruction_name, instruction_template in optimized_instructions.items():
            instruction_text = instruction_template.format(color1=color1, color2=color2)
            print(f"  🧪 Testing: {instruction_name}")
            
            exp_file = version_dir / f"{color1}_{color2}_{instruction_name}.pt"
            collect_conscious_control_data(
                prompts_file=str(exp_prompts_file),
                output_file=str(exp_file),
                system_prompt=instruction_text,
                hook_points=layers,
                batch_size=100
            )
            
            eval_results = trainer.evaluate_dual_probes_on_data(
                probe_file=str(baseline_probe_file),
                data_file=str(exp_file)
            )
            
            differential = eval_results['differential']['mean_differential']
            instruction_differentials[instruction_name] = differential
            
            # Bootstrap confidence intervals
            n_bootstrap = 1000
            bootstrap_diffs = []
            original_diff = eval_results['differential']['blue_minus_red_preds']
            
            for _ in range(n_bootstrap):
                boot_sample = np.random.choice(original_diff, size=len(original_diff), replace=True)
                bootstrap_diffs.append(np.mean(boot_sample))
            
            ci_lower = np.percentile(bootstrap_diffs, 2.5)
            ci_upper = np.percentile(bootstrap_diffs, 97.5)
            
            pair_results['instructions'][instruction_name] = {
                'system_prompt': instruction_text,
                'color1_auroc': float(eval_results['blue']['auroc']),
                'color2_auroc': float(eval_results['red']['auroc']),
                'mean_differential': float(differential),
                'bootstrap_ci': [float(ci_lower), float(ci_upper)],
                'effect_significant': not (ci_lower <= 0 <= ci_upper)
            }
        
        # Neutral baseline correction (v050 methodology)
        neutral_baseline = instruction_differentials['neutral_control']
        alt_neutral = instruction_differentials['neutral_alternative']
        avg_neutral = (neutral_baseline + alt_neutral) / 2
        
        print(f"  📊 Neutral baselines: {neutral_baseline:+.4f}, {alt_neutral:+.4f} (avg: {avg_neutral:+.4f})")
        
        # Calculate corrected effects
        corrected_effects = {}
        for instruction, raw_diff in instruction_differentials.items():
            if 'neutral' not in instruction:
                corrected_effects[instruction] = raw_diff - avg_neutral
        
        # Primary conscious control analysis
        color1_emphasis_corrected = corrected_effects.get('emphasis_color1', 0)
        color2_emphasis_corrected = corrected_effects.get('emphasis_color2', 0)
        color1_dual_corrected = corrected_effects.get('dual_color1', 0)
        color2_dual_corrected = corrected_effects.get('dual_color2', 0)
        
        # Enhanced asymmetry analysis
        emphasis_asymmetry = abs(color2_emphasis_corrected) / max(abs(color1_emphasis_corrected), 0.001)
        dual_asymmetry = abs(color2_dual_corrected) / max(abs(color1_dual_corrected), 0.001)
        
        # Effect size calculation (Cohen's d equivalent)
        emphasis_effect_size = abs(color1_emphasis_corrected) + abs(color2_emphasis_corrected)
        dual_effect_size = abs(color1_dual_corrected) + abs(color2_dual_corrected)
        
        # Conscious control detection with enhanced criteria
        emphasis_conscious_control = (
            abs(color1_emphasis_corrected) > 0.05 or abs(color2_emphasis_corrected) > 0.05
        )
        dual_conscious_control = (
            color1_dual_corrected > 0.05 and color2_dual_corrected < -0.05
        ) or (
            color1_dual_corrected < -0.05 and color2_dual_corrected > 0.05
        )
        
        pair_results['analysis'] = {
            'neutral_baseline_avg': float(avg_neutral),
            'corrected_effects': {k: float(v) for k, v in corrected_effects.items()},
            'emphasis_asymmetry_ratio': float(emphasis_asymmetry),
            'dual_asymmetry_ratio': float(dual_asymmetry),
            'emphasis_effect_size': float(emphasis_effect_size),
            'dual_effect_size': float(dual_effect_size),
            'emphasis_conscious_control': emphasis_conscious_control,
            'dual_conscious_control': dual_conscious_control,
            'overall_conscious_control': emphasis_conscious_control or dual_conscious_control,
            'dominant_color_emphasis': color2 if abs(color2_emphasis_corrected) > abs(color1_emphasis_corrected) else color1,
            'dominant_color_dual': color2 if abs(color2_dual_corrected) > abs(color1_dual_corrected) else color1
        }
        
        # Statistical significance testing
        t_stat_emphasis, p_val_emphasis = stats.ttest_1samp(
            [pair_results['instructions']['emphasis_color1']['mean_differential'],
             pair_results['instructions']['emphasis_color2']['mean_differential']], 
            avg_neutral
        )
        
        pair_results['statistical_tests'] = {
            'emphasis_t_statistic': float(t_stat_emphasis),
            'emphasis_p_value': float(p_val_emphasis),
            'emphasis_significant': p_val_emphasis < statistical_alpha,
        }
        
        results['color_pair_results'][f"{color1}_{color2}"] = pair_results
        all_effect_sizes.append(emphasis_effect_size)
        all_asymmetry_ratios.append(emphasis_asymmetry)
        
        print(f"  🎯 Results: Emphasis CC={'YES' if emphasis_conscious_control else 'NO'}, "
              f"Dual CC={'YES' if dual_conscious_control else 'NO'}")
        print(f"  📈 Effect sizes: Emphasis={emphasis_effect_size:.4f}, Dual={dual_effect_size:.4f}")
        
    # Cross-color analysis
    if include_cross_color_analysis and len(color_pairs) > 1:
        print(f"\n🔄 CROSS-COLOR ANALYSIS")
        # Test how well red/blue probes work on other colors
        red_blue_probe_file = version_dir / "blue_red_probes.pkl"
        if red_blue_probe_file.exists():
            cross_results = _run_cross_color_analysis(
                color_pairs, red_blue_probe_file, version_dir, trainer, n_condition
            )
            results['cross_color_analysis'] = cross_results
    
    # Summary statistics
    results['summary_statistics'] = {
        'total_color_pairs_tested': len(color_pairs),
        'avg_effect_size': float(np.mean(all_effect_sizes)),
        'max_effect_size': float(np.max(all_effect_sizes)),
        'avg_asymmetry_ratio': float(np.mean(all_asymmetry_ratios)),
        'max_asymmetry_ratio': float(np.max(all_asymmetry_ratios)),
        'pairs_with_conscious_control': sum(1 for pair_data in results['color_pair_results'].values() 
                                          if pair_data['analysis']['overall_conscious_control']),
        'overall_success_rate': sum(1 for pair_data in results['color_pair_results'].values() 
                                  if pair_data['analysis']['overall_conscious_control']) / len(color_pairs)
    }
    
    # Save comprehensive results
    results_file = version_dir / "ultimate_conscious_control_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🏆 ULTIMATE EXPERIMENT COMPLETE")
    print(f"Success rate: {results['summary_statistics']['overall_success_rate']:.1%}")
    print(f"Average effect size: {results['summary_statistics']['avg_effect_size']:.4f}")
    print(f"Results saved to: {results_file}")
    
    return results


def _run_cross_color_analysis(color_pairs, red_blue_probe_file, version_dir, trainer, n_condition):
    """Test cross-color probe transfer."""
    print("Testing red/blue probe transfer to other color pairs...")
    
    cross_results = {}
    
    for color1, color2 in color_pairs:
        if color1 == 'blue' and color2 == 'red':
            continue  # Skip the original pair
            
        # Test emphasis instruction only (most reliable)
        exp_dataset = generate_balanced_color_dataset(
            n_total=n_condition,
            blue_ratio=0.05,
            red_ratio=0.05,
            neutral_ratio=0.9,
            random_seed=43,
            color_overrides={'blue': color1, 'red': color2}
        )
        
        exp_prompts_file = version_dir / f"cross_{color1}_{color2}_prompts.json"
        with open(exp_prompts_file, 'w') as f:
            json.dump(exp_dataset['all_prompts'], f)
        
        # Test with red/blue probes
        exp_file = version_dir / f"cross_{color1}_{color2}_emphasis.pt"
        collect_conscious_control_data(
            prompts_file=str(exp_prompts_file),
            output_file=str(exp_file),
            system_prompt=f"Focus on the color {color1}.",
            hook_points=["blocks.31.hook_resid_post"],
            batch_size=100
        )
        
        eval_results = trainer.evaluate_dual_probes_on_data(
            probe_file=str(red_blue_probe_file),
            data_file=str(exp_file)
        )
        
        cross_results[f"{color1}_{color2}"] = {
            'cross_transfer_differential': float(eval_results['differential']['mean_differential']),
            'note': 'Using red/blue probes on other colors'
        }
        
        print(f"  {color1} emphasis with red/blue probes: {eval_results['differential']['mean_differential']:+.4f}")
    
    return cross_results