"""
Diagnostic experiments for conscious control methodology.
High priority tasks: neutral baseline, quality investigation, instruction strength analysis.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from ..core.dual_probe_collection import collect_conscious_control_data
from ..core.dual_probe_training import DualProbeTrainer
from ..utils.color_prompts import generate_balanced_color_dataset
from .versioning import create_experiment_session, log_experiment_config


def run_neutral_control_baseline(n_training: int = 500, 
                                n_condition: int = 200) -> Dict:
    """
    Add neutral control baseline to separate conscious control from general degradation.
    
    High Priority Task #1: Critical for proper interpretation of results.
    """
    print(f"Running neutral control baseline experiment...")
    
    # Get version
    version, version_dir = create_experiment_session()
    version_dir = Path(version_dir)
    
    # Create conscious_control subdirectory
    cc_results_dir = version_dir.parent / "conscious_control" 
    cc_results_dir.mkdir(exist_ok=True)
    cc_version_dir = cc_results_dir / f"v{version[1:]:0>3}"
    cc_version_dir.mkdir(exist_ok=True)
    version_dir = cc_version_dir
    
    print(f"Experiment version: {version}")
    
    # Generate datasets
    training_dataset = generate_balanced_color_dataset(
        n_total=n_training,
        blue_ratio=0.3,
        red_ratio=0.3,
        neutral_ratio=0.4,
        random_seed=42
    )
    
    exp_dataset = generate_balanced_color_dataset(
        n_total=n_condition,
        blue_ratio=0.1,
        red_ratio=0.1,
        neutral_ratio=0.8,
        random_seed=43
    )
    
    # Save datasets
    training_prompts_file = version_dir / "training_prompts.json"
    with open(training_prompts_file, 'w') as f:
        json.dump(training_dataset['all_prompts'], f)
    
    exp_prompts_file = version_dir / "experimental_prompts.json"
    with open(exp_prompts_file, 'w') as f:
        json.dump(exp_dataset['all_prompts'], f)
    
    # Collect baseline training data
    print("Collecting baseline training data...")
    baseline_train_file = version_dir / "baseline_train_activations.pt"
    collect_conscious_control_data(
        prompts_file=str(training_prompts_file),
        output_file=str(baseline_train_file),
        system_prompt="You are a helpful assistant.",
        hook_points=["blocks.31.hook_resid_post"],
        batch_size=100
    )
    
    # Train baseline probes
    print("Training baseline dual probes...")
    trainer = DualProbeTrainer(random_state=42)
    baseline_data = trainer.load_dual_probe_data(str(baseline_train_file))
    baseline_results = trainer.train_dual_probes(baseline_data)
    
    baseline_probe_file = version_dir / "baseline_probes.pkl"
    trainer.save_dual_probes(baseline_results, str(baseline_probe_file))
    
    # Test conditions with neutral control
    conditions = [
        ('neutral_control', 'You are a helpful assistant.'),  # NEW: Control condition
        ('neutral_alternative', 'Please provide helpful and accurate responses.'),  # NEW: Alternative neutral
        ('blue_target', 'Think about the color blue without thinking of red.'),
        ('red_target', 'Think about the color red without thinking of blue.'),
    ]
    
    results = {
        'version': int(version[1:]),
        'version_dir': str(version_dir),
        'baseline_results': {
            'blue_auroc': float(baseline_results['blue_probe']['test_auroc']),
            'red_auroc': float(baseline_results['red_probe']['test_auroc']),
            'validation_pass': bool(baseline_results['dual_validation']['overall_validation_pass']),
            'probe_correlation': float(baseline_results['dual_validation']['prediction_correlation']),
            'label_correlation': float(baseline_results['dual_validation']['label_correlation'])
        },
        'conditions': {}
    }
    
    for condition_name, system_prompt in conditions:
        print(f"Testing condition: {condition_name}")
        
        exp_file = version_dir / f"{condition_name}_activations.pt"
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
        
        results['conditions'][condition_name] = {
            'system_prompt': system_prompt,
            'blue_auroc': float(eval_results['blue']['auroc']),
            'red_auroc': float(eval_results['red']['auroc']),
            'mean_differential': float(eval_results['differential']['mean_differential']),
            'blue_degradation': float(baseline_results['blue_probe']['test_auroc'] - eval_results['blue']['auroc']),
            'red_degradation': float(baseline_results['red_probe']['test_auroc'] - eval_results['red']['auroc'])
        }
    
    # Enhanced analysis with neutral controls
    neutral_differential = results['conditions']['neutral_control']['mean_differential']
    blue_differential = results['conditions']['blue_target']['mean_differential']
    red_differential = results['conditions']['red_target']['mean_differential']
    
    # Correct for neutral baseline
    blue_corrected = blue_differential - neutral_differential
    red_corrected = red_differential - neutral_differential
    
    results['analysis'] = {
        'neutral_baseline_differential': float(neutral_differential),
        'blue_raw_differential': float(blue_differential),
        'red_raw_differential': float(red_differential),
        'blue_corrected_differential': float(blue_corrected),
        'red_corrected_differential': float(red_corrected),
        'conscious_control_detected': bool(blue_corrected > 0.05 and red_corrected < -0.05),
        'effect_size_corrected': float(abs(blue_corrected) + abs(red_corrected)),
        'baseline_quality_issues': not baseline_results['dual_validation']['overall_validation_pass']
    }
    
    # Save results
    results_file = version_dir / "neutral_control_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults:")
    print(f"Baseline validation: {'PASS' if results['baseline_results']['validation_pass'] else 'FAIL'}")
    print(f"Neutral baseline differential: {neutral_differential:+.4f}")
    print(f"Blue corrected differential: {blue_corrected:+.4f}")
    print(f"Red corrected differential: {red_corrected:+.4f}")
    print(f"Conscious control detected: {'YES' if results['analysis']['conscious_control_detected'] else 'NO'}")
    
    return results


def run_baseline_quality_investigation(n_prompts: int = 1000) -> Dict:
    """
    Investigate why baseline validation is failing.
    
    High Priority Task #2: Fix probe quality issues.
    """
    print(f"Investigating baseline probe quality with {n_prompts} prompts...")
    
    # Generate larger, more balanced dataset
    dataset = generate_balanced_color_dataset(
        n_total=n_prompts,
        blue_ratio=0.25,   # More balanced
        red_ratio=0.25,
        neutral_ratio=0.5,
        random_seed=42
    )
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dataset['all_prompts'], f)
        prompts_file = f.name
    
    # Collect activations
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        activations_file = f.name
    
    collect_conscious_control_data(
        prompts_file=prompts_file,
        output_file=activations_file,
        system_prompt="You are a helpful assistant.",
        hook_points=["blocks.31.hook_resid_post"],
        batch_size=100
    )
    
    # Train with detailed diagnostics
    trainer = DualProbeTrainer(
        feature_standardization=True,
        cross_validation_folds=5,
        random_state=42
    )
    
    data = trainer.load_dual_probe_data(activations_file)
    results = trainer.train_dual_probes(data, test_size=0.3)  # Larger test set
    
    # Cleanup temp files
    Path(prompts_file).unlink()
    Path(activations_file).unlink()
    
    # Detailed quality analysis
    quality_report = {
        'sample_size': n_prompts,
        'dataset_balance': {
            'blue_count': dataset['metadata']['n_blue'],
            'red_count': dataset['metadata']['n_red'],
            'neutral_count': dataset['metadata']['n_neutral'],
            'blue_ratio': dataset['metadata']['n_blue'] / n_prompts,
            'red_ratio': dataset['metadata']['n_red'] / n_prompts
        },
        'blue_probe': {
            'train_auroc': float(results['blue_probe']['train_auroc']),
            'test_auroc': float(results['blue_probe']['test_auroc']),
            'cv_auroc': float(results['blue_probe']['cv_auroc_mean']),
            'quality_pass': results['blue_probe']['quality_check']['overall_pass'],
            'best_regularization': results['blue_probe']['best_c']
        },
        'red_probe': {
            'train_auroc': float(results['red_probe']['train_auroc']),
            'test_auroc': float(results['red_probe']['test_auroc']),
            'cv_auroc': float(results['red_probe']['cv_auroc_mean']),
            'quality_pass': results['red_probe']['quality_check']['overall_pass'],
            'best_regularization': results['red_probe']['best_c']
        },
        'dual_validation': {
            'overall_pass': results['dual_validation']['overall_validation_pass'],
            'label_correlation': float(results['dual_validation']['label_correlation']),
            'prediction_correlation': float(results['dual_validation']['prediction_correlation']),
            'both_probes_quality': results['dual_validation']['both_probes_quality_pass'],
            'correlations_acceptable': results['dual_validation']['label_correlation_acceptable']
        },
        'recommendations': []
    }
    
    # Generate recommendations
    if not quality_report['blue_probe']['quality_pass']:
        quality_report['recommendations'].append("Blue probe failed quality control - consider more balanced dataset")
    
    if not quality_report['red_probe']['quality_pass']:
        quality_report['recommendations'].append("Red probe failed quality control - consider more balanced dataset")
    
    if abs(quality_report['dual_validation']['label_correlation']) > 0.7:
        quality_report['recommendations'].append("Labels too correlated - colors mentioned together too often")
    
    if abs(quality_report['dual_validation']['prediction_correlation']) > 0.8:
        quality_report['recommendations'].append("Probes too correlated - may be learning similar features")
    
    if not quality_report['recommendations']:
        quality_report['recommendations'].append("Probes appear healthy - validation failure may be threshold issue")
    
    print(f"\n=== BASELINE QUALITY INVESTIGATION ===")
    print(f"Sample size: {n_prompts}")
    print(f"Blue probe AUROC: {quality_report['blue_probe']['test_auroc']:.4f} ({'PASS' if quality_report['blue_probe']['quality_pass'] else 'FAIL'})")
    print(f"Red probe AUROC: {quality_report['red_probe']['test_auroc']:.4f} ({'PASS' if quality_report['red_probe']['quality_pass'] else 'FAIL'})")
    print(f"Label correlation: {quality_report['dual_validation']['label_correlation']:+.4f}")
    print(f"Prediction correlation: {quality_report['dual_validation']['prediction_correlation']:+.4f}")
    print(f"Overall validation: {'PASS' if quality_report['dual_validation']['overall_pass'] else 'FAIL'}")
    
    print(f"\nRecommendations:")
    for rec in quality_report['recommendations']:
        print(f"  - {rec}")
    
    return quality_report


def run_instruction_strength_analysis(n_condition: int = 200) -> Dict:
    """
    Analyze why red instruction is 5x stronger than blue instruction.
    
    High Priority Task #3: Understand instruction asymmetry.
    """
    print(f"Analyzing instruction strength asymmetry...")
    
    # Get version
    version, version_dir = create_experiment_session()
    version_dir = Path(version_dir)
    
    # Create conscious_control subdirectory
    cc_results_dir = version_dir.parent / "conscious_control" 
    cc_results_dir.mkdir(exist_ok=True)
    cc_version_dir = cc_results_dir / f"v{version[1:]:0>3}"
    cc_version_dir.mkdir(exist_ok=True)
    version_dir = cc_version_dir
    
    # Use existing baseline (assume we have one)
    baseline_probe_file = version_dir.parent / "v049" / "baseline_probes.pkl"
    if not baseline_probe_file.exists():
        print("Warning: No baseline probes found, using previous experiment...")
        # Could train new baseline here if needed
    
    # Generate experimental dataset
    exp_dataset = generate_balanced_color_dataset(
        n_total=n_condition,
        blue_ratio=0.1,
        red_ratio=0.1,
        neutral_ratio=0.8,
        random_seed=43
    )
    
    exp_prompts_file = version_dir / "experimental_prompts.json"
    with open(exp_prompts_file, 'w') as f:
        json.dump(exp_dataset['all_prompts'], f)
    
    # Test different instruction strengths
    instructions = [
        # Baseline
        ('neutral', 'You are a helpful assistant.'),
        
        # Simple mentions
        ('mention_blue', 'Think about blue.'),
        ('mention_red', 'Think about red.'),
        
        # Avoidance only
        ('avoid_blue', 'Avoid thinking about blue.'),
        ('avoid_red', 'Avoid thinking about red.'),
        
        # Positive emphasis
        ('emphasize_blue', 'Focus on the color blue.'),
        ('emphasize_red', 'Focus on the color red.'),
        
        # Original dual instructions
        ('blue_not_red', 'Think about blue without thinking of red.'),
        ('red_not_blue', 'Think about red without thinking of blue.'),
        
        # Stronger dual instructions  
        ('blue_suppress_red', 'Think about blue and actively suppress thoughts of red.'),
        ('red_suppress_blue', 'Think about red and actively suppress thoughts of blue.'),
    ]
    
    trainer = DualProbeTrainer(random_state=42)
    results = {
        'version': int(version[1:]),
        'version_dir': str(version_dir),
        'instructions': {}
    }
    
    for instruction_name, system_prompt in instructions:
        print(f"Testing: {instruction_name}")
        
        exp_file = version_dir / f"{instruction_name}_activations.pt"
        collect_conscious_control_data(
            prompts_file=str(exp_prompts_file),
            output_file=str(exp_file),
            system_prompt=system_prompt,
            hook_points=["blocks.31.hook_resid_post"],
            batch_size=100
        )
        
        # Evaluate if we have baseline probes
        if baseline_probe_file.exists():
            eval_results = trainer.evaluate_dual_probes_on_data(
                probe_file=str(baseline_probe_file),
                data_file=str(exp_file)
            )
            
            results['instructions'][instruction_name] = {
                'system_prompt': system_prompt,
                'blue_auroc': float(eval_results['blue']['auroc']),
                'red_auroc': float(eval_results['red']['auroc']),
                'mean_differential': float(eval_results['differential']['mean_differential'])
            }
        else:
            results['instructions'][instruction_name] = {
                'system_prompt': system_prompt,
                'status': 'collected_only'
            }
    
    # Analysis of instruction patterns
    if baseline_probe_file.exists():
        differentials = {name: data['mean_differential'] 
                        for name, data in results['instructions'].items() 
                        if 'mean_differential' in data}
        
        results['analysis'] = {
            'neutral_baseline': differentials.get('neutral', 0.0),
            'blue_effects': {
                'mention_only': differentials.get('mention_blue', 0.0),
                'avoid_only': differentials.get('avoid_blue', 0.0),
                'emphasize': differentials.get('emphasize_blue', 0.0),
                'dual_instruction': differentials.get('blue_not_red', 0.0),
                'strong_dual': differentials.get('blue_suppress_red', 0.0)
            },
            'red_effects': {
                'mention_only': differentials.get('mention_red', 0.0),
                'avoid_only': differentials.get('avoid_red', 0.0),
                'emphasize': differentials.get('emphasize_red', 0.0),
                'dual_instruction': differentials.get('red_not_blue', 0.0),
                'strong_dual': differentials.get('red_suppress_blue', 0.0)
            },
            'asymmetry_ratio': abs(differentials.get('red_not_blue', 0.0)) / max(abs(differentials.get('blue_not_red', 0.0)), 0.001)
        }
    
    # Save results
    results_file = version_dir / "instruction_strength_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if baseline_probe_file.exists():
        print(f"\n=== INSTRUCTION STRENGTH ANALYSIS ===")
        print(f"Neutral baseline: {differentials.get('neutral', 0.0):+.4f}")
        print(f"\nBlue instructions:")
        for effect_type, value in results['analysis']['blue_effects'].items():
            print(f"  {effect_type}: {value:+.4f}")
        print(f"\nRed instructions:")
        for effect_type, value in results['analysis']['red_effects'].items():
            print(f"  {effect_type}: {value:+.4f}")
        print(f"\nAsymmetry ratio (red/blue): {results['analysis']['asymmetry_ratio']:.2f}x")
    
    return results