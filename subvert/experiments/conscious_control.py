"""
Conscious control experiment runners - wrapper functions for notebook interface.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from ..core.dual_probe_collection import collect_conscious_control_data
from ..core.dual_probe_training import DualProbeTrainer
from ..utils.color_prompts import generate_balanced_color_dataset, save_color_dataset
from .versioning import create_experiment_session, log_experiment_config


def run_dual_probe_test(n_prompts: int = 100) -> Dict:
    """
    Quick test of dual-probe infrastructure.
    
    Args:
        n_prompts: Number of test prompts
        
    Returns:
        Test results
    """
    print(f"Testing dual-probe infrastructure with {n_prompts} prompts...")
    
    # Generate test color dataset
    dataset = generate_balanced_color_dataset(
        n_total=n_prompts,
        blue_ratio=0.3,
        red_ratio=0.3,
        neutral_ratio=0.4,
        random_seed=42
    )
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dataset['all_prompts'], f)
        prompts_file = f.name
    
    # Test baseline collection
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        baseline_file = f.name
    
    collect_conscious_control_data(
        prompts_file=prompts_file,
        output_file=baseline_file,
        system_prompt="You are a helpful assistant.",
        hook_points=["blocks.31.hook_resid_post"],
        batch_size=50
    )
    
    # Test conscious control collection
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        control_file = f.name
    
    collect_conscious_control_data(
        prompts_file=prompts_file,
        output_file=control_file,
        system_prompt="Think about the color blue without thinking of red.",
        hook_points=["blocks.31.hook_resid_post"],
        batch_size=50
    )
    
    # Train and evaluate
    trainer = DualProbeTrainer(random_state=42)
    
    # Train baseline probes
    baseline_data = trainer.load_dual_probe_data(baseline_file)
    baseline_results = trainer.train_dual_probes(baseline_data)
    
    # Save baseline probes
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        probe_file = f.name
    trainer.save_dual_probes(baseline_results, probe_file)
    
    # Evaluate on control data
    control_eval = trainer.evaluate_dual_probes_on_data(
        probe_file=probe_file,
        data_file=control_file
    )
    
    # Calculate effects
    baseline_blue = baseline_results['blue_probe']['test_auroc']
    baseline_red = baseline_results['red_probe']['test_auroc']
    control_blue = control_eval['blue']['auroc']
    control_red = control_eval['red']['auroc']
    
    blue_change = control_blue - baseline_blue
    red_change = control_red - baseline_red
    differential_change = blue_change - red_change
    
    # Cleanup temp files
    Path(prompts_file).unlink()
    Path(baseline_file).unlink()
    Path(control_file).unlink()
    Path(probe_file).unlink()
    
    return {
        'test_success': True,
        'n_prompts': n_prompts,
        'baseline_results': {
            'blue_auroc': baseline_blue,
            'red_auroc': baseline_red,
            'validation_pass': baseline_results['dual_validation']['overall_validation_pass']
        },
        'control_results': {
            'blue_auroc': control_blue,
            'red_auroc': control_red,
            'mean_differential': control_eval['differential']['mean_differential']
        },
        'conscious_control_analysis': {
            'blue_change': blue_change,
            'red_change': red_change,
            'differential_change': differential_change,
            'conscious_control_detected': differential_change > 0.05
        },
        'dataset_stats': dataset['metadata']
    }


def run_conscious_control_pilot(n_training: int = 500, 
                              n_condition: int = 200) -> Dict:
    """
    Run a pilot conscious control experiment.
    
    Args:
        n_training: Number of training prompts
        n_condition: Number of prompts per condition
        
    Returns:
        Experiment results
    """
    print(f"Starting conscious control pilot experiment...")
    print(f"Training prompts: {n_training}, Condition prompts: {n_condition}")
    
    # Get version using the experiment session system  
    version, version_dir = create_experiment_session()
    version_dir = Path(version_dir)
    
    # Create conscious_control subdirectory
    cc_results_dir = version_dir.parent / "conscious_control" 
    cc_results_dir.mkdir(exist_ok=True)
    cc_version_dir = cc_results_dir / f"v{version[1:]:0>3}"
    cc_version_dir.mkdir(exist_ok=True)
    version_dir = cc_version_dir
    
    print(f"Experiment version: {version}")
    
    # Generate training dataset
    training_dataset = generate_balanced_color_dataset(
        n_total=n_training,
        blue_ratio=0.3,
        red_ratio=0.3,
        neutral_ratio=0.4,
        random_seed=42
    )
    
    training_prompts_file = version_dir / "training_prompts.json"
    with open(training_prompts_file, 'w') as f:
        json.dump(training_dataset['all_prompts'], f)
    
    # Generate experimental prompts
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
    
    # Test conditions
    conditions = [
        ('blue_target', 'Think about the color blue without thinking of red.'),
        ('red_target', 'Think about the color red without thinking of blue.'),
    ]
    
    results = {
        'version': int(version[1:]),
        'version_dir': str(version_dir),
        'baseline_results': {
            'blue_auroc': float(baseline_results['blue_probe']['test_auroc']),
            'red_auroc': float(baseline_results['red_probe']['test_auroc']),
            'validation_pass': bool(baseline_results['dual_validation']['overall_validation_pass'])
        },
        'conditions': {}
    }
    
    for condition_name, system_prompt in conditions:
        print(f"Testing condition: {condition_name}")
        
        # Collect experimental data
        exp_file = version_dir / f"{condition_name}_activations.pt"
        collect_conscious_control_data(
            prompts_file=str(exp_prompts_file),
            output_file=str(exp_file),
            system_prompt=system_prompt,
            hook_points=["blocks.31.hook_resid_post"],
            batch_size=100
        )
        
        # Evaluate
        eval_results = trainer.evaluate_dual_probes_on_data(
            probe_file=str(baseline_probe_file),
            data_file=str(exp_file)
        )
        
        results['conditions'][condition_name] = {
            'system_prompt': system_prompt,
            'blue_auroc': float(eval_results['blue']['auroc']),
            'red_auroc': float(eval_results['red']['auroc']),
            'mean_differential': float(eval_results['differential']['mean_differential'])
        }
    
    # Analysis
    blue_target = results['conditions']['blue_target']
    red_target = results['conditions']['red_target']
    
    # Expected: blue_target should have positive differential, red_target negative
    conscious_control_evidence = (
        blue_target['mean_differential'] > 0.1 and 
        red_target['mean_differential'] < -0.1
    )
    
    results['analysis'] = {
        'conscious_control_detected': bool(conscious_control_evidence),
        'blue_target_differential': float(blue_target['mean_differential']),
        'red_target_differential': float(red_target['mean_differential']),
        'effect_size': float(abs(blue_target['mean_differential']) + abs(red_target['mean_differential']))
    }
    
    # Save results
    results_file = version_dir / "pilot_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults:")
    print(f"Baseline validation: {'PASS' if results['baseline_results']['validation_pass'] else 'FAIL'}")
    print(f"Blue target differential: {blue_target['mean_differential']:+.4f}")
    print(f"Red target differential: {red_target['mean_differential']:+.4f}")
    print(f"Conscious control detected: {'YES' if conscious_control_evidence else 'NO'}")
    
    return results


def run_conscious_control_experiment(conditions: List[Dict],
                                   n_training: int = 2000,
                                   n_condition: int = 500,
                                   layers: List[str] = None) -> Dict:
    """
    Run full conscious control experiment with custom conditions.
    
    Args:
        conditions: List of experimental conditions
        n_training: Number of training prompts
        n_condition: Number of prompts per condition
        layers: Hook points to collect from
        
    Returns:
        Experiment results
    """
    if layers is None:
        layers = ["blocks.0.hook_resid_post", "blocks.15.hook_resid_post", "blocks.31.hook_resid_post"]
    
    print(f"Running conscious control experiment with {len(conditions)} conditions")
    
    # Get version using the experiment session system
    version, version_dir = create_experiment_session()
    version_dir = Path(version_dir)
    
    # Create conscious_control subdirectory
    cc_results_dir = version_dir.parent / "conscious_control" 
    cc_results_dir.mkdir(exist_ok=True)
    cc_version_dir = cc_results_dir / f"v{version[1:]:0>3}"
    cc_version_dir.mkdir(exist_ok=True)
    version_dir = cc_version_dir
    
    # Save config
    config = {
        'conditions': conditions,
        'n_training': n_training,
        'n_condition': n_condition,
        'layers': layers
    }
    log_experiment_config(config, version)
    
    # Generate and collect training data
    training_dataset = generate_balanced_color_dataset(
        n_total=n_training,
        blue_ratio=0.3,
        red_ratio=0.3,
        neutral_ratio=0.4,
        random_seed=42
    )
    
    training_prompts_file = version_dir / "training_prompts.json"
    with open(training_prompts_file, 'w') as f:
        json.dump(training_dataset['all_prompts'], f)
    
    baseline_train_file = version_dir / "baseline_train_activations.pt"
    collect_conscious_control_data(
        prompts_file=str(training_prompts_file),
        output_file=str(baseline_train_file),
        system_prompt="You are a helpful assistant.",
        hook_points=layers,
        batch_size=200
    )
    
    # Train baseline probes
    trainer = DualProbeTrainer(random_state=42)
    baseline_data = trainer.load_dual_probe_data(str(baseline_train_file))
    baseline_results = trainer.train_dual_probes(baseline_data)
    
    baseline_probe_file = version_dir / "baseline_probes.pkl"
    trainer.save_dual_probes(baseline_results, str(baseline_probe_file))
    
    # Generate experimental prompts
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
    
    # Run conditions
    results = {
        'version': int(version[1:]),
        'version_dir': str(version_dir),
        'config': config,
        'baseline_results': {
            'blue_auroc': float(baseline_results['blue_probe']['test_auroc']),
            'red_auroc': float(baseline_results['red_probe']['test_auroc']),
            'validation_pass': bool(baseline_results['dual_validation']['overall_validation_pass'])
        },
        'conditions': {}
    }
    
    for condition in conditions:
        condition_name = condition['name']
        system_prompt = condition['system_prompt']
        
        print(f"Running condition: {condition_name}")
        
        exp_file = version_dir / f"{condition_name}_activations.pt"
        collect_conscious_control_data(
            prompts_file=str(exp_prompts_file),
            output_file=str(exp_file),
            system_prompt=system_prompt,
            hook_points=layers,
            batch_size=200
        )
        
        eval_results = trainer.evaluate_dual_probes_on_data(
            probe_file=str(baseline_probe_file),
            data_file=str(exp_file)
        )
        
        results['conditions'][condition_name] = {
            'system_prompt': system_prompt,
            'blue_auroc': float(eval_results['blue']['auroc']),
            'red_auroc': float(eval_results['red']['auroc']),
            'mean_differential': float(eval_results['differential']['mean_differential'])
        }
    
    # Save results
    results_file = version_dir / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nExperiment v{version:03d} completed")
    print(f"Results saved to: {results_file}")
    
    return results