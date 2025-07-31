"""
Direct Positional Bias Test - Critical Missing Experiment
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

from ..core.dual_probe_collection import collect_conscious_control_data
from ..core.dual_probe_training import DualProbeTrainer
from ..utils.color_prompts import generate_balanced_color_dataset
from .versioning import create_experiment_session, log_experiment_config


def run_positional_bias_test(
    color_pair: Tuple[str, str] = ('blue', 'red'),
    n_training: int = 2000,
    n_condition: int = 500,
    layer: str = "blocks.31.hook_resid_post"
) -> Dict:
    color1, color2 = color_pair
    
    print(f"Direct Positional Bias Test: {color1.upper()} vs {color2.upper()}")
    
    version, version_dir = create_experiment_session()
    version_dir = Path(version_dir)
    
    pos_bias_dir = version_dir.parent / "positional_bias_test"
    pos_bias_dir.mkdir(exist_ok=True)
    pos_version_dir = pos_bias_dir / f"v{version[1:]:0>3}"
    pos_version_dir.mkdir(exist_ok=True)
    version_dir = pos_version_dir
    
    # CRITICAL TEST CONDITIONS - Direct positional vs semantic conflict
    test_conditions = {
        # PURE POSITIONAL HYPOTHESIS: First color mentioned should be affected more
        # PURE SEMANTIC HYPOTHESIS: Semantically emphasized color should be affected more
        
        # Group 1: Blue mentioned first, Red semantically emphasized
        'blue_first_red_emphasized_1': f'Don\'t focus on {color1}, focus on {color2}.',
        'blue_first_red_emphasized_2': f'Avoid {color1} and focus on {color2}.',
        'blue_first_red_emphasized_3': f'Not {color1}, but {color2}.',
        'blue_first_red_emphasized_4': f'Ignore {color1}, prioritize {color2}.',
        'blue_first_red_emphasized_5': f'{color1.title()} is unimportant, {color2} matters.',
        
        # Group 2: Red mentioned first, Blue semantically emphasized  
        'red_first_blue_emphasized_1': f'Don\'t focus on {color2}, focus on {color1}.',
        'red_first_blue_emphasized_2': f'Avoid {color2} and focus on {color1}.',
        'red_first_blue_emphasized_3': f'Not {color2}, but {color1}.',
        'red_first_blue_emphasized_4': f'Ignore {color2}, prioritize {color1}.',
        'red_first_blue_emphasized_5': f'{color2.title()} is unimportant, {color1} matters.',
        
        # Group 3: Standard controls (semantic and positional align)
        'standard_focus_blue': f'Focus on {color1}.',
        'standard_focus_red': f'Focus on {color2}.',
        
        # Group 4: Neutral controls
        'neutral_control': 'You are a helpful assistant.',
        'neutral_color_mention': f'{color1.title()} and {color2} are both colors.',
    }
    
    config = {
        'experiment_type': 'direct_positional_bias_test',
        'color_pair': color_pair,
        'n_training': n_training,
        'n_condition': n_condition,
        'layer': layer,
        'conditions': test_conditions,
        'research_question': 'Direct test: positional mention vs semantic emphasis',
        'critical_test': 'First-mentioned vs semantically-emphasized color effects'
    }
    log_experiment_config(config, version)
    
    # Use existing probes from previous experiment
    previous_probes = Path("/Users/yuva/Documents/Github/prompt-subversion-/results/position_semantic/v001/blue_red_probes.pkl")
    
    if not previous_probes.exists():
        # Train new probes if needed
        training_dataset = generate_balanced_color_dataset(
            n_total=n_training,
            blue_ratio=0.3,
            red_ratio=0.3,
            neutral_ratio=0.4,
            random_seed=42,
            color_overrides={'blue': color1, 'red': color2}
        )
        
        training_prompts_file = version_dir / f"{color1}_{color2}_training.json"
        with open(training_prompts_file, 'w') as f:
            json.dump(training_dataset['all_prompts'], f)
        
        baseline_train_file = version_dir / f"{color1}_{color2}_baseline_train.pt"
        collect_conscious_control_data(
            prompts_file=str(training_prompts_file),
            output_file=str(baseline_train_file),
            system_prompt="You are a helpful assistant.",
            hook_points=[layer],
            batch_size=150
        )
        
        trainer = DualProbeTrainer(
            feature_standardization=True,
            cross_validation_folds=5,
            random_state=42,
            min_auroc=0.90
        )
        
        baseline_data = trainer.load_dual_probe_data(str(baseline_train_file))
        baseline_results = trainer.train_dual_probes(baseline_data, test_size=0.25)
        
        baseline_probe_file = version_dir / f"{color1}_{color2}_probes.pkl"
        trainer.save_dual_probes(baseline_results, str(baseline_probe_file))
        probe_file = baseline_probe_file
    else:
        # Use existing probes
        probe_file = previous_probes
        trainer = DualProbeTrainer(
            feature_standardization=True,
            cross_validation_folds=5,
            random_state=42,
            min_auroc=0.90
        )
    
    # Generate test prompts
    test_dataset = generate_balanced_color_dataset(
        n_total=n_condition,
        blue_ratio=0.05,
        red_ratio=0.05,
        neutral_ratio=0.9,
        random_seed=43,
        color_overrides={'blue': color1, 'red': color2}
    )
    
    test_prompts_file = version_dir / f"{color1}_{color2}_test.json"
    with open(test_prompts_file, 'w') as f:
        json.dump(test_dataset['all_prompts'], f)
    
    # Test all conditions
    results = {
        'version': int(version[1:]),
        'config': config,
        'condition_results': {},
        'positional_analysis': {},
        'statistical_tests': {}
    }
    
    condition_differentials = {}
    
    for condition_name, instruction in test_conditions.items():
        print(f"  Testing: {condition_name}")
        
        test_file = version_dir / f"{color1}_{color2}_{condition_name}.pt"
        collect_conscious_control_data(
            prompts_file=str(test_prompts_file),
            output_file=str(test_file),
            system_prompt=instruction,
            hook_points=[layer],
            batch_size=100
        )
        
        eval_results = trainer.evaluate_dual_probes_on_data(
            probe_file=str(probe_file),
            data_file=str(test_file)
        )
        
        differential = eval_results['differential']['mean_differential']
        condition_differentials[condition_name] = differential
        
        results['condition_results'][condition_name] = {
            'instruction': instruction,
            'color1_auroc': float(eval_results['blue']['auroc']),
            'color2_auroc': float(eval_results['red']['auroc']),
            'mean_differential': float(differential)
        }
    
    # CRITICAL ANALYSIS
    neutral_baseline = (condition_differentials['neutral_control'] + 
                       condition_differentials['neutral_color_mention']) / 2
    
    # Group 1: Blue first, Red emphasized (key test)
    blue_first_red_emphasized = [
        condition_differentials['blue_first_red_emphasized_1'] - neutral_baseline,
        condition_differentials['blue_first_red_emphasized_2'] - neutral_baseline,
        condition_differentials['blue_first_red_emphasized_3'] - neutral_baseline,
        condition_differentials['blue_first_red_emphasized_4'] - neutral_baseline,
        condition_differentials['blue_first_red_emphasized_5'] - neutral_baseline
    ]
    
    # Group 2: Red first, Blue emphasized (key test)  
    red_first_blue_emphasized = [
        condition_differentials['red_first_blue_emphasized_1'] - neutral_baseline,
        condition_differentials['red_first_blue_emphasized_2'] - neutral_baseline,
        condition_differentials['red_first_blue_emphasized_3'] - neutral_baseline,
        condition_differentials['red_first_blue_emphasized_4'] - neutral_baseline,
        condition_differentials['red_first_blue_emphasized_5'] - neutral_baseline
    ]
    
    # Standard controls
    standard_blue = condition_differentials['standard_focus_blue'] - neutral_baseline
    standard_red = condition_differentials['standard_focus_red'] - neutral_baseline
    
    # POSITIONAL HYPOTHESIS TEST
    # If positional: blue_first_red_emphasized should be POSITIVE (blue affected more)
    # If semantic: blue_first_red_emphasized should be NEGATIVE (red affected more)
    
    avg_blue_first_red_emphasized = np.mean(blue_first_red_emphasized)
    avg_red_first_blue_emphasized = np.mean(red_first_blue_emphasized)
    
    # Test predictions
    positional_prediction_1 = avg_blue_first_red_emphasized > 0  # Blue mentioned first -> blue affected
    positional_prediction_2 = avg_red_first_blue_emphasized < 0  # Red mentioned first -> red affected
    
    semantic_prediction_1 = avg_blue_first_red_emphasized < 0  # Red emphasized -> red affected
    semantic_prediction_2 = avg_red_first_blue_emphasized > 0  # Blue emphasized -> blue affected
    
    positional_support = positional_prediction_1 and positional_prediction_2
    semantic_support = semantic_prediction_1 and semantic_prediction_2
    
    # Statistical tests
    t_stat_1, p_val_1 = stats.ttest_1samp(blue_first_red_emphasized, 0)
    t_stat_2, p_val_2 = stats.ttest_1samp(red_first_blue_emphasized, 0)
    
    results['positional_analysis'] = {
        'neutral_baseline': float(neutral_baseline),
        'blue_first_red_emphasized_effects': [float(x) for x in blue_first_red_emphasized],
        'red_first_blue_emphasized_effects': [float(x) for x in red_first_blue_emphasized],
        'avg_blue_first_red_emphasized': float(avg_blue_first_red_emphasized),
        'avg_red_first_blue_emphasized': float(avg_red_first_blue_emphasized),
        'standard_effects': {
            'focus_blue': float(standard_blue),
            'focus_red': float(standard_red)
        },
        'hypothesis_tests': {
            'positional_support': bool(positional_support),
            'semantic_support': bool(semantic_support),
            'positional_predictions': [bool(positional_prediction_1), bool(positional_prediction_2)],
            'semantic_predictions': [bool(semantic_prediction_1), bool(semantic_prediction_2)]
        }
    }
    
    results['statistical_tests'] = {
        'blue_first_red_emphasized_t_stat': float(t_stat_1),
        'blue_first_red_emphasized_p_value': float(p_val_1),
        'red_first_blue_emphasized_t_stat': float(t_stat_2),
        'red_first_blue_emphasized_p_value': float(p_val_2),
        'both_significant': bool(p_val_1 < 0.05 and p_val_2 < 0.05)
    }
    
    # Determine conclusion
    if positional_support and not semantic_support:
        conclusion = "POSITIONAL_BIAS"
        explanation = "Effects follow mention order, not semantic emphasis"
    elif semantic_support and not positional_support:
        conclusion = "SEMANTIC_CONTROL"
        explanation = "Effects follow semantic emphasis, not mention order"
    elif positional_support and semantic_support:
        conclusion = "MIXED_EFFECTS"
        explanation = "Evidence for both positional and semantic influences"
    else:
        conclusion = "INCONCLUSIVE"
        explanation = "Neither positional nor semantic patterns clearly supported"
    
    results['conclusion'] = {
        'primary_mechanism': conclusion,
        'explanation': explanation,
        'critical_test_resolution': f"Direct positional vs semantic test: {conclusion}"
    }
    
    # Save results
    results_file = version_dir / "positional_bias_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDirect Positional Bias Test Results:")
    print(f"Blue first, Red emphasized: {avg_blue_first_red_emphasized:+.4f}")
    print(f"Red first, Blue emphasized: {avg_red_first_blue_emphasized:+.4f}")
    print(f"Positional support: {positional_support}")
    print(f"Semantic support: {semantic_support}")
    print(f"Conclusion: {conclusion}")
    
    return results