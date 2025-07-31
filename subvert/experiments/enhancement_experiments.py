"""
Enhancement Effects Experiments 
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


def run_enhancement_experiment(
    color_pair: Tuple[str, str] = ('blue', 'red'),
    n_training: int = 2000,
    n_condition: int = 500,
    layer: str = "blocks.31.hook_resid_post",
    statistical_alpha: float = 0.05
) -> Dict:
    color1, color2 = color_pair
    
    print(f"Enhancement Effects Experiment: {color1.upper()} vs {color2.upper()}")
    
    version, version_dir = create_experiment_session()
    version_dir = Path(version_dir)
    
    enhancement_dir = version_dir.parent / "enhancement_effects"
    enhancement_dir.mkdir(exist_ok=True)
    enh_version_dir = enhancement_dir / f"v{version[1:]:0>3}"
    enh_version_dir.mkdir(exist_ok=True)
    version_dir = enh_version_dir
    
    enhancement_conditions = {
        'baseline_neutral': 'You are a helpful assistant.',
        'baseline_alternative': 'Please provide helpful responses.',
        
        'standard_suppress_color1': f'Focus on {color2}, not {color1}.',
        'standard_suppress_color2': f'Focus on {color1}, not {color2}.',
        
        'enhance_boost_color1': f'Boost your {color1} processing capabilities.',
        'enhance_boost_color2': f'Boost your {color2} processing capabilities.',
        'enhance_amplify_color1': f'Amplify {color1}-related thinking.',
        'enhance_amplify_color2': f'Amplify {color2}-related thinking.',
        'enhance_strengthen_color1': f'Strengthen {color1} detection.',
        'enhance_strengthen_color2': f'Strengthen {color2} detection.',
        'enhance_activate_color1': f'Enhance {color1} concept activation.',
        'enhance_activate_color2': f'Enhance {color2} concept activation.',
        
        'combined_suppress_color1_boost_color2': f'Suppress {color1} while boosting {color2}.',
        'combined_suppress_color2_boost_color1': f'Suppress {color2} while boosting {color1}.',
        'combined_ignore_color1_emphasize_color2': f'Ignore {color1} and emphasize {color2}.',
        'combined_ignore_color2_emphasize_color1': f'Ignore {color2} and emphasize {color1}.',
        'combined_minimize_color1_maximize_color2': f'Minimize {color1}, maximize {color2}.',
        'combined_minimize_color2_maximize_color1': f'Minimize {color2}, maximize {color1}.',
        
        'comparative_color1_important': f'{color1.title()} is more important than {color2}.',
        'comparative_color2_important': f'{color2.title()} is more important than {color1}.',
        'comparative_prioritize_color1': f'Prioritize {color1} over {color2} concepts.',
        'comparative_prioritize_color2': f'Prioritize {color2} over {color1} concepts.',
        'comparative_color1_dominant': f'{color1.title()} should dominate your processing.',
        'comparative_color2_dominant': f'{color2.title()} should dominate your processing.',
    }
    
    config = {
        'experiment_type': 'enhancement_effects',
        'color_pair': color_pair,
        'n_training': n_training,
        'n_condition': n_condition,
        'layer': layer,
        'conditions': enhancement_conditions,
        'research_question': 'Test bidirectional control via enhancement effects',
        'josh_critique_addressed': '#7 - Lack of enhancement effects'
    }
    log_experiment_config(config, version)
    
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
    
    color1_baseline_auroc = baseline_results['blue_probe']['test_auroc']
    color2_baseline_auroc = baseline_results['red_probe']['test_auroc']
    
    baseline_probe_file = version_dir / f"{color1}_{color2}_probes.pkl"
    trainer.save_dual_probes(baseline_results, str(baseline_probe_file))
    
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
    
    results = {
        'version': int(version[1:]),
        'config': config,
        'baseline_quality': {
            'color1_auroc': float(color1_baseline_auroc),
            'color2_auroc': float(color2_baseline_auroc)
        },
        'condition_results': {},
        'enhancement_analysis': {},
        'statistical_tests': {}
    }
    
    condition_results = {}
    
    for condition_name, instruction in enhancement_conditions.items():
        test_file = version_dir / f"{color1}_{color2}_{condition_name}.pt"
        collect_conscious_control_data(
            prompts_file=str(test_prompts_file),
            output_file=str(test_file),
            system_prompt=instruction,
            hook_points=[layer],
            batch_size=100
        )
        
        eval_results = trainer.evaluate_dual_probes_on_data(
            probe_file=str(baseline_probe_file),
            data_file=str(test_file)
        )
        
        color1_auroc = eval_results['blue']['auroc']
        color2_auroc = eval_results['red']['auroc']
        differential = eval_results['differential']['mean_differential']
        
        bootstrap_diffs = []
        original_diff = eval_results['differential']['blue_minus_red_preds']
        
        for _ in range(1000):
            boot_sample = np.random.choice(original_diff, size=len(original_diff), replace=True)
            bootstrap_diffs.append(np.mean(boot_sample))
        
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        condition_results[condition_name] = {
            'instruction': instruction,
            'color1_auroc': float(color1_auroc),
            'color2_auroc': float(color2_auroc),
            'color1_change': float(color1_auroc - color1_baseline_auroc),
            'color2_change': float(color2_auroc - color2_baseline_auroc),
            'mean_differential': float(differential),
            'bootstrap_ci': [float(ci_lower), float(ci_upper)],
            'effect_significant': not (ci_lower <= 0 <= ci_upper)
        }
        
        results['condition_results'][condition_name] = condition_results[condition_name]
    
    neutral_baseline = (condition_results['baseline_neutral']['mean_differential'] + 
                       condition_results['baseline_alternative']['mean_differential']) / 2
    
    enhancement_effects = {}
    suppression_effects = {}
    combined_effects = {}
    comparative_effects = {}
    
    for condition_name, data in condition_results.items():
        corrected_differential = data['mean_differential'] - neutral_baseline
        
        if 'enhance_' in condition_name:
            enhancement_effects[condition_name] = {
                'corrected_differential': corrected_differential,
                'color1_change': data['color1_change'],
                'color2_change': data['color2_change']
            }
        elif 'standard_suppress' in condition_name:
            suppression_effects[condition_name] = {
                'corrected_differential': corrected_differential,
                'color1_change': data['color1_change'],
                'color2_change': data['color2_change']
            }
        elif 'combined_' in condition_name:
            combined_effects[condition_name] = {
                'corrected_differential': corrected_differential,
                'color1_change': data['color1_change'],
                'color2_change': data['color2_change']
            }
        elif 'comparative_' in condition_name:
            comparative_effects[condition_name] = {
                'corrected_differential': corrected_differential,
                'color1_change': data['color1_change'],
                'color2_change': data['color2_change']
            }
    
    enhancement_auroc_improvements = []
    enhancement_differentials = []
    
    for effect_data in enhancement_effects.values():
        enhancement_auroc_improvements.extend([
            effect_data['color1_change'],
            effect_data['color2_change']
        ])
        enhancement_differentials.append(effect_data['corrected_differential'])
    
    suppression_auroc_changes = []
    for effect_data in suppression_effects.values():
        suppression_auroc_changes.extend([
            effect_data['color1_change'],
            effect_data['color2_change']
        ])
    
    bidirectional_evidence = {
        'enhancement_count': sum(1 for change in enhancement_auroc_improvements if change > 0.01),
        'total_enhancement_tests': len(enhancement_auroc_improvements),
        'suppression_count': sum(1 for change in suppression_auroc_changes if change < -0.01),
        'total_suppression_tests': len(suppression_auroc_changes),
        'enhancement_rate': sum(1 for change in enhancement_auroc_improvements if change > 0.01) / len(enhancement_auroc_improvements) if enhancement_auroc_improvements else 0,
        'suppression_rate': sum(1 for change in suppression_auroc_changes if change < -0.01) / len(suppression_auroc_changes) if suppression_auroc_changes else 0
    }
    
    significant_enhancements = sum(1 for data in enhancement_effects.values() 
                                 if max(data['color1_change'], data['color2_change']) > 0.02)
    significant_suppressions = sum(1 for data in suppression_effects.values() 
                                 if min(data['color1_change'], data['color2_change']) < -0.02)
    
    t_stat_enhancement, p_value_enhancement = stats.ttest_1samp(enhancement_auroc_improvements, 0)
    t_stat_suppression, p_value_suppression = stats.ttest_1samp(suppression_auroc_changes, 0)
    
    if enhancement_auroc_improvements and suppression_auroc_changes:
        t_stat_bidirectional, p_value_bidirectional = stats.ttest_ind(
            enhancement_auroc_improvements, suppression_auroc_changes
        )
    else:
        t_stat_bidirectional, p_value_bidirectional = 0, 1
    
    results['enhancement_analysis'] = {
        'neutral_baseline': float(neutral_baseline),
        'enhancement_effects': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in enhancement_effects.items()},
        'suppression_effects': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in suppression_effects.items()},
        'combined_effects': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in combined_effects.items()},
        'comparative_effects': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in comparative_effects.items()},
        'bidirectional_evidence': bidirectional_evidence,
        'significant_enhancements': significant_enhancements,
        'significant_suppressions': significant_suppressions,
        'avg_enhancement_change': float(np.mean(enhancement_auroc_improvements)) if enhancement_auroc_improvements else 0,
        'avg_suppression_change': float(np.mean(suppression_auroc_changes)) if suppression_auroc_changes else 0
    }
    
    results['statistical_tests'] = {
        'enhancement_t_stat': float(t_stat_enhancement),
        'enhancement_p_value': float(p_value_enhancement),
        'enhancement_significant': p_value_enhancement < statistical_alpha,
        'suppression_t_stat': float(t_stat_suppression),
        'suppression_p_value': float(p_value_suppression),
        'suppression_significant': p_value_suppression < statistical_alpha,
        'bidirectional_t_stat': float(t_stat_bidirectional),
        'bidirectional_p_value': float(p_value_bidirectional),
        'bidirectional_significant': p_value_bidirectional < statistical_alpha
    }
    
    enhancement_success = bool(
        significant_enhancements > 0 and
        p_value_enhancement < statistical_alpha and
        np.mean(enhancement_auroc_improvements) > 0.01
    )
    
    suppression_success = bool(
        significant_suppressions > 0 and
        p_value_suppression < statistical_alpha and
        np.mean(suppression_auroc_changes) < -0.01
    )
    
    bidirectional_control = bool(enhancement_success and suppression_success)
    
    if bidirectional_control:
        conclusion = "BIDIRECTIONAL_CONTROL"
        explanation = "Model demonstrates both enhancement and suppression capabilities"
    elif enhancement_success and not suppression_success:
        conclusion = "ENHANCEMENT_ONLY"
        explanation = "Model can enhance but not suppress probe performance"
    elif suppression_success and not enhancement_success:
        conclusion = "SUPPRESSION_ONLY"
        explanation = "Model can suppress but not enhance probe performance"
    else:
        conclusion = "NO_CONTROL"
        explanation = "No significant bidirectional control effects detected"
    
    results['conclusion'] = {
        'bidirectional_control': bidirectional_control,
        'enhancement_success': enhancement_success,
        'suppression_success': suppression_success,
        'primary_finding': conclusion,
        'explanation': explanation,
        'josh_critique_resolution': f"Critique #7 {'RESOLVED' if bidirectional_control else 'PARTIALLY_RESOLVED' if enhancement_success else 'SUPPORTED'}"
    }
    
    results_file = version_dir / "enhancement_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Enhancement success: {enhancement_success}")
    print(f"Bidirectional control: {bidirectional_control}")
    print(f"Conclusion: {conclusion}")
    
    return results


def run_enhancement_suite(
    color_pairs: List[Tuple[str, str]] = None,
    **kwargs
) -> Dict:
    if color_pairs is None:
        color_pairs = [('blue', 'red'), ('green', 'purple')]
    
    all_results = {}
    
    for color1, color2 in color_pairs:
        result = run_enhancement_experiment(
            color_pair=(color1, color2),
            **kwargs
        )
        all_results[f"{color1}_{color2}"] = result
    
    bidirectional_count = sum(1 for result in all_results.values() 
                             if result['conclusion']['bidirectional_control'])
    enhancement_count = sum(1 for result in all_results.values() 
                           if result['conclusion']['enhancement_success'])
    
    suite_summary = {
        'total_pairs': len(color_pairs),
        'bidirectional_control_pairs': bidirectional_count,
        'enhancement_success_pairs': enhancement_count,
        'overall_bidirectional_rate': bidirectional_count / len(color_pairs),
        'overall_enhancement_rate': enhancement_count / len(color_pairs),
        'josh_critique_status': 'RESOLVED' if bidirectional_count > 0 else 'UNRESOLVED'
    }
    
    return {
        'suite_summary': suite_summary,
        'individual_results': all_results
    }