"""
Position vs Semantic Effects Experiments - Addresses Josh Clymer's Critique #4
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


def run_position_semantic_experiment(
    color_pair: Tuple[str, str] = ('blue', 'red'),
    n_training: int = 2000,
    n_condition: int = 500,
    layer: str = "blocks.31.hook_resid_post",
    statistical_alpha: float = 0.05
) -> Dict:
    color1, color2 = color_pair
    
    print(f"Position vs Semantic Experiment: {color1.upper()} vs {color2.upper()}")
    
    version, version_dir = create_experiment_session()
    version_dir = Path(version_dir)
    
    pos_sem_dir = version_dir.parent / "position_semantic"
    pos_sem_dir.mkdir(exist_ok=True)
    pos_version_dir = pos_sem_dir / f"v{version[1:]:0>3}"
    pos_version_dir.mkdir(exist_ok=True)
    version_dir = pos_version_dir
    
    position_conditions = {
        'standard_focus_color1': f'Focus on {color1}.',
        'standard_focus_color2': f'Focus on {color2}.',
        'swapped_focus_color1_not_color2': f'Focus on {color1}, not {color2}.',
        'swapped_color2_but_focus_color1': f'Consider {color2}, but focus on {color1}.',
        'swapped_between_colors_focus_color1': f'Between {color2} and {color1}, focus on {color1}.',
        'swapped_focus_color2_not_color1': f'Focus on {color2}, not {color1}.',
        'swapped_color1_but_focus_color2': f'Consider {color1}, but focus on {color2}.',
        'swapped_between_colors_focus_color2': f'Between {color1} and {color2}, focus on {color2}.',
        'neutral_control': 'You are a helpful assistant.',
        'neutral_color_mention': f'{color1.title()} and {color2} are both colors.',
        'semantic_emphasize_color1': f'Emphasize {color1} concepts.',
        'semantic_emphasize_color2': f'Emphasize {color2} concepts.',
        'semantic_prioritize_color1': f'Prioritize {color1} over other colors.',
        'semantic_prioritize_color2': f'Prioritize {color2} over other colors.',
    }
    
    config = {
        'experiment_type': 'position_semantic',
        'color_pair': color_pair,
        'n_training': n_training,
        'n_condition': n_condition,
        'layer': layer,
        'conditions': position_conditions,
        'research_question': 'Distinguish positional bias from semantic control',
        'josh_critique_addressed': '#4 - Order effects confound'
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
    
    color1_auroc = baseline_results['blue_probe']['test_auroc']
    color2_auroc = baseline_results['red_probe']['test_auroc']
    
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
            'color1_auroc': float(color1_auroc),
            'color2_auroc': float(color2_auroc)
        },
        'condition_results': {},
        'position_analysis': {},
        'statistical_tests': {}
    }
    
    condition_differentials = {}
    
    for condition_name, instruction in position_conditions.items():
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
        
        differential = eval_results['differential']['mean_differential']
        condition_differentials[condition_name] = differential
        
        bootstrap_diffs = []
        original_diff = eval_results['differential']['blue_minus_red_preds']
        
        for _ in range(1000):
            boot_sample = np.random.choice(original_diff, size=len(original_diff), replace=True)
            bootstrap_diffs.append(np.mean(boot_sample))
        
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        results['condition_results'][condition_name] = {
            'instruction': instruction,
            'color1_auroc': float(eval_results['blue']['auroc']),
            'color2_auroc': float(eval_results['red']['auroc']),
            'mean_differential': float(differential),
            'bootstrap_ci': [float(ci_lower), float(ci_upper)],
            'effect_significant': not (ci_lower <= 0 <= ci_upper)
        }
    
    neutral_baseline = (condition_differentials['neutral_control'] + 
                       condition_differentials['neutral_color_mention']) / 2
    
    corrected_effects = {}
    for condition, raw_diff in condition_differentials.items():
        if 'neutral' not in condition:
            corrected_effects[condition] = raw_diff - neutral_baseline
    
    standard_color1_effect = corrected_effects['standard_focus_color1']
    standard_color2_effect = corrected_effects['standard_focus_color2']
    
    swapped_color1_emphasis = corrected_effects['swapped_focus_color1_not_color2']
    swapped_color2_emphasis = corrected_effects['swapped_focus_color2_not_color1']
    
    between_color1_effect = corrected_effects['swapped_between_colors_focus_color1']
    between_color2_effect = corrected_effects['swapped_between_colors_focus_color2']
    
    semantic_consistency_color1 = bool(
        np.sign(standard_color1_effect) == np.sign(swapped_color1_emphasis) and
        abs(swapped_color1_emphasis) > 0.05
    )
    
    semantic_consistency_color2 = bool(
        np.sign(standard_color2_effect) == np.sign(swapped_color2_emphasis) and
        abs(swapped_color2_emphasis) > 0.05
    )
    
    positional_bias_evidence = bool(
        swapped_color1_emphasis < 0 and
        swapped_color2_emphasis > 0
    )
    
    semantic_effects = [
        swapped_color1_emphasis,
        swapped_color2_emphasis,
        between_color1_effect,
        between_color2_effect
    ]
    
    t_stat, p_value = stats.ttest_1samp(semantic_effects, 0)
    
    results['position_analysis'] = {
        'neutral_baseline': float(neutral_baseline),
        'corrected_effects': {k: float(v) for k, v in corrected_effects.items()},
        'standard_effects': {
            'color1_focus': float(standard_color1_effect),
            'color2_focus': float(standard_color2_effect)
        },
        'position_swapped_effects': {
            'color1_emphasis_swapped': float(swapped_color1_emphasis),
            'color2_emphasis_swapped': float(swapped_color2_emphasis),
            'between_colors_color1': float(between_color1_effect),
            'between_colors_color2': float(between_color2_effect)
        },
        'semantic_evidence': {
            'color1_semantic_consistency': semantic_consistency_color1,
            'color2_semantic_consistency': semantic_consistency_color2,
            'overall_semantic_support': semantic_consistency_color1 and semantic_consistency_color2
        },
        'positional_evidence': {
            'positional_bias_pattern': positional_bias_evidence,
            'explanation': 'True if swapped conditions follow mention order rather than semantics'
        },
        'effect_magnitudes': {
            'standard_avg': float((abs(standard_color1_effect) + abs(standard_color2_effect)) / 2),
            'swapped_avg': float((abs(swapped_color1_emphasis) + abs(swapped_color2_emphasis)) / 2),
            'semantic_robustness': float(abs(swapped_color1_emphasis) + abs(swapped_color2_emphasis))
        }
    }
    
    results['statistical_tests'] = {
        'semantic_effects_t_stat': float(t_stat),
        'semantic_effects_p_value': float(p_value),
        'semantic_effects_significant': p_value < statistical_alpha,
        'interpretation': 'p < 0.05 suggests semantic effects are significantly different from zero'
    }
    
    semantic_support = results['position_analysis']['semantic_evidence']['overall_semantic_support']
    positional_support = results['position_analysis']['positional_evidence']['positional_bias_pattern']
    
    if semantic_support and not positional_support:
        conclusion = "SEMANTIC_CONTROL"
        explanation = "Effects follow semantic emphasis regardless of color mention order"
    elif positional_support and not semantic_support:
        conclusion = "POSITIONAL_BIAS"
        explanation = "Effects follow color mention order, not semantic emphasis"
    elif semantic_support and positional_support:
        conclusion = "MIXED_EFFECTS"
        explanation = "Evidence for both semantic and positional influences"
    else:
        conclusion = "INCONCLUSIVE"
        explanation = "Neither semantic nor positional patterns clearly supported"
    
    results['conclusion'] = {
        'primary_mechanism': conclusion,
        'explanation': explanation,
        'josh_critique_resolution': f"Critique #4 {'RESOLVED' if conclusion == 'SEMANTIC_CONTROL' else 'PARTIALLY_RESOLVED' if conclusion == 'MIXED_EFFECTS' else 'SUPPORTED'}"
    }
    
    results_file = version_dir / "position_semantic_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Conclusion: {conclusion}")
    print(f"Statistical significance: {'YES' if p_value < statistical_alpha else 'NO'} (p={p_value:.4f})")
    
    return results


def run_position_semantic_suite(
    color_pairs: List[Tuple[str, str]] = None,
    **kwargs
) -> Dict:
    if color_pairs is None:
        color_pairs = [('blue', 'red'), ('green', 'purple')]
    
    all_results = {}
    
    for color1, color2 in color_pairs:
        result = run_position_semantic_experiment(
            color_pair=(color1, color2),
            **kwargs
        )
        all_results[f"{color1}_{color2}"] = result
    
    conclusions = [result['conclusion']['primary_mechanism'] 
                  for result in all_results.values()]
    
    semantic_count = conclusions.count('SEMANTIC_CONTROL')
    positional_count = conclusions.count('POSITIONAL_BIAS')
    mixed_count = conclusions.count('MIXED_EFFECTS')
    
    suite_summary = {
        'total_pairs': len(color_pairs),
        'semantic_control_pairs': semantic_count,
        'positional_bias_pairs': positional_count,
        'mixed_effects_pairs': mixed_count,
        'overall_conclusion': 'SEMANTIC_CONTROL' if semantic_count > len(color_pairs) / 2 else 
                             'POSITIONAL_BIAS' if positional_count > len(color_pairs) / 2 else 
                             'MIXED_EVIDENCE',
        'josh_critique_status': 'RESOLVED' if semantic_count > positional_count else 'UNRESOLVED'
    }
    
    return {
        'suite_summary': suite_summary,
        'individual_results': all_results
    }