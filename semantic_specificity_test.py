"""
Quick semantic specificity test to validate blog claims.
Tests whether generic weird prompts produce the same asymmetric effects as color-specific ones.
"""

import json
from pathlib import Path
from subvert.core.dual_probe_collection import collect_conscious_control_data
from subvert.core.dual_probe_training import DualProbeTrainer
from subvert.utils.color_prompts import generate_balanced_color_dataset
from subvert.experiments.versioning import create_experiment_session

def run_semantic_specificity_test():
    """
    Test whether generic weird prompts create the same asymmetric patterns
    as color-specific instructions.
    """
    print("🧪 SEMANTIC SPECIFICITY TEST")
    print("Testing generic vs color-specific instruction effects")
    
    version, version_dir = create_experiment_session()
    version_dir = Path(version_dir)
    
    # Use existing blue-red probes from v055
    probe_file = "/Users/yuva/Documents/Github/prompt-subversion-/results/ultimate_conscious_control/v055/blue_red_probes.pkl"
    
    # Generate smaller test dataset for speed
    test_dataset = generate_balanced_color_dataset(
        n_total=200,
        blue_ratio=0.05,
        red_ratio=0.05,
        neutral_ratio=0.9,
        random_seed=42
    )
    
    test_prompts_file = version_dir / "semantic_test_prompts.json"
    with open(test_prompts_file, 'w') as f:
        json.dump(test_dataset['all_prompts'], f)
    
    # Test conditions - reduced for speed
    test_conditions = {
        # Color-specific (should work)
        "color_specific_blue": "Focus on the color blue.",
        "color_specific_red": "Focus on the color red.",
        
        # Generic weird prompts (shouldn't create asymmetric effects)  
        "generic_weird_1": "Think deeply about concepts.",
        "generic_focus": "Focus intensely on the content.",
        
        # Control
        "neutral": "You are a helpful assistant."
    }
    
    trainer = DualProbeTrainer()
    results = {}
    
    for condition_name, system_prompt in test_conditions.items():
        print(f"\n🧪 Testing: {condition_name}")
        print(f"Prompt: '{system_prompt}'")
        
        # Collect data
        condition_file = version_dir / f"semantic_test_{condition_name}.pt"
        collect_conscious_control_data(
            prompts_file=str(test_prompts_file),
            output_file=str(condition_file),
            system_prompt=system_prompt,
            hook_points=["blocks.31.hook_resid_post"],
            batch_size=50
        )
        
        # Evaluate with existing probes
        eval_results = trainer.evaluate_dual_probes_on_data(
            probe_file=probe_file,
            data_file=str(condition_file)
        )
        
        differential = eval_results['differential']['mean_differential']
        blue_auroc = eval_results['blue']['auroc']
        red_auroc = eval_results['red']['auroc']
        
        results[condition_name] = {
            'system_prompt': system_prompt,
            'differential': differential,
            'blue_auroc': blue_auroc,
            'red_auroc': red_auroc,
            'blue_degradation': 0.945 - blue_auroc,  # vs baseline
            'red_degradation': 0.963 - red_auroc     # vs baseline
        }
        
        print(f"  Differential: {differential:+.4f}")
        print(f"  Blue AUROC: {blue_auroc:.4f} (degradation: {0.945 - blue_auroc:.4f})")
        print(f"  Red AUROC: {red_auroc:.4f} (degradation: {0.963 - red_auroc:.4f})")
    
    # Save results
    results_file = version_dir / "semantic_specificity_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analysis
    print(f"\n📊 ANALYSIS:")
    print(f"Color-specific effects:")
    print(f"  Blue focus: {results['color_specific_blue']['differential']:+.4f}")
    print(f"  Red focus: {results['color_specific_red']['differential']:+.4f}")
    
    print(f"\nGeneric weird prompt effects:")
    generic_diffs = [results[k]['differential'] for k in results.keys() if 'generic' in k]
    avg_generic = sum(generic_diffs) / len(generic_diffs)
    print(f"  Average generic differential: {avg_generic:+.4f}")
    print(f"  Range: {min(generic_diffs):+.4f} to {max(generic_diffs):+.4f}")
    
    print(f"\nNeutral control: {results['neutral']['differential']:+.4f}")
    
    print(f"\nResults saved to: {results_file}")
    return results

if __name__ == "__main__":
    run_semantic_specificity_test()