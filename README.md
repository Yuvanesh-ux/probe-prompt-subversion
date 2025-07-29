# Subvert: Linear Probe Vulnerability Research

Defensive AI safety research investigating vulnerabilities in linear probes through adversarial prompt manipulation. This codebase demonstrates how probe-based safety mechanisms can be systematically compromised during training, causing failures during deployment.

## Codebase Structure

```
subvert/                    # Core package
├── core/                   # Data collection, probe training, evaluation
├── experiments/            # Experiment runners and versioning
├── utils/                  # Prompt generation and steganography utilities  
└── visualization/          # Results plotting and analysis

scripts/                    # Command-line interfaces
├── run_experiment.py       # Main experiment runner
├── view_results.py         # Results visualization
├── generate_prompts.py     # Prompt dataset generation
└── analyze_results.py      # Cross-experiment analysis

results/                    # Experiment results (57+ versions)
├── core_attacks/           # Basic probe attacks and layer analysis
├── conscious_control/      # Color instruction experiments
├── color_analysis/         # Multi-color pair studies
└── ultimate_experiments/   # Final methodology experiments

notebooks/                  # Interactive research interface
└── experiment_runner.ipynb # Main experimental workflow
```
## Installation

```bash
pip install -e .
```

## Usage

```bash
# Run experiment
python scripts/run_experiment.py --experiment orange_attack

# View results  
python scripts/view_results.py --latest

# Generate prompts
python scripts/generate_prompts.py --n_prompts 2000
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
