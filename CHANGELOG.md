# Changelog

All notable changes to the Subvert project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-07-22

### Added
- **Core Package Structure**: Organized codebase with `subvert/` package containing:
  - `core/`: Data collection, probe training, and evaluation
  - `visualization/`: Plotting and results viewing  
  - `experiments/`: Experiment management and versioning
  - `utils/`: Prompt generation and utilities

- **Professional CLI Tools**:
  - `scripts/run_experiment.py`: Main experiment runner with multiple attack types
  - `scripts/generate_prompts.py`: Balanced prompt dataset generation  
  - `scripts/analyze_results.py`: Comprehensive results analysis and comparison
  - `scripts/view_results.py`: Interactive results visualization

- **Research Notebooks**:
  - `experiment_runner.ipynb`: Core experimental workflow for running and tracking experiments
  - `attack_variants.ipynb`: Test different attack methods (ORANGE, BANANA, PURPLE, invisible Unicode)
  - `results_analysis.ipynb`: Statistical analysis, version comparison, and detailed result examination
  - `defense_testing.ipynb`: Test defense mechanisms against probe sabotage attacks

- **Comprehensive Documentation**:
  - `docs/setup.md`: Installation and configuration guide
  - `docs/methodology.md`: Technical methodology and approach  
  - `docs/experiments.md`: Experiment protocols and procedures
  - Professional README with usage examples

- **Experiment Features**:
  - **Automatic Versioning**: Auto-incrementing experiment IDs (v001, v002, etc.)
  - **Configuration Logging**: Complete experiment metadata tracking
  - **Statistical Analysis**: Bootstrap confidence intervals and significance testing
  - **Batch Processing**: 32x speedup with batched GPU inference
  - **Result Caching**: Persistent model weights and activation caching

- **Attack Variants**:
  - Explicit ORANGE trigger attacks
  - Invisible Unicode steganographic attacks  
  - Subtle conservation prompt attacks
  - Alternative trigger word testing (BANANA, PURPLE, etc.)

- **Visualization Suite**:
  - Automatic plot generation (ROC curves, PCA, success summaries)
  - Interactive results viewing with inline plots
  - Experiment comparison tables
  - Statistical significance indicators

- **Infrastructure**:
  - Modal serverless GPU deployment (A100-40GB)
  - TransformerLens integration for activation extraction
  - Llama-2-7b-chat-hf model support
  - Professional error handling and logging

### Technical Improvements
- **Performance Optimization**: Batched inference reduces runtime from 3+ hours to 8-15 minutes
- **Memory Management**: fp16 precision and efficient GPU utilization
- **Reproducibility**: Fixed seeds, deterministic results, version control
- **Cost Optimization**: Model caching, spot instance support, usage tracking
- **Import System**: Fixed relative imports in visualization modules, added ROC comparison plotting
- **Notebook Structure**: Restructured from tutorial format to practical research workflow

### Research Capabilities
- **Success Metrics**: 30+ percentage point AUROC drop threshold
- **Statistical Validation**: Mann-Whitney U tests, bootstrap sampling
- **Multi-Layer Analysis**: Hook point flexibility across transformer layers  
- **Defense Testing**: Ensemble methods and adversarial training frameworks

## [Unreleased]

### Planned Features
- **Layer Sweep Analysis**: Systematic vulnerability testing across all 32 layers
- **Ensemble Defense Methods**: Multi-probe voting systems
- **Cross-Model Validation**: Support for Llama-13B and Llama-70B
- **Multi-Concept Extensions**: Beyond elephant detection to other semantic categories
- **Advanced Steganography**: Frequency-based and pattern-based hidden triggers
- **Real-time Monitoring**: Live experiment tracking and cost monitoring
- **Automated Defense**: Dynamic prompt modification and robustness testing

### Future Improvements
- **Performance**: GPU multi-tenancy and parallel experiment execution
- **Usability**: Web interface for experiment management
- **Integration**: HuggingFace datasets integration, Weights & Biases logging
- **Research**: Publication-ready experiment templates and analysis tools

---

## Version Numbering

- **Major version** (X.0.0): Breaking changes to API or core methodology
- **Minor version** (0.X.0): New features, attack methods, or significant improvements  
- **Patch version** (0.0.X): Bug fixes, documentation updates, minor improvements

## Development Status

This is an active research project. The codebase is production-ready for research use but may undergo API changes as new attack methods and defense mechanisms are developed.

For the latest development updates, see the [GitHub repository](https://github.com/yuvanesh/subvert).