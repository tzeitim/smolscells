# SmolScells: Single-Molecule Single-Cells Lineage Tracing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SmolScells is a Python package for simulating and analyzing lineage tracing experiments that integrate single-cell and bulk single-molecule sequencing data. By combining these complementary modalities, SmolScells demonstrates how to achieve enhanced statistical power for detecting fitness effects and reconstructing accurate cell lineages.

## Overview

### The Problem

Single-cell lineage tracing faces fundamental limitations:
- **Extreme sparsity**: Only 1-10% of cells are typically captured
- **Technical dropout**: 15-30% of molecular markers fail to be detected
- **High costs**: $0.50-$5 per cell limits experiment scale
- **Limited statistical power**: Small sample sizes reduce ability to detect fitness effects

### The Solution

SmolScells implements a hybrid approach that leverages:
- **Single-cell data**: Provides high-resolution cell-to-cell relationships
- **Bulk single-molecule data**: Offers complete population coverage and accurate frequency estimates
- **Data integration**: Combines both modalities for enhanced reconstruction accuracy

## Features

- 🌳 **Ground truth simulation**: Generate lineage trees with fitness effects using birth-death processes
- 🧬 **Cas9 recording**: Simulate CRISPR-based lineage recording with realistic mutation patterns
- 📊 **Flexible sampling**: Model various single-cell and bulk sampling strategies
- 💧 **Dropout simulation**: Apply realistic dropout patterns (uniform, per-intbc, per-cell)
- 🔄 **Tree reconstruction**: Use neighbor joining or UPGMA algorithms
- 📈 **Fitness analysis**: Detect clonal fitness using Local Branching Index (LBI)
- 🎯 **Comparison metrics**: Evaluate reconstruction accuracy with Robinson-Foulds and triplet metrics

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/yourusername/smolscells.git
cd smolscells
pip install -e .
```

### Using pip

```bash
pip install smolscells
```

### Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- cassiopeia-lineage >= 2.0.0
- pyyaml >= 6.0
- click >= 8.0.0

## Quick Start

### Command Line Interface

```bash
# Run simulation with default configuration
smolscells simulate

# Use custom configuration
smolscells simulate --config my_config.yaml --run-name my_experiment

# Generate a default configuration file
smolscells generate-config -o my_config.yaml

# Validate configuration
smolscells validate my_config.yaml
```

### Python API

```python
from smolscells import SmolScellsSimulator

# Create simulator with configuration file
sim = SmolScellsSimulator(config_path="config.yaml")

# Run complete simulation pipeline
results = sim.run_complete_simulation()

# Access results
pure_gt = results['pure_gt_tree']
recorded_gt = results['recorded_gt_tree']
single_cell_tree = results['trees']['single_cell']
bulk_trees = [v for k, v in results['trees'].items() if 'single_molecule' in k]

# Save results
sim.save_results(save_trees=True, save_matrices=True)
```

### Analyzing Results

```python
from smolscells.analysis import FitnessAnalyzer, TreeComparator

# Analyze fitness detection
analyzer = FitnessAnalyzer(results['trees'])
fitness_comparison = analyzer.compare_fitness_detection()

# Compare tree reconstruction accuracy
comparator = TreeComparator(recorded_gt)
metrics = comparator.compare_tree(single_cell_tree)
print(f"Robinson-Foulds distance: {metrics['robinson_foulds']:.3f}")
print(f"Triplet accuracy: {metrics['triplet_accuracy']:.3f}")
```

## Configuration

SmolScells uses YAML configuration files. Here's a minimal example:

```yaml
# Tree parameters
tree:
  original_cells: 10000
  sampled_cells: 1000
  fitness:
    initial_birth_scale: 2.0
    mutation_probability: 0.5

# Cassette parameters
cassette:
  num_intbc: 4
  sites_per_intbc: 10
  num_states: 50
  mutation_start_rate: 0.5
  mutation_end_rate: 0.05

# Sampling
sampling:
  single_molecule_rate: 0.5
  single_cell_rate: 0.2

# Dropout
dropout:
  enabled: true
  rate: 0.15
  pattern: "uniform"
```

See `configs/` directory for more examples.

## Project Structure

```
smolscells/
├── smolscells/
│   ├── core/              # Core simulation modules
│   │   ├── simulator.py   # Main simulator class
│   │   ├── tree_generation.py
│   │   ├── sampling.py
│   │   └── reconstruction.py
│   ├── config/            # Configuration management
│   ├── analysis/          # Analysis tools
│   │   ├── fitness.py
│   │   └── comparison.py
│   ├── visualization/     # Plotting utilities
│   └── cli.py            # Command-line interface
├── configs/              # Example configurations
├── examples/             # Example scripts
└── tests/               # Unit tests
```

## Scientific Background

SmolScells implements the simulation framework described in our manuscript demonstrating how integrating bulk single-molecule data with single-cell lineage tracing can:

1. **Improve branch length accuracy** by incorporating true population frequencies
2. **Detect fitness effects** with higher statistical power
3. **Identify rare subclones** missed by single-cell sampling alone
4. **Reduce experimental costs** while maintaining reconstruction quality

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use SmolScells in your research, please cite:

```bibtex
@software{smolscells2024,
  title = {SmolScells: Single-Molecule Single-Cells Lineage Tracing Simulator},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/smolscells}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of the [Cassiopeia](https://github.com/YosefLab/Cassiopeia) lineage tracing framework
- Inspired by recent advances in CRISPR-based lineage recording
- Developed as part of research on enhanced lineage tracing methods

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.