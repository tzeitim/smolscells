"""
SmolScells: Single-Molecule Single-Cells Lineage Tracing

A Python package for simulating and analyzing lineage tracing experiments that integrate
single-cell and bulk single-molecule sequencing data for enhanced statistical power.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from .core.simulator import SmolScellsSimulator
from .core.tree_generation import (
    generate_ground_truth,
    generate_state_priors,
    generate_mutation_rates,
)
from .core.sampling import (
    perform_mutually_exclusive_sampling,
    apply_single_cell_dropout,
    sample_single_molecule_data,
)
from .core.reconstruction import (
    reconstruct_tree,
    build_all_trees,
)

# Config imports
from .config.loader import ConfigLoader, load_config

# Analysis imports
from .analysis.fitness import FitnessAnalyzer
from .analysis.comparison import TreeComparator

# Visualization imports (optional)
try:
    from .visualization.plots import TreeVisualizer
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

__all__ = [
    # Version info
    "__version__",
    # Core classes
    "SmolScellsSimulator",
    # Core functions
    "generate_ground_truth",
    "generate_state_priors",
    "generate_mutation_rates",
    "perform_mutually_exclusive_sampling",
    "apply_single_cell_dropout",
    "sample_single_molecule_data",
    "reconstruct_tree",
    "build_all_trees",
    # Config
    "ConfigLoader",
    "load_config",
    # Analysis
    "FitnessAnalyzer",
    "TreeComparator",
]

if _HAS_VIZ:
    __all__.append("TreeVisualizer")