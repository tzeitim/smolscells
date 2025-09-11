"""Command-line interface for SmolScells"""

import click
import sys
from pathlib import Path
from typing import Optional

from .core.simulator import SmolScellsSimulator
from .config.loader import ConfigLoader


@click.group()
@click.version_option(version="0.1.0", prog_name="smolscells")
def main():
    """SmolScells: Single-Molecule Single-Cells Lineage Tracing Simulator"""
    pass


@main.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True),
    help='Path to YAML configuration file'
)
@click.option(
    '--run-name', '-n',
    type=str,
    help='Name for this simulation run'
)
@click.option(
    '--seed', '-s',
    type=int,
    help='Random seed for reproducibility'
)
@click.option(
    '--save-trees/--no-save-trees',
    default=True,
    help='Save reconstructed trees to disk'
)
@click.option(
    '--save-matrices/--no-save-matrices',
    default=True,
    help='Save character matrices to disk'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
def simulate(
    config: Optional[str],
    run_name: Optional[str],
    seed: Optional[int],
    save_trees: bool,
    save_matrices: bool,
    verbose: bool
):
    """Run a SmolScells simulation"""
    
    # Load configuration
    if config:
        config_path = Path(config)
        if not config_path.exists():
            click.echo(f"Error: Configuration file not found: {config}", err=True)
            sys.exit(1)
    else:
        # Use default config
        config_path = None
        click.echo("Using default configuration")
    
    try:
        # Load config
        loader = ConfigLoader(config_path)
        sim_config = loader.load()
        
        # Override seed if provided
        if seed is not None:
            sim_config['random_seed'] = seed
        
        # Set verbosity
        if verbose:
            sim_config['verbosity'] = 2
        
        # Create and run simulator
        click.echo("\n" + "="*60)
        click.echo("SmolScells Simulation Starting")
        click.echo("="*60)
        
        simulator = SmolScellsSimulator(config=sim_config, run_name=run_name)
        results = simulator.run_complete_simulation()
        
        # Save results if requested
        if save_trees or save_matrices:
            click.echo("\nSaving results...")
            simulator.save_results(save_trees=save_trees, save_matrices=save_matrices)
        
        click.echo("\n" + "="*60)
        click.echo("Simulation completed successfully!")
        click.echo(f"Results saved to: {simulator.output_dirs['base']}")
        click.echo("="*60)
        
    except Exception as e:
        click.echo(f"Error during simulation: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('config_path', type=click.Path())
def validate(config_path: str):
    """Validate a configuration file"""
    
    config_file = Path(config_path)
    if not config_file.exists():
        click.echo(f"Error: Configuration file not found: {config_path}", err=True)
        sys.exit(1)
    
    try:
        loader = ConfigLoader(config_path)
        config = loader.load()
        
        # Validate config
        if loader.validate_config(config):
            click.echo(f"✓ Configuration is valid: {config_path}")
            
            # Print summary
            click.echo("\nConfiguration Summary:")
            click.echo(f"  Tree size: {config['tree_config']['N']} → {config['tree_config']['n']} cells")
            click.echo(f"  Integration barcodes: {config['tree_config']['k']}")
            click.echo(f"  Sites per intbc: {config['tree_config']['cassette_size']}")
            click.echo(f"  Single-cell sampling: {config['sc_sampling_rate']*100}%")
            click.echo(f"  Single-molecule sampling: {config['sm_sampling_rate']*100}%")
            
            dropout = config.get('dropout', {})
            if dropout.get('enabled', False):
                click.echo(f"  Dropout: {dropout.get('rate', 0)*100}% ({dropout.get('pattern', 'uniform')})")
            else:
                click.echo("  Dropout: Disabled")
                
    except Exception as e:
        click.echo(f"Error: Invalid configuration - {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    '--output', '-o',
    type=click.Path(),
    default='default_config.yaml',
    help='Output path for the configuration file'
)
def generate_config(output: str):
    """Generate a default configuration file"""
    
    default_config = """# SmolScells Default Configuration

# Tree generation parameters
tree:
  original_cells: 10000    # Number of cells in original tree
  sampled_cells: 1000      # Number of cells after Cas9 recording
  
  fitness:
    initial_birth_scale: 2.0
    mutation_probability: 0.5
    fitness_mean: 0.5
    fitness_std: 0.25
    fitness_base: 1.1

# Cassette parameters
cassette:
  num_intbc: 4             # Number of integration barcodes
  sites_per_intbc: 10      # Edit sites per integration barcode
  num_states: 50           # Number of possible mutation states
  mutation_start_rate: 0.5
  mutation_end_rate: 0.05
  mutation_pattern: "exponential_decay"
  intbc_start_scale: 1.0
  intbc_end_scale: 1.0
  intbc_scale_pattern: "uniform"
  state_priors_exp: 1.0e-5

# Sampling parameters
sampling:
  single_molecule_rate: 0.5  # Fraction for bulk sequencing
  single_cell_rate: 0.2      # Fraction for single-cell sequencing

# Dropout simulation
dropout:
  enabled: true
  rate: 0.15
  pattern: "uniform"  # Options: uniform, per_intbc, per_cell
  intbc_variability: 0.1
  cell_variability: 0.2

# Solver parameters
solver:
  primary: "neighbor_joining"
  fallback: "upgma"
  add_root: true

# Output parameters
output:
  plot_dir: "plots"
  results_dir: "results"
  save_intermediate: true
  save_trees: true
  
  plot:
    collapse_mutationless: true

# Random seed for reproducibility
random_seed: 42

# Verbosity level (0=quiet, 1=normal, 2=verbose)
verbosity: 1
"""
    
    output_path = Path(output)
    output_path.write_text(default_config)
    click.echo(f"Generated default configuration: {output_path}")


@main.command()
def info():
    """Display information about SmolScells"""
    
    info_text = """
SmolScells: Single-Molecule Single-Cells Lineage Tracing Simulator
===================================================================

SmolScells is a simulation framework for evaluating the integration of 
single-cell and bulk single-molecule sequencing data in lineage tracing 
experiments.

Key Features:
- Simulate ground truth lineage trees with fitness effects
- Apply Cas9-based lineage recording
- Model single-cell dropout patterns
- Sample single-molecule data per integration barcode
- Reconstruct trees using neighbor joining or UPGMA
- Compare reconstruction accuracy across modalities

For more information:
- Documentation: https://github.com/yourusername/smolscells
- Issues: https://github.com/yourusername/smolscells/issues

Version: 0.1.0
"""
    click.echo(info_text)


if __name__ == "__main__":
    main()