"""Main simulator class for SmolScells"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from ..config.loader import load_config
from .tree_generation import generate_ground_truth
from .sampling import (
    perform_mutually_exclusive_sampling,
    sample_single_molecule_data,
    apply_single_cell_dropout
)
from .reconstruction import build_all_trees


class SmolScellsSimulator:
    """Core smol-scells simulation orchestrator"""
    
    def __init__(self, config_path: str = None, config: Dict = None, run_name: str = None):
        """
        Initialize simulator with configuration
        
        Args:
            config_path: Path to YAML configuration file
            config: Configuration dictionary (alternative to config_path)
            run_name: Name for this simulation run
        """
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)
        
        # Set run name and output directories
        self.run_name = run_name or self._generate_run_name()
        self._setup_output_directories()
        
        # Set random seed
        if 'random_seed' in self.config and self.config['random_seed'] is not None:
            np.random.seed(self.config['random_seed'])
            print(f"Random seed set to: {self.config['random_seed']}")
        
        # Initialize tree storage
        self.pure_gt_tree = None
        self.recorded_gt_tree = None
        self.single_cell_data = None
        self.single_molecule_data = {}
        self.trees = {}
        
        print(f"Initialized SmolScellsSimulator with run_name: {self.run_name}")
    
    def _generate_run_name(self) -> str:
        """Generate a default run name based on timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"smol_scells_{timestamp}"
    
    def _setup_output_directories(self):
        """Create standardized output directory structure"""
        output_config = self.config.get('output', {})
        plot_dir = output_config.get('plot_dir', f"output/{self.run_name}/plots")
        results_dir = output_config.get('results_dir', f"output/{self.run_name}/results")
        
        # Convert to Path objects
        plot_path = Path(plot_dir)
        results_path = Path(results_dir)
        base_output_dir = plot_path.parent if plot_path.parent != Path('.') else Path("output") / self.run_name
        
        self.output_dirs = {
            'base': base_output_dir,
            'plots': plot_path,
            'results': results_path, 
            'data': base_output_dir / "data",
            'trees': base_output_dir / "trees",
            'fitness': base_output_dir / "fitness",
            'pycea': plot_path / "pycea"
        }
        
        # Create all directories
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Created output directory structure at: {base_output_dir}")
    
    def get_output_path(self, category: str, filename: str) -> Path:
        """
        Get full output path for a file in a specific category
        
        Args:
            category: Output category (e.g., 'plots', 'results', 'trees')
            filename: Name of the file
            
        Returns:
            Full path to the output file
        """
        if category not in self.output_dirs:
            raise ValueError(f"Unknown output category: {category}. "
                           f"Available: {list(self.output_dirs.keys())}")
        return self.output_dirs[category] / filename
    
    def simulate_ground_truth(self):
        """Generate ground truth trees"""
        self.pure_gt_tree, self.recorded_gt_tree = generate_ground_truth(
            self.config['tree_config']
        )
        print(f"Generated GT trees: {len(self.pure_gt_tree.leaves)} → "
              f"{len(self.recorded_gt_tree.leaves)} cells")
    
    def simulate_sampling(self):
        """Perform mutually exclusive sampling for single-cell and single-molecule data"""
        character_matrix = self.recorded_gt_tree.character_matrix.copy()
        
        # Perform mutually exclusive sampling
        sc_matrix, sm_matrix, sc_cell_ids, sm_cell_ids = perform_mutually_exclusive_sampling(
            character_matrix,
            self.config['sc_sampling_rate'],
            self.config['sm_sampling_rate']
        )
        
        # Apply dropout to single-cell data
        self.single_cell_data = apply_single_cell_dropout(
            sc_matrix,
            self.config.get('dropout', {}),
            self.config.get('cassette', {})
        )
        
        # Sample single-molecule data per intbc
        self.single_molecule_data = sample_single_molecule_data(
            character_matrix,
            self.config['tree_config']['k'],
            self.config['tree_config']['cassette_size'],
            self.config['sm_sampling_rate'],
            sm_cell_ids
        )
    
    def reconstruct_trees(self):
        """Reconstruct trees from sampled data"""
        self.trees = build_all_trees(
            self.recorded_gt_tree,
            self.single_cell_data,
            self.single_molecule_data,
            self.config
        )
    
    def run_complete_simulation(self):
        """
        Run the complete simulation pipeline:
        1. Generate ground truth
        2. Sample data
        3. Reconstruct trees
        
        Returns:
            Dictionary containing all results
        """
        print("\n" + "="*60)
        print("Starting SmolScells Simulation")
        print("="*60)
        
        # Step 1: Generate ground truth
        print("\n[Step 1] Generating ground truth trees...")
        self.simulate_ground_truth()
        
        # Step 2: Sample data
        print("\n[Step 2] Sampling single-cell and single-molecule data...")
        self.simulate_sampling()
        
        # Step 3: Reconstruct trees
        print("\n[Step 3] Reconstructing trees from sampled data...")
        self.reconstruct_trees()
        
        print("\n" + "="*60)
        print("Simulation Complete!")
        print("="*60)
        
        # Return results
        return {
            'pure_gt_tree': self.pure_gt_tree,
            'recorded_gt_tree': self.recorded_gt_tree,
            'single_cell_data': self.single_cell_data,
            'single_molecule_data': self.single_molecule_data,
            'trees': self.trees,
            'config': self.config,
            'output_dirs': self.output_dirs
        }
    
    def save_results(self, save_trees: bool = True, save_matrices: bool = True):
        """
        Save simulation results to disk
        
        Args:
            save_trees: Whether to save tree files
            save_matrices: Whether to save character matrices
        """
        if save_trees and self.config.get('output', {}).get('save_trees', True):
            print("Saving trees...")
            for tree_name, tree in self.trees.items():
                tree_path = self.get_output_path('trees', f"{tree_name}.pkl")
                tree.to_pickle(tree_path)
                print(f"  Saved {tree_name} to {tree_path}")
        
        if save_matrices:
            print("Saving character matrices...")
            # Save single-cell matrix
            if self.single_cell_data is not None:
                sc_path = self.get_output_path('data', 'single_cell_matrix.csv')
                self.single_cell_data.to_csv(sc_path)
                print(f"  Saved single-cell matrix to {sc_path}")
            
            # Save single-molecule matrices
            for intbc_name, intbc_matrix in self.single_molecule_data.items():
                sm_path = self.get_output_path('data', f'{intbc_name}_matrix.csv')
                intbc_matrix.to_csv(sm_path)
                print(f"  Saved {intbc_name} matrix to {sm_path}")