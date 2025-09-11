"""
Configuration loader for SMOL-SCELLS simulation
Handles loading and validation of YAML configuration files
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class ConfigLoader:
    """Load and validate simulation configuration from YAML files"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default config.
        """
        self.config_path = config_path
        self.config = None
        
    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Dictionary containing configuration parameters
        """
        if config_path:
            self.config_path = config_path
            
        if not self.config_path:
            # Use default config if no path specified
            default_path = Path(__file__).parent / "config" / "default_config.yaml"
            if default_path.exists():
                self.config_path = str(default_path)
            else:
                raise FileNotFoundError(f"Default config not found at {default_path}")
        
        # Load YAML file
        with open(self.config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Process and validate configuration
        self.config = self._process_config(raw_config)
        
        return self.config
    
    def _process_config(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw configuration into format expected by simulator
        
        Args:
            raw_config: Raw configuration from YAML
            
        Returns:
            Processed configuration dictionary
        """
        config = {}
        
        # Process tree configuration
        tree_cfg = raw_config.get('tree', {})
        fitness_cfg = tree_cfg.get('fitness', {})
        
        config['tree_config'] = {
            'N': tree_cfg.get('original_cells', 10000),
            'n': tree_cfg.get('sampled_cells', 1000),
            'fitness': self._create_fitness_config(fitness_cfg),
            'k': raw_config.get('cassette', {}).get('num_intbc', 4),
            'cassette_size': raw_config.get('cassette', {}).get('sites_per_intbc', 10),
            'm': raw_config.get('cassette', {}).get('num_states', 50),
            'mutation_start_rate': raw_config.get('cassette', {}).get('mutation_start_rate', 0.5),
            'mutation_end_rate': raw_config.get('cassette', {}).get('mutation_end_rate', 0.05),
            'mutation_pattern': raw_config.get('cassette', {}).get('mutation_pattern', 'exponential_decay'),
            'intbc_start_scale': raw_config.get('cassette', {}).get('intbc_start_scale', 1.0),
            'intbc_end_scale': raw_config.get('cassette', {}).get('intbc_end_scale', 1.0),
            'intbc_scale_pattern': raw_config.get('cassette', {}).get('intbc_scale_pattern', 'uniform'),
            'state_priors_exponents': raw_config.get('cassette', {}).get('state_priors_exp', 1e-5),
        }
        
        # Process sampling configuration
        sampling_cfg = raw_config.get('sampling', {})
        config['sm_sampling_rate'] = sampling_cfg.get('single_molecule_rate', 0.5)
        config['sc_sampling_rate'] = sampling_cfg.get('single_cell_rate', 0.2)
        
        # Process dropout configuration
        config['dropout'] = raw_config.get('dropout', {})
        
        # Process solver configuration
        solver_cfg = raw_config.get('solver', {})
        config['solver'] = {
            'primary': solver_cfg.get('primary', 'neighbor_joining'),
            'fallback': solver_cfg.get('fallback', 'upgma'),
            'add_root': solver_cfg.get('add_root', True)
        }
        
        # Process output configuration
        output_cfg = raw_config.get('output', {})
        config['output'] = {
            'plot_dir': output_cfg.get('plot_dir', 'plots'),
            'results_dir': output_cfg.get('results_dir', 'results'),
            'save_intermediate': output_cfg.get('save_intermediate', True),
            'save_trees': output_cfg.get('save_trees', True),  # Add save_trees option
            'plot': output_cfg.get('plot', {})
        }
        
        # Random seed and verbosity
        config['random_seed'] = raw_config.get('random_seed', None)
        config['verbosity'] = raw_config.get('verbosity', 1)
        
        return config
    
    def _create_fitness_config(self, fitness_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create fitness configuration with lambda functions
        
        Args:
            fitness_cfg: Fitness configuration from YAML
            
        Returns:
            Fitness configuration with lambda functions
        """
        birth_scale = fitness_cfg.get('initial_birth_scale', 2)
        mutation_prob = fitness_cfg.get('mutation_probability', 0.5)
        fitness_mean = fitness_cfg.get('fitness_mean', 0.5)
        fitness_std = fitness_cfg.get('fitness_std', 0.25)
        fitness_base = fitness_cfg.get('fitness_base', 1.1)
        
        return {
            'birth_waiting_distribution': lambda scale: np.random.exponential(1/scale),
            'initial_birth_scale': birth_scale,
            'death_waiting_distribution': lambda: np.inf,
            'mutation_distribution': lambda: 1 if np.random.uniform() < mutation_prob else 0,
            'fitness_distribution': lambda: np.random.normal(fitness_mean, fitness_std),
            'fitness_base': fitness_base
        }
    
    def save_config(self, config: Dict[str, Any], output_path: str):
        """
        Save configuration to YAML file (without lambda functions)
        
        Args:
            config: Configuration dictionary
            output_path: Path to save configuration
        """
        # Create a serializable version of the config
        serializable_config = self._make_serializable(config)
        
        with open(output_path, 'w') as f:
            yaml.dump(serializable_config, f, default_flow_style=False, sort_keys=False)
    
    def _make_serializable(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert configuration to serializable format (remove lambda functions)
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Serializable configuration
        """
        serializable = copy.deepcopy(config)
        
        # Remove lambda functions from fitness config
        if 'tree_config' in serializable and 'fitness' in serializable['tree_config']:
            fitness = serializable['tree_config']['fitness']
            # Keep only serializable parameters
            serializable['tree_config']['fitness'] = {
                'initial_birth_scale': fitness.get('initial_birth_scale', 2),
                'fitness_base': fitness.get('fitness_base', 1.1)
            }
        
        return serializable
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate tree parameters
        if config['tree_config']['N'] <= config['tree_config']['n']:
            raise ValueError("Original tree size (N) must be larger than sampled size (n)")
        
        # Validate cassette parameters
        if config['tree_config']['k'] <= 0:
            raise ValueError("Number of intbc (k) must be positive")
        
        if config['tree_config']['cassette_size'] <= 0:
            raise ValueError("Cassette size must be positive")
        
        # Validate sampling rates
        if not 0 < config['sm_sampling_rate'] <= 1:
            raise ValueError("Single-molecule sampling rate must be between 0 and 1")
        
        if not 0 < config['sc_sampling_rate'] <= 1:
            raise ValueError("Single-cell sampling rate must be between 0 and 1")
        
        return True
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries
        
        Args:
            base_config: Base configuration
            override_config: Configuration with override values
            
        Returns:
            Merged configuration
        """
        merged = copy.deepcopy(base_config)
        
        def deep_merge(dict1, dict2):
            for key, value in dict2.items():
                if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                    deep_merge(dict1[key], value)
                else:
                    dict1[key] = value
        
        deep_merge(merged, override_config)
        return merged


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load configuration
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.load()