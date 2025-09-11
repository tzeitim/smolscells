"""Tree generation functions for SmolScells simulations"""

import numpy as np
import copy
from typing import Dict, Any

import cassiopeia as cass
from cassiopeia.simulator import (
    BirthDeathFitnessSimulator,
    UniformLeafSubsampler,
    Cas9LineageTracingDataSimulator
)


def generate_state_priors(k: int, m: int, exp: float) -> Dict[int, float]:
    """
    Generate state priors (q_i) for character states
    
    Args:
        k: Number of integration barcodes (not used directly here)
        m: Number of possible mutation states
        exp: Exponential parameter for prior distribution
        
    Returns:
        Dictionary mapping state index to prior probability
    """
    state_priors = np.array([np.random.exponential(exp) for _ in range(m)])
    state_priors /= np.sum(state_priors)
    return {i: state_priors[i] for i in range(m)}


def generate_mutation_rates(k: int, cassette_size: int, mutation_config: Dict[str, Any]) -> np.ndarray:
    """
    Generate site-specific mutation rates using different patterns
    
    Args:
        k: Number of integration barcodes
        cassette_size: Number of edit sites per integration barcode
        mutation_config: Configuration for mutation rate patterns
        
    Returns:
        Array of mutation rates for each site
    """
    total_sites = k * cassette_size
    mutation_rates = np.zeros(total_sites)
    
    start_rate = mutation_config['start_rate']
    end_rate = mutation_config['end_rate']
    pattern = mutation_config.get('pattern', 'exponential_decay')
    
    # Generate per-intbc scaling factors
    intbc_scales = np.ones(k)
    if k > 1:
        intbc_start_scale = mutation_config.get('intbc_start_scale', 1.0)
        intbc_end_scale = mutation_config.get('intbc_end_scale', 1.0)
        intbc_pattern = mutation_config.get('intbc_scale_pattern', 'uniform')
        
        if intbc_pattern == 'linear_decay':
            intbc_scales = np.linspace(intbc_start_scale, intbc_end_scale, k)
        elif intbc_pattern == 'exponential_decay':
            if k > 1:
                decay_factor = (intbc_end_scale / intbc_start_scale) ** (1 / (k - 1))
                intbc_scales = intbc_start_scale * (decay_factor ** np.arange(k))
    
    # Generate mutation rates for each intbc
    for intbc_idx in range(k):
        intbc_scale = intbc_scales[intbc_idx]
        
        for site_idx in range(cassette_size):
            if cassette_size > 1:
                normalized_idx = site_idx / (cassette_size - 1)
            else:
                normalized_idx = 0
            
            if pattern == 'linear_decay':
                site_rate = start_rate + (end_rate - start_rate) * normalized_idx
            elif pattern == 'exponential_decay':
                if cassette_size > 1:
                    decay_factor = (end_rate / start_rate) ** (1 / (cassette_size - 1))
                    site_rate = start_rate * (decay_factor ** site_idx)
                else:
                    site_rate = start_rate
            else:  # uniform
                site_rate = start_rate
            
            global_site_idx = intbc_idx * cassette_size + site_idx
            mutation_rates[global_site_idx] = site_rate * intbc_scale
    
    return mutation_rates


def generate_ground_truth(tree_config: Dict[str, Any]) -> tuple:
    """
    Simulate both pure lineage tree and recorded ground truth tree
    
    Args:
        tree_config: Configuration for tree generation
        
    Returns:
        Tuple of (pure_gt_tree, recorded_gt_tree)
    """
    print("Generating ground truth trees...")
    
    # Simulate original tree topology
    topology_simulator = BirthDeathFitnessSimulator(
        **tree_config['fitness'], 
        num_extant=int(tree_config['N'])
    )
    original_topology = topology_simulator.simulate_tree()
    
    # Pure GT tree: Keep the full original lineage (N cells)
    pure_gt_tree = copy.deepcopy(original_topology)
    pure_gt_tree.scale_to_unit_length()
    
    # Recorded GT tree: Subsample to recorded cells (n cells) for Cas9 recording
    leaf_subsampler = UniformLeafSubsampler(number_of_leaves=int(tree_config['n']))
    recorded_gt_tree = leaf_subsampler.subsample_leaves(original_topology)
    recorded_gt_tree.scale_to_unit_length()
    
    # Get parameters for Cas9 simulation
    k, m, exp = tree_config['k'], tree_config['m'], tree_config['state_priors_exponents']
    cassette_size = tree_config['cassette_size']
    
    # Create mutation config
    mutation_config = {
        'start_rate': tree_config.get('mutation_start_rate', 0.5),
        'end_rate': tree_config.get('mutation_end_rate', 0.1),
        'pattern': tree_config.get('mutation_pattern', 'exponential_decay'),
        'intbc_start_scale': tree_config.get('intbc_start_scale', 1.0),
        'intbc_end_scale': tree_config.get('intbc_end_scale', 1.0),
        'intbc_scale_pattern': tree_config.get('intbc_scale_pattern', 'uniform')
    }
    
    # Generate mutation rates
    mutation_rates = generate_mutation_rates(k, cassette_size, mutation_config)
    
    # Generate state priors for all m mutation states
    state_priors = generate_state_priors(m, m, float(exp))
    
    # Apply Cas9 lineage tracing to recorded tree
    lt_simulator = Cas9LineageTracingDataSimulator(
        number_of_cassettes=k * cassette_size,
        size_of_cassette=1,
        number_of_states=m,
        mutation_rate=mutation_rates,
        state_priors=state_priors
    )
    
    lt_simulator.overlay_data(recorded_gt_tree)
    
    # Set tree parameters - assign the same state prior distribution to each character position
    recorded_gt_tree.priors = {i: state_priors for i in range(k * cassette_size)}
    recorded_gt_tree.parameters["stochastic_missing_rate"] = 0
    recorded_gt_tree.parameters["heritable_missing_rate"] = 0
    
    return pure_gt_tree, recorded_gt_tree