"""Tree reconstruction functions for SmolScells simulations"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Set
import cassiopeia as cass
from cassiopeia.solver import NeighborJoiningSolver, UPGMASolver


def reconstruct_tree(
    character_matrix: pd.DataFrame,
    priors: Optional[Dict] = None,
    solver_config: Optional[Dict] = None,
    collapse_mutationless: bool = True,
    tree_name: str = "tree"
) -> Optional[cass.data.CassiopeiaTree]:
    """
    Reconstruct a tree from character matrix using specified solver
    
    Args:
        character_matrix: Character matrix for tree reconstruction
        priors: Prior probabilities for states
        solver_config: Solver configuration
        collapse_mutationless: Whether to collapse mutationless edges
        tree_name: Name for debugging output
        
    Returns:
        Reconstructed CassiopeiaTree or None if failed
    """
    if solver_config is None:
        solver_config = {'primary': 'neighbor_joining', 'fallback': 'upgma', 'add_root': True}
    
    # Create tree object
    tree = cass.data.CassiopeiaTree(character_matrix=character_matrix)
    
    if priors is not None:
        tree.priors = priors
    
    # Try primary solver
    primary_solver = solver_config.get('primary', 'neighbor_joining')
    add_root = solver_config.get('add_root', True)
    
    if primary_solver == 'neighbor_joining':
        solver = NeighborJoiningSolver(add_root=add_root)
    elif primary_solver == 'upgma':
        solver = UPGMASolver()
    else:
        solver = NeighborJoiningSolver(add_root=add_root)
    
    try:
        solver.solve(tree, collapse_mutationless_edges=collapse_mutationless)
        tree.reconstruct_ancestral_characters()
        print(f"{tree_name} reconstruction succeeded with {primary_solver} "
              f"(collapse_mutationless={collapse_mutationless})")
        return tree
    except Exception as e:
        print(f"{tree_name} reconstruction failed with {primary_solver}: {e}")
        
        # Try fallback solver
        fallback_solver = solver_config.get('fallback', 'upgma')
        if fallback_solver and fallback_solver != primary_solver:
            try:
                if fallback_solver == 'upgma':
                    solver = UPGMASolver()
                else:
                    solver = NeighborJoiningSolver(add_root=add_root)
                    
                solver.solve(tree, collapse_mutationless_edges=collapse_mutationless)
                tree.reconstruct_ancestral_characters()
                print(f"{tree_name} reconstruction succeeded with fallback {fallback_solver}")
                return tree
            except Exception as e2:
                print(f"{tree_name} reconstruction also failed with {fallback_solver}: {e2}")
    
    return None


def build_all_trees(
    recorded_gt_tree: cass.data.CassiopeiaTree,
    single_cell_data: Optional[pd.DataFrame],
    single_molecule_data: Dict[str, pd.DataFrame],
    config: Dict[str, Any]
) -> Dict[str, cass.data.CassiopeiaTree]:
    """
    Build all trees from the various data modalities
    
    Args:
        recorded_gt_tree: Recorded ground truth tree
        single_cell_data: Single-cell character matrix
        single_molecule_data: Dictionary of single-molecule matrices per intbc
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping tree names to CassiopeiaTree objects
    """
    trees = {}
    
    # Get solver and plot configuration
    solver_config = config.get('solver', {})
    collapse_mutationless = config.get('output', {}).get('plot', {}).get('collapse_mutationless', True)
    
    # Build recorded GT tree
    if recorded_gt_tree is not None:
        char_matrix = recorded_gt_tree.character_matrix
        priors = recorded_gt_tree.priors
        
        # Validate and clean character matrix
        m = config['tree_config']['m']
        unique_states = set(char_matrix.values.flatten())
        problematic_states = [s for s in unique_states if s >= m and s != -1]
        
        if problematic_states:
            print(f"Warning: Filtering character states beyond range 0-{m-1}: {problematic_states}")
            char_matrix = char_matrix.copy()
            for state in problematic_states:
                char_matrix = char_matrix.replace(state, -1)
        
        recorded_tree = reconstruct_tree(
            char_matrix, priors, solver_config, 
            collapse_mutationless, "Recorded GT"
        )
        if recorded_tree:
            trees['recorded_ground_truth'] = recorded_tree
    
    # Build single-cell tree
    if single_cell_data is not None:
        sc_tree = reconstruct_tree(
            single_cell_data,
            recorded_gt_tree.priors if recorded_gt_tree else None,
            solver_config,
            collapse_mutationless,
            "Single-cell"
        )
        if sc_tree:
            trees['single_cell'] = sc_tree
    
    # Build single-molecule trees
    cassette_size = config['tree_config']['cassette_size']
    for intbc_name, intbc_matrix in single_molecule_data.items():
        # Set priors for this intbc
        intbc_idx = int(intbc_name.split('_')[1])
        intbc_priors = {}
        
        if recorded_gt_tree and hasattr(recorded_gt_tree, 'priors'):
            for site_idx in range(cassette_size):
                gt_character_idx = intbc_idx * cassette_size + site_idx
                if gt_character_idx in recorded_gt_tree.priors:
                    intbc_priors[site_idx] = recorded_gt_tree.priors[gt_character_idx]
        
        sm_tree = reconstruct_tree(
            intbc_matrix,
            intbc_priors if intbc_priors else None,
            solver_config,
            collapse_mutationless,
            f"Single-molecule {intbc_name}"
        )
        if sm_tree:
            trees[f'single_molecule_{intbc_name}'] = sm_tree
    
    print(f"Built {len(trees)} trees total")
    return trees