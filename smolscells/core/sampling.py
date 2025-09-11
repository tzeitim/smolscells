"""Sampling strategies for SmolScells simulations"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any


def perform_mutually_exclusive_sampling(
    character_matrix: pd.DataFrame,
    sc_rate: float,
    sm_rate: float
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Perform mutually exclusive sampling: first sample total pool, then split into SC vs bulk
    
    Args:
        character_matrix: Full character matrix from recorded GT tree
        sc_rate: Single-cell sampling rate
        sm_rate: Single-molecule (bulk) sampling rate
        
    Returns:
        Tuple of (sc_matrix, sm_matrix, sc_cell_ids, sm_cell_ids)
    """
    print(f"Performing mutually exclusive sampling: SC={sc_rate*100}%, SM={sm_rate*100}%")
    
    total_cells = len(character_matrix)
    
    # Step 1: Sample total pool (sum of fractions)
    total_sample_rate = sc_rate + sm_rate
    n_total_sampled = int(total_cells * total_sample_rate)
    
    # Sample cells for the combined pool
    all_cell_ids = list(character_matrix.index)
    sampled_cell_ids = np.random.choice(all_cell_ids, size=n_total_sampled, replace=False)
    
    # Step 2: Split sampled pool into SC vs bulk 
    sc_fraction_of_sample = sc_rate / total_sample_rate
    n_sc = int(len(sampled_cell_ids) * sc_fraction_of_sample)
    
    # Split the sampled cells
    sc_cell_ids = sampled_cell_ids[:n_sc]
    sm_cell_ids = sampled_cell_ids[n_sc:]
    
    # Create matrices
    sc_matrix = character_matrix.loc[sc_cell_ids].copy()
    sm_matrix = character_matrix.loc[sm_cell_ids].copy() 
    
    print(f"Sampling results:")
    print(f"  Total cells: {total_cells}")
    print(f"  Total sampled: {n_total_sampled} ({total_sample_rate*100:.1f}%)")
    print(f"  SC cells: {len(sc_cell_ids)} ({len(sc_cell_ids)/total_cells*100:.1f}%)")
    print(f"  SM cells: {len(sm_cell_ids)} ({len(sm_cell_ids)/total_cells*100:.1f}%)")
    print(f"  Overlap: 0 cells (mutually exclusive)")
    
    return sc_matrix, sm_matrix, sc_cell_ids, sm_cell_ids


def sample_single_molecule_data(
    character_matrix: pd.DataFrame,
    k: int,
    cassette_size: int,
    sampling_rate: float = 0.5,
    sm_cell_ids: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Sample single-molecule data from recorded GT tree
    
    Args:
        character_matrix: Full character matrix
        k: Number of integration barcodes
        cassette_size: Number of sites per integration barcode
        sampling_rate: Fraction of cells to sample
        sm_cell_ids: Optional list of specific cell IDs to use
        
    Returns:
        Dictionary mapping intbc names to their character matrices
    """
    print(f"Simulating single-molecule fraction with {sampling_rate*100}% sampling rate...")
    
    # If specific cell IDs provided, use them, otherwise sample normally
    if sm_cell_ids is not None:
        available_cells = sm_cell_ids
        print(f"Using {len(available_cells)} pre-selected cells for bulk sampling")
    else:
        available_cells = list(character_matrix.index)
    
    single_molecule_data = {}
    
    for intbc_idx in range(k):
        # Get edit sites for this intbc
        start_col = intbc_idx * cassette_size
        end_col = (intbc_idx + 1) * cassette_size
        intbc_data = character_matrix.iloc[:, start_col:end_col].copy()
        
        # For bulk sampling, cells can be reused across different intbcs
        n_sample = min(int(len(character_matrix) * sampling_rate), len(available_cells))
        if n_sample == 0:
            print(f"Warning: No cells available for intbc_{intbc_idx}")
            continue
            
        sampled_cells = np.random.choice(available_cells, size=n_sample, replace=False)
        
        # Create intbc matrix using only the relevant integration barcode data
        intbc_matrix = intbc_data.loc[sampled_cells].copy()
        intbc_matrix.columns = [f'intbc_{intbc_idx}_site_{i}' for i in range(cassette_size)]
        
        single_molecule_data[f'intbc_{intbc_idx}'] = intbc_matrix
        print(f"Generated intbc_{intbc_idx} dataset: {intbc_matrix.shape}")
    
    print(f"Generated {len(single_molecule_data)} single-molecule datasets")
    print(f"Each dataset samples from {len(available_cells)} available bulk cells")
    
    return single_molecule_data


def apply_single_cell_dropout(
    character_matrix: pd.DataFrame,
    dropout_config: Dict[str, Any],
    cassette_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply dropout to single-cell data based on specified pattern
    
    Args:
        character_matrix: Input character matrix
        dropout_config: Dropout configuration
        cassette_config: Cassette configuration (for intbc information)
        
    Returns:
        Character matrix with dropout applied
    """
    if not dropout_config.get('enabled', False):
        return character_matrix
    
    matrix = character_matrix.copy()
    n_cells, n_sites = matrix.shape
    base_dropout_rate = dropout_config.get('rate', 0.15)
    pattern = dropout_config.get('pattern', 'uniform')
    
    print(f"  Applying {pattern} dropout pattern with base rate {base_dropout_rate:.1%}")
    
    if pattern == 'uniform':
        # Uniform dropout across all sites
        dropout_mask = np.random.random((n_cells, n_sites)) < base_dropout_rate
        valid_positions = matrix != -1
        dropout_mask = dropout_mask & valid_positions
        matrix[dropout_mask] = -1
        
    elif pattern == 'per_intbc':
        # Block-level dropout: entire intbc blocks are lost per cell
        intbc_variability = dropout_config.get('intbc_variability', 0.1)
        sites_per_intbc = cassette_config.get('sites_per_intbc', 10)
        num_intbc = cassette_config.get('num_intbc', 4)
        
        # Track intbc dropout statistics
        intbc_dropout_stats = {}
        
        # For each cell, decide which intbc blocks to drop
        for cell_idx in range(n_cells):
            for intbc_idx in range(num_intbc):
                start_site = intbc_idx * sites_per_intbc
                end_site = min((intbc_idx + 1) * sites_per_intbc, n_sites)
                
                # Vary dropout rate for this intbc
                intbc_dropout_rate = base_dropout_rate * (1 + np.random.uniform(-intbc_variability, intbc_variability))
                intbc_dropout_rate = np.clip(intbc_dropout_rate, 0.0, 1.0)
                
                # Decide if this entire intbc block is dropped for this cell
                if np.random.random() < intbc_dropout_rate:
                    # Drop entire intbc block
                    matrix.iloc[cell_idx, start_site:end_site] = -1
                    
                    # Track statistics
                    if intbc_idx not in intbc_dropout_stats:
                        intbc_dropout_stats[intbc_idx] = 0
                    intbc_dropout_stats[intbc_idx] += 1
        
        # Print block-level dropout statistics
        for intbc_idx in range(num_intbc):
            start_site = intbc_idx * sites_per_intbc
            end_site = min((intbc_idx + 1) * sites_per_intbc, n_sites)
            cells_dropped = intbc_dropout_stats.get(intbc_idx, 0)
            block_dropout_rate = cells_dropped / n_cells
            print(f"    intbc_{intbc_idx}: sites {start_site}-{end_site-1}, "
                  f"block dropped in {cells_dropped}/{n_cells} cells ({block_dropout_rate:.1%})")
            
    elif pattern == 'per_cell':
        # Different dropout rates per cell (simulating cell quality variation)
        cell_variability = dropout_config.get('cell_variability', 0.2)
        
        # Generate per-cell dropout rates
        for cell_idx in range(n_cells):
            # Higher variability means some cells can have much higher dropout
            cell_dropout_multiplier = 1 + np.random.uniform(0, cell_variability)
            cell_dropout_rate = base_dropout_rate * cell_dropout_multiplier
            cell_dropout_rate = np.clip(cell_dropout_rate, 0.0, 1.0)
            
            # Apply dropout to this cell
            dropout_mask = np.random.random(n_sites) < cell_dropout_rate
            valid_positions = matrix.iloc[cell_idx] != -1
            dropout_mask = dropout_mask & valid_positions.values
            matrix.iloc[cell_idx, dropout_mask] = -1
    
    # Calculate and print dropout statistics
    total_sites = n_cells * n_sites
    valid_sites_before = (character_matrix != -1).sum().sum()
    valid_sites_after = (matrix != -1).sum().sum()
    actual_dropout = (valid_sites_before - valid_sites_after) / valid_sites_before if valid_sites_before > 0 else 0
    
    print(f"  Actual dropout applied: {actual_dropout:.1%}")
    print(f"  Valid sites: {valid_sites_before} → {valid_sites_after}")
    
    return matrix