"""Sampling strategies for Smolscells simulations"""

import numpy as np
import polars as pl
import pandas as pd 
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

def mutually_exclusive_sampling(
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
    
    logger.info(f"Sampling results:")
    logger.info(f"  Total cells: {total_cells}")
    logger.info(f"  Total sampled: {n_total_sampled} ({total_sample_rate*100:.1f}%)")
    logger.info(f"  SC cells: {len(sc_cell_ids)} ({len(sc_cell_ids)/total_cells*100:.1f}%)")
    logger.info(f"  SM cells: {len(sm_cell_ids)} ({len(sm_cell_ids)/total_cells*100:.1f}%)")
    logger.info(f"  Overlap: 0 cells (mutually exclusive)")
    
    return sc_matrix, sm_matrix, sc_cell_ids, sm_cell_ids

def split_single_molecule_data(
    character_matrix: pd.DataFrame,
    number_of_cassettes: Optional[int] = None,
    size_of_cassette: Optional[int] = None,
    sampling_rate: float = 1.0,
    sm_cell_ids: Optional[List[str]] = None,
    **config
) -> Dict[str, pl.DataFrame]:
    """
    Sample single-molecule data from recorded GT tree

    Args:
        character_matrix: Full character matrix
        number_of_cassettes: Number of integration barcodes
        size_of_cassette: Number of sites per integration barcode
        sampling_rate: Fraction of cells to sample
        sm_cell_ids: Optional list of specific cell IDs to use
        **config: Optional config dict with the same keys as above

    Returns:
        Dictionary mapping intbc names to their character matrices
    """
    # Prefer explicit params, fall back to config
    number_of_cassettes = number_of_cassettes or config.get('number_of_cassettes')
    size_of_cassette = size_of_cassette or config.get('size_of_cassette')
    sampling_rate = sampling_rate if sampling_rate != 1.0 else config.get('sampling_rate', 1.0)

    if number_of_cassettes is None or size_of_cassette is None:
        raise ValueError("Must provide 'number_of_cassettes' and 'size_of_cassette' either as arguments or in config")

    logger.info(f"Simulating single-molecule fraction with {sampling_rate*100}% sampling rate...")
    logger.info(f"Using {len(sm_cell_ids)} pre-selected cells for bulk sampling")

    single_molecule_data = {}

    for intbc_idx in range(number_of_cassettes):
        # Get edit sites for this intbc
        start_col = intbc_idx * size_of_cassette
        end_col = (intbc_idx + 1) * size_of_cassette
        intbc_data = character_matrix.iloc[:, start_col:end_col].copy()

        # For bulk sampling, cells can be reused across different intbcs
        n_sample = min(int(len(character_matrix) * sampling_rate), len(sm_cell_ids))

        if n_sample == 0:
            logger.warning(f"No cells available for intBC {intbc_idx}")
            continue

        sampled_cells = np.random.choice(sm_cell_ids, size=n_sample, replace=False)

        # Create intbc matrix using only the relevant integration barcode data
        intbc_matrix = intbc_data.loc[sampled_cells].copy()
        intbc_matrix.columns = [f'site_{i}' for i in range(size_of_cassette)]

        single_molecule_data[str(intbc_idx)] = intbc_matrix
        logger.info(f"Generated intBC {intbc_idx} dataset: {intbc_matrix.shape}")

    logger.info(f"Generated {len(single_molecule_data)} single-molecule datasets")
    logger.info(f"Each dataset samples from {len(sm_cell_ids)} available bulk cells")

    return single_molecule_data

def compute_single_cell_dropout(
    character_matrix: pd.DataFrame,
    cell_multiplier: str = "inverse",
    base_dropout_rate: float = 0.15,
    dropout_config: Dict[str, Any]={},
    cassette_config: Dict[str, Any]={},
    number_of_cassettes: Optional[int] = None
    ) :
    """
    Compute dropout to single-cell data based on specified pattern

    Args:
        character_matrix: Input character matrix
        dropout_config: Dropout configuration
        cassette_config: Cassette configuration (for intbc information)
        number_of_cassettes: Number of integration barcodes (cassettes)

    Returns:
        Character matrix with dropout applied
    """
    matrix = character_matrix.copy()
    base_dropout_rate = dropout_config.get('rate', 0.15)
    intbc_variability = dropout_config.get('intbc_variability', 0.1)

    # Block-level dropout: entire intbc blocks are lost per cell
    # Use provided number_of_cassettes or fall back to cassette_config or default to 4
    num_intbc = number_of_cassettes or cassette_config.get('num_intbc') or cassette_config.get('number_of_cassettes', 4)
    # Different dropout rates per cell (simulating cell quality variation)
    cell_variability = dropout_config.get('cell_variability', 0.2)

    # Parameters
    n_cells, num_sites = matrix.shape

    # Gene-specific dropout rates (uniform prior with variability)
    intbc_dropout_rates = base_dropout_rate * (1 + np.random.uniform(-intbc_variability, intbc_variability, num_intbc))
    intbc_dropout_rates = np.clip(intbc_dropout_rates, 0.0, 1.0)

    cell_multipliers = np.array([])
    match cell_multiplier:
        case "higher":
            # Cell-specific multipliers (higher multiplier = worse quality cell)
            # all cells worse (multiplier ≥ 1)
            cell_multipliers = 1 + np.random.uniform(0, cell_variability, n_cells)
        case "lower":
            # some cells to be better than average 
            cell_multipliers = 1 + np.random.uniform(-cell_variability, cell_variability, n_cells)
        case "inverse":
            # Most cells are good quality
            cell_quality = np.random.beta(5, 2, n_cells)
            # Convert quality to dropout multiplier (inverse relationship)
            # Quality 1 → multiplier 1, Quality 0 → multiplier 2
            cell_multipliers = 2 - cell_quality  
    return cell_multipliers, intbc_dropout_rates

def apply_single_cell_dropout(
    character_matrix: pd.DataFrame,
    cell_multipliers,
    intbc_dropout_rates,
    sites_per_intbc,
) -> pd.DataFrame:
    # Final dropout probability for each cell-gene pair
    dropout_probs = intbc_dropout_rates[np.newaxis, :] * cell_multipliers[:, np.newaxis]
    dropout_probs = np.clip(dropout_probs, 0.0, 1.0)

    # Apply dropout
    dropout_mask = np.random.random((len(cell_multipliers), len(intbc_dropout_rates))) < dropout_probs
    dropout_mask = np.repeat(dropout_mask, sites_per_intbc, axis=1)

    masked_matrix = character_matrix.copy()
    masked_matrix[dropout_mask] = -1

    return masked_matrix, dropout_mask
