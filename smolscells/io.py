"""Import routines for loading experimental data into LineageForest.

Bridges fracture/ogtk pipeline outputs (allele tables) to smolscells LineageForest
objects via cassiopeia's allele-to-character-matrix conversion.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cassiopeia as cas
import pandas as pd
import polars as pl
from cassiopeia.data import CassiopeiaTree

from .lineage_forest import LineageForest

logger = logging.getLogger(__name__)


def read_file(path: Path | str) -> pl.DataFrame:
    """Read an allele table from parquet or arrow format."""
    path_str = str(path)
    if path_str.endswith(".arrow"):
        return pl.read_ipc(path_str)
    return pl.read_parquet(path_str)


def collapse_umis_to_cells(
    allele_df: pl.DataFrame,
    cell_col: str = "cellBC",
    intbc_col: str = "intBC",
    min_umis_per_cell: int = 4,
    min_umi_agreement: float = 0.5,
) -> pl.DataFrame:
    """Collapse multiple UMIs per cell into one consensus allele per (cell, intBC).

    For single-cell lineage tracing, each cell may have multiple UMIs for the same
    integration barcode. This function collapses them to one consensus row per
    (cell, intBC) using mode-based voting.

    Parameters
    ----------
    allele_df
        DataFrame with cellBC, intBC, r1, r2, ..., readCount columns.
    cell_col
        Column name for cell barcode.
    intbc_col
        Column name for integration barcode.
    min_umis_per_cell
        Minimum UMIs required for valid consensus. Groups with fewer UMIs
        are filtered out.
    min_umi_agreement
        Minimum fraction for consensus. If the mode has support below this
        threshold, mark as None (missing).

    Returns
    -------
    pl.DataFrame
        One row per (cell, intBC), with an added ``n_umis`` column.
        Cells with fewer than *min_umis_per_cell* UMIs are excluded.
    """
    if allele_df.height == 0:
        return allele_df.with_columns(pl.lit(0).alias("n_umis"))

    # Identify r columns (allele columns)
    rcols = allele_df.select(pl.col("^r\\d+$")).columns
    if not rcols:
        logger.warning("No r columns found in allele_df, returning as-is with n_umis=1")
        return allele_df.with_columns(pl.lit(1).alias("n_umis"))

    # Columns to preserve (take first value per group)
    preserve_cols = [c for c in allele_df.columns if c not in rcols + [cell_col, intbc_col]]

    # Add row index for tracking
    df = allele_df.with_row_index("_row_idx")

    # Step 1: Get group sizes and filter by min_umis_per_cell
    group_sizes = df.group_by([cell_col, intbc_col]).agg(pl.len().alias("n_umis"))

    n_groups_before = group_sizes.height
    valid_groups = group_sizes.filter(pl.col("n_umis") >= min_umis_per_cell)
    n_groups_after = valid_groups.height

    if n_groups_before != n_groups_after:
        logger.info(
            f"UMI collapse: filtered {n_groups_before - n_groups_after} (cell, intBC) groups "
            f"with < {min_umis_per_cell} UMIs ({n_groups_after} remaining)"
        )

    if valid_groups.height == 0:
        empty_cols = [cell_col, intbc_col, "n_umis"] + rcols + preserve_cols
        return pl.DataFrame(
            schema={
                c: allele_df.schema.get(c, pl.Utf8)
                for c in empty_cols
                if c in allele_df.columns or c == "n_umis"
            }
        )

    # Filter to only valid groups
    df = df.join(valid_groups.select([cell_col, intbc_col]), on=[cell_col, intbc_col], how="semi")

    # Step 2: For each r column, compute mode and support
    id_cols = [cell_col, intbc_col, "_row_idx"] + preserve_cols
    unpivoted = df.unpivot(
        on=rcols,
        index=id_cols,
        variable_name="_rcol",
        value_name="_allele",
    )

    consensus = (
        unpivoted.group_by([cell_col, intbc_col, "_rcol"])
        .agg(
            [
                pl.len().alias("_n_total"),
                pl.col("_allele").value_counts(sort=True).first().alias("_mode_struct"),
            ]
        )
        .with_columns(
            [
                pl.col("_mode_struct").struct.field("_allele").alias("_mode_value"),
                pl.col("_mode_struct").struct.field("count").alias("_mode_count"),
            ]
        )
        .with_columns((pl.col("_mode_count") / pl.col("_n_total")).alias("_support"))
        .with_columns(
            pl.when(pl.col("_support") >= min_umi_agreement)
            .then(pl.col("_mode_value"))
            .otherwise(pl.lit(None))
            .alias("_consensus_value")
        )
        .select([cell_col, intbc_col, "_rcol", "_consensus_value"])
    )

    # Pivot back to wide format
    result = consensus.pivot(
        on="_rcol",
        index=[cell_col, intbc_col],
        values="_consensus_value",
    )

    # Add n_umis column
    result = result.join(valid_groups, on=[cell_col, intbc_col], how="left")

    # Add preserved columns (take first value per group)
    if preserve_cols:
        preserved = df.group_by([cell_col, intbc_col]).agg(
            [pl.col(c).first().alias(c) for c in preserve_cols]
        )
        result = result.join(preserved, on=[cell_col, intbc_col], how="left")

    # Reorder columns to match expected output
    output_cols = [cell_col, intbc_col, "n_umis"] + rcols + preserve_cols
    output_cols = [c for c in output_cols if c in result.columns]
    result = result.select(output_cols)

    logger.info(f"UMI collapse: {allele_df.height} rows -> {result.height} (cell, intBC) pairs")

    return result


def _allele_df_to_character_matrix(
    allele_df: pl.DataFrame,
    allele_rep_thresh: float = 1.0,
) -> tuple[pd.DataFrame, dict | None, dict | None]:
    """Convert a polars allele DataFrame to an integer character matrix.

    Thin wrapper: polars → pandas → cas.pp.convert_alleletable_to_character_matrix.

    Parameters
    ----------
    allele_df
        Allele table as polars DataFrame.
    allele_rep_thresh
        Threshold for allele representation passed to cassiopeia.

    Returns
    -------
    tuple of (character_matrix, priors, state_2_indel)
        character_matrix: pandas DataFrame with integer states
        priors: mutation prior dict or None
        state_2_indel: state-to-indel mapping dict or None
    """
    allele_pd = allele_df.to_pandas()
    character_matrix, priors, state_2_indel = cas.pp.convert_alleletable_to_character_matrix(
        allele_pd,
        allele_rep_thresh=allele_rep_thresh,
    )
    return character_matrix, priors, state_2_indel


def from_trees(
    sc_tree: CassiopeiaTree | None = None,
    sm_trees: dict[str | int, CassiopeiaTree] | list[CassiopeiaTree] | None = None,
) -> LineageForest:
    """Create a LineageForest from pre-existing CassiopeiaTree objects.

    Parameters
    ----------
    sc_tree
        Single-cell CassiopeiaTree. Stored as 'sc'.
    sm_trees
        Single-molecule trees. If dict, keys become tree IDs (e.g. 'sm_{key}').
        If list, trees are auto-indexed (sm_0, sm_1, ...).

    Returns
    -------
    LineageForest
    """
    lf = LineageForest()

    if sc_tree is not None:
        lf.add_sc_tree(sc_tree)

    if sm_trees is not None:
        if isinstance(sm_trees, dict):
            for intbc_id, tree in sm_trees.items():
                lf.add_sm_tree_with_id(intbc_id, tree)
        else:
            for tree in sm_trees:
                lf.add_sm_tree(tree)

    return lf


def from_allele_table(
    path: Path | str,
    modality: str = "single-molecule",
    cell_col: str = "cellBC",
    intbc_col: str = "intBC",
    collapse: bool = True,
    min_umis_per_cell: int = 4,
    min_umi_agreement: float = 0.5,
    allele_rep_thresh: float = 1.0,
    intbc_whitelist: list[str] | None = None,
) -> LineageForest:
    """Import an allele table into a LineageForest.

    Reads a parquet/arrow allele table (from fracture) and converts it to
    CassiopeiaTree objects via cassiopeia's character matrix conversion.

    Parameters
    ----------
    path
        Path to allele table file (parquet or arrow).
    modality
        'single-molecule' or 'single-cell'.
    cell_col
        Column name for cell barcodes (SC only).
    intbc_col
        Column name for integration barcodes.
    collapse
        Whether to collapse UMIs to cells (SC only). Set False if already collapsed.
    min_umis_per_cell
        Minimum UMIs per cell for collapse consensus.
    min_umi_agreement
        Minimum fraction for collapse consensus.
    allele_rep_thresh
        Threshold for allele representation in character matrix conversion.
    intbc_whitelist
        If provided, only include these intBCs.

    Returns
    -------
    LineageForest
    """
    allele_df = read_file(path)
    logger.info(f"Read allele table: {allele_df.shape[0]} rows, {allele_df.shape[1]} columns")

    if intbc_whitelist is not None:
        allele_df = allele_df.filter(pl.col(intbc_col).is_in(intbc_whitelist))
        logger.info(f"Filtered to {allele_df.shape[0]} rows with {len(intbc_whitelist)} intBCs")

    lf = LineageForest()

    if modality == "single-cell":
        lf = _import_single_cell(
            allele_df,
            lf,
            collapse=collapse,
            cell_col=cell_col,
            intbc_col=intbc_col,
            min_umis_per_cell=min_umis_per_cell,
            min_umi_agreement=min_umi_agreement,
            allele_rep_thresh=allele_rep_thresh,
        )
    elif modality == "single-molecule":
        lf = _import_single_molecule(
            allele_df,
            lf,
            intbc_col=intbc_col,
            allele_rep_thresh=allele_rep_thresh,
        )
    else:
        raise ValueError(f"Unknown modality: {modality!r}. Use 'single-cell' or 'single-molecule'.")

    return lf


def _import_single_cell(
    allele_df: pl.DataFrame,
    lf: LineageForest,
    *,
    collapse: bool,
    cell_col: str,
    intbc_col: str,
    min_umis_per_cell: int,
    min_umi_agreement: float,
    allele_rep_thresh: float,
) -> LineageForest:
    """Import single-cell allele table into LineageForest."""
    if collapse:
        logger.info("Collapsing UMIs to cells...")
        allele_df = collapse_umis_to_cells(
            allele_df,
            cell_col=cell_col,
            intbc_col=intbc_col,
            min_umis_per_cell=min_umis_per_cell,
            min_umi_agreement=min_umi_agreement,
        )
        logger.info(f"After collapse: {allele_df.shape[0]} rows")

    character_matrix, priors, state_2_indel = _allele_df_to_character_matrix(
        allele_df,
        allele_rep_thresh=allele_rep_thresh,
    )
    logger.info(f"SC character matrix: {character_matrix.shape}")

    tree = CassiopeiaTree(character_matrix=character_matrix)
    lf.add_sc_tree(tree)

    lf.uns["sc_priors"] = priors
    lf.uns["sc_state_2_indel"] = state_2_indel

    return lf


def _import_single_molecule(
    allele_df: pl.DataFrame,
    lf: LineageForest,
    *,
    intbc_col: str,
    allele_rep_thresh: float,
) -> LineageForest:
    """Import single-molecule allele table into LineageForest, partitioned by intBC."""
    intbcs = allele_df[intbc_col].unique().sort().to_list()
    logger.info(f"Found {len(intbcs)} intBCs")

    priors_dict = {}
    state_2_indel_dict = {}

    for intbc in intbcs:
        subset = allele_df.filter(pl.col(intbc_col) == intbc)
        if subset.shape[0] == 0:
            continue

        try:
            character_matrix, priors, state_2_indel = _allele_df_to_character_matrix(
                subset,
                allele_rep_thresh=allele_rep_thresh,
            )
        except Exception as e:
            logger.warning(f"Skipping intBC {intbc}: {e}")
            continue

        if character_matrix.shape[0] == 0:
            logger.warning(f"Skipping intBC {intbc}: empty character matrix")
            continue

        tree = CassiopeiaTree(character_matrix=character_matrix)
        lf.add_sm_tree_with_id(intbc, tree)

        priors_dict[intbc] = priors
        state_2_indel_dict[intbc] = state_2_indel

    lf.uns["sm_priors"] = priors_dict
    lf.uns["sm_state_2_indel"] = state_2_indel_dict

    logger.info(f"Imported {len(lf.smtrees_keys)} SM trees")
    return lf


def from_allele_tables(
    sc_path: Path | str | None = None,
    sm_path: Path | str | None = None,
    **kwargs,
) -> LineageForest:
    """Create a LineageForest from separate SC and SM allele table files.

    Convenience function that calls from_allele_table for each modality
    and merges results into a single LineageForest.

    Parameters
    ----------
    sc_path
        Path to single-cell allele table.
    sm_path
        Path to single-molecule allele table.
    **kwargs
        Additional arguments passed to from_allele_table (e.g. allele_rep_thresh,
        collapse, min_umis_per_cell, etc.).

    Returns
    -------
    LineageForest
    """
    if sc_path is None and sm_path is None:
        raise ValueError("At least one of sc_path or sm_path must be provided.")

    lf = LineageForest()

    if sm_path is not None:
        sm_lf = from_allele_table(sm_path, modality="single-molecule", **kwargs)
        # Transfer SM trees and metadata
        for key, tree in sm_lf.trees.items():
            lf.trees[key] = tree
        lf._sm_counter = sm_lf._sm_counter
        lf.uns.update(sm_lf.uns)

    if sc_path is not None:
        sc_lf = from_allele_table(sc_path, modality="single-cell", **kwargs)
        # Transfer SC tree and metadata
        if "sc" in sc_lf.trees:
            lf.trees["sc"] = sc_lf.trees["sc"]
        lf.uns.update(sc_lf.uns)

    return lf
