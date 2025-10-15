"""
LineageForest - TreeData extension for lineage tracing experiments.

This module provides a TreeData subclass optimized for single-cell and
single-molecule lineage tracing data from experimental datasets.
For simulation workflows, use SimulatedLineageForest from sim module.
"""

from __future__ import annotations

from typing import Literal, Any
from pathlib import Path
from collections.abc import Mapping, Sequence
import warnings

import networkx as nx
import pandas as pd
import numpy as np
from scipy import sparse
from anndata import AnnData
from treedata import TreeData
from cassiopeia.data import CassiopeiaTree

import logging

logger = logging.getLogger(__name__)


class LineageForest(TreeData):
    """
    TreeData extension for lineage tracing experiments.

    Extends TreeData with:
    - Character matrix management across layers
    - Cell/molecule observation types
    - Cassiopeia integration
    - Proper slicing/copying that preserves subclass type

    Attributes
    ----------
    obst : dict-like
        Observation trees (inherited from TreeData)
        Keys: 'sc' (single-cell tree), 'sm_<cell_id>' (single-molecule trees)
    layers : dict-like
        Character matrices (inherited from TreeData)
        Keys: 'character_matrix_sc', 'character_matrix_sm', etc.
    obs : DataFrame
        Observations with 'observation_type' ('cell' or 'molecule')
        and 'parent_cell' columns

    Examples
    --------
    >>> # From experimental data
    >>> ltdata = LineageForest.from_experiment(adata, sm_data)
    >>> ltdata.reconstruct_trees(solver='nj')
    """

    def __init__(
        self,
        # TreeData parameters with proper type hints
        X: np.ndarray | sparse.spmatrix | pd.DataFrame | None = None,
        obs: pd.DataFrame | Mapping[str, Any] | None = None,
        var: pd.DataFrame | Mapping[str, Any] | None = None,
        uns: Mapping[str, Any] | None = None,
        obsm: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        obst: Mapping[str, nx.DiGraph] | None = None,
        varm: np.ndarray | Mapping[str, Sequence[Any]] | None = None,
        vart: Mapping[str, nx.DiGraph] | None = None,
        layers: Mapping[str, np.ndarray | sparse.spmatrix] | None = None,
        raw: Mapping[str, Any] | None = None,
        dtype: np.dtype | type | str | None = None,
        shape: tuple[int, int] | None = None,
        filename: Path | str | None = None,
        filemode: Literal["r", "r+"] | None = None,
        asview: bool = False,
        label: str | None = "tree",
        alignment: Literal["leaves", "nodes", "subset"] = "leaves",
        allow_overlap: bool = True,
        # Lineage-specific parameters
        gt_tree: CassiopeiaTree | None = None,
        exp_tree: CassiopeiaTree | None = None,
        conf_gt: dict[str, Any] | None = None,
        conf_exp: dict[str, Any] | None = None,
        conf_dropout: dict[str, Any] | None = None,
        is_simulated: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LineageForest.

        Parameters
        ----------
        X
            Data matrix (#observations × #variables)
        obs
            Observation annotations
        var
            Variable annotations
        uns
            Unstructured annotations
        obsm
            Multi-dimensional observation annotations
        obst
            Observation trees (networkx DiGraphs)
        varm
            Multi-dimensional variable annotations
        vart
            Variable trees (networkx DiGraphs)
        layers
            Additional data layers
        raw
            Raw data
        dtype
            Data type for X
        shape
            Shape tuple when X is None
        filename
            Backing file path
        filemode
            File open mode
        asview
            Whether to initialize as view
        label
            Column name for tree keys in obs/var
        alignment
            Tree-observation alignment type
        allow_overlap
            Whether overlapping trees are allowed
        gt_tree
            Ground truth tree (simulations only)
        exp_tree
            Experimental tree with recording (simulations only)
        conf_gt
            Ground truth simulation configuration
        conf_exp
            Recording simulation configuration
        conf_dropout
            Dropout configuration
        is_simulated
            Whether this is simulated data
        **kwargs
            Additional arguments

        Notes
        -----
        When asview=True, X should be a LineageForest instance.
        Custom attributes default to None to avoid mutable default arguments.
        """
        # Store lineage attributes BEFORE calling super().__init__
        # because _init_as_view might access them
        self._gt_tree = gt_tree
        self._exp_tree = exp_tree
        self._conf_gt = conf_gt
        self._conf_exp = conf_exp
        self._conf_dropout = conf_dropout
        self._is_simulated = is_simulated

        # Initialize parent TreeData
        super().__init__(
            X=X,
            obs=obs,
            var=var,
            uns=uns,
            obsm=obsm,
            obst=obst,
            varm=varm,
            vart=vart,
            layers=layers,
            raw=raw,
            dtype=dtype,
            shape=shape,
            filename=filename,
            filemode=filemode,
            asview=asview,
            label=label,
            alignment=alignment,
            allow_overlap=allow_overlap,
            **kwargs,
        )

    # ═══════════════════════════════════════════════════════════
    # CRITICAL OVERRIDES - Preserve subclass type and attributes
    # ═══════════════════════════════════════════════════════════

    def _init_as_view(
        self, tdata_ref: TreeData, oidx: Any, vidx: Any
    ) -> None:
        """Override to preserve lineage attributes when creating views."""
        # Call parent
        super()._init_as_view(tdata_ref, oidx, vidx)

        # Copy lineage attributes from reference if it's a LineageForest
        if isinstance(tdata_ref, LineageForest):
            self._gt_tree = tdata_ref._gt_tree
            self._exp_tree = tdata_ref._exp_tree
            self._conf_gt = tdata_ref._conf_gt
            self._conf_exp = tdata_ref._conf_exp
            self._conf_dropout = tdata_ref._conf_dropout
            self._is_simulated = tdata_ref._is_simulated
        else:
            # Default values if creating view from plain TreeData
            self._gt_tree = None
            self._exp_tree = None
            self._conf_gt = None
            self._conf_exp = None
            self._conf_dropout = None
            self._is_simulated = False

    def __getitem__(self, index: Any) -> LineageForest:
        """Returns a sliced view, preserving LineageForest type."""
        oidx, vidx = self._normalize_indices(index)
        # Return LineageForest instead of TreeData!
        return LineageForest(self, oidx=oidx, vidx=vidx, asview=True)

    def _mutated_copy(self, **kwargs: Any) -> LineageForest:
        """Create copy with mutations, preserving LineageForest type."""
        # Get base TreeData dict
        if self.isbacked:
            if "X" not in kwargs or (self.raw is not None and "raw" not in kwargs):
                raise NotImplementedError(
                    "This function does not currently handle backed objects internally"
                )

        new = {}
        new["label"] = self.label
        new["allow_overlap"] = self.allow_overlap
        new["alignment"] = self.alignment

        # Copy TreeData attributes
        for key in ["obs", "var", "obsm", "varm", "obsp", "varp", "obst", "vart", "layers"]:
            if key in kwargs:
                new[key] = kwargs[key]
            else:
                new[key] = getattr(self, key).copy()

        if "X" in kwargs:
            new["X"] = kwargs["X"]
        elif self._has_X():
            new["X"] = self.X.copy()

        if "uns" in kwargs:
            new["uns"] = kwargs["uns"]
        else:
            from copy import deepcopy

            new["uns"] = deepcopy(self._uns)

        if "raw" in kwargs:
            new["raw"] = kwargs["raw"]
        elif self.raw is not None:
            new["raw"] = self.raw.copy()

        # Add lineage-specific attributes
        new["gt_tree"] = self._gt_tree
        new["exp_tree"] = self._exp_tree
        new["conf_gt"] = self._conf_gt
        new["conf_exp"] = self._conf_exp
        new["conf_dropout"] = self._conf_dropout
        new["is_simulated"] = self._is_simulated

        # Return LineageForest instead of TreeData!
        return LineageForest(**new)

    def copy(self, filename: Path | str | None = None) -> LineageForest:
        """Full copy, preserving LineageForest type."""
        if not self.isbacked:
            if self.is_view and self._has_X():
                from anndata._core.index import _subset

                return self._mutated_copy(
                    X=_subset(self._adata_ref.X, (self._oidx, self._vidx)).copy()
                )
            else:
                return self._mutated_copy()
        else:
            raise NotImplementedError("Backed mode not yet supported for LineageForest")

    def __repr__(self) -> str:
        """String representation with lineage-specific info."""
        # Get base repr
        base_repr = super().__repr__()

        # Add lineage-specific info
        sim_str = " [Simulated]" if self.is_simulated else " [Experimental]"
        extras = []

        if "observation_type" in self.obs.columns:
            extras.append(f"cells: {self.n_cells}")
            if self.n_molecules > 0:
                extras.append(f"molecules: {self.n_molecules}")

        # Replace first line
        lines = base_repr.split("\n")
        lines[0] = lines[0].replace("TreeData object", f"LineageForest{sim_str}")

        if extras:
            lines.append("    " + ", ".join(extras))

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # PROPERTIES - Simulation-specific
    # ═══════════════════════════════════════════════════════════

    @property
    def is_simulated(self) -> bool:
        """Whether this is simulated data."""
        return self._is_simulated

    @property
    def gt_tree(self) -> CassiopeiaTree | None:
        """Ground truth tree (simulations only)."""
        return self._gt_tree

    @gt_tree.setter
    def gt_tree(self, tree: CassiopeiaTree | None) -> None:
        self._gt_tree = tree

    @property
    def exp_tree(self) -> CassiopeiaTree | None:
        """Experimental tree with recording (simulations only)."""
        return self._exp_tree

    @exp_tree.setter
    def exp_tree(self, tree: CassiopeiaTree | None) -> None:
        self._exp_tree = tree

    @property
    def conf_gt(self) -> dict[str, Any] | None:
        """Ground truth simulation configuration."""
        return self._conf_gt

    @property
    def conf_exp(self) -> dict[str, Any] | None:
        """Recording simulation configuration."""
        return self._conf_exp

    @property
    def conf_dropout(self) -> dict[str, Any] | None:
        """Dropout configuration."""
        return self._conf_dropout

    # ═══════════════════════════════════════════════════════════
    # PROPERTIES - Data accessors
    # ═══════════════════════════════════════════════════════════

    @property
    def n_cells(self) -> int:
        """Number of cell-level observations."""
        if "observation_type" in self.obs.columns:
            return (self.obs["observation_type"] == "cell").sum()
        return self.n_obs

    @property
    def n_molecules(self) -> int:
        """Number of molecule-level observations."""
        if "observation_type" in self.obs.columns:
            return (self.obs["observation_type"] == "molecule").sum()
        return 0

    @property
    def cell_ids(self) -> pd.Index:
        """Cell IDs (cell-level observations)."""
        if "observation_type" in self.obs.columns:
            return self.obs[self.obs["observation_type"] == "cell"].index
        return self.obs.index

    @property
    def sm_cell_ids(self) -> list[str]:
        """Cell IDs with single-molecule data."""
        if "parent_cell" not in self.obs.columns:
            return []
        mol_mask = self.obs["observation_type"] == "molecule"
        return self.obs.loc[mol_mask, "parent_cell"].unique().tolist()

    @property
    def sc_tree_key(self) -> str:
        """Key for single-cell tree in obst."""
        return "sc"

    def sm_tree_key(self, cell_id: str) -> str:
        """Key for single-molecule tree in obst."""
        return f"sm_{cell_id}"

    # ═══════════════════════════════════════════════════════════
    # CLASS METHODS - Construction
    # ═══════════════════════════════════════════════════════════


    @classmethod
    def from_experiment(
        cls,
        adata: AnnData,
        sm_data: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> LineageForest:
        """
        Create LineageForest from experimental data.

        Parameters
        ----------
        adata
            AnnData with single-cell data
        sm_data
            Dict of cell_id -> character matrix for single-molecule data
        **kwargs
            Additional TreeData parameters

        Returns
        -------
        LineageForest with experimental data loaded
        """
        # Convert AnnData to TreeData format
        obj = cls(
            X=adata.X,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy() if adata.uns else None,
            obsm=adata.obsm.copy() if adata.obsm else None,
            layers=adata.layers.copy() if adata.layers else None,
            is_simulated=False,
            **kwargs,
        )

        # Mark as cell observations
        obj.obs["observation_type"] = "cell"
        obj.obs["parent_cell"] = obj.obs.index

        # Add molecule observations if provided
        if sm_data:
            obj._add_molecule_observations(sm_data)

        return obj


    def _add_molecule_observations(self, sm_data: dict[str, pd.DataFrame]) -> None:
        """Add molecule-level observations to existing TreeData."""
        mol_obs_list = []

        for cell_id, char_matrix in sm_data.items():
            mol_df = pd.DataFrame(index=char_matrix.index)
            mol_df["observation_type"] = "molecule"
            mol_df["parent_cell"] = cell_id
            mol_obs_list.append(mol_df)

        if mol_obs_list:
            import treedata as td

            mol_obs = pd.concat(mol_obs_list)
            mol_tdata = TreeData(obs=mol_obs)

            # Concatenate with existing data
            combined = td.concat([self, mol_tdata], axis=0)

            # Update self
            self._init_as_actual(
                X=combined.X if combined._has_X() else None,
                obs=combined.obs,
                var=combined.var if hasattr(combined, "var") else None,
                layers=dict(combined.layers) if hasattr(combined, "layers") else None,
            )

    # ═══════════════════════════════════════════════════════════
    # TREE RECONSTRUCTION
    # ═══════════════════════════════════════════════════════════

    def reconstruct_trees(
        self,
        solver: Any = None,
        fraction: Literal["all", "sc", "sm"] = "all",
        collapse_mutationless_edges: bool = True,
    ) -> None:
        """
        Reconstruct phylogenetic trees from character matrices.

        Stores trees in .obst as networkx DiGraphs.

        Parameters
        ----------
        solver
            Cassiopeia solver instance or string
        fraction
            Which fraction to reconstruct ('all', 'sc', 'sm')
        collapse_mutationless_edges
            Whether to collapse mutationless edges after reconstruction
        """
        from . import solvers

        if solver is None:
            solver = solvers.get_solver_class("nj")
        elif isinstance(solver, str):
            solver = solvers.get_solver_class(solver)

        # Reconstruct SC tree
        if fraction in ["all", "sc"]:
            if "character_matrix_sc" not in self.layers:
                raise ValueError("No single-cell character matrix found")

            sc_char_matrix = self._get_character_matrix("sc")

            cas_tree = CassiopeiaTree(
                character_matrix=sc_char_matrix, missing_state_indicator=-1
            )
            solver.solve(cas_tree, collapse_mutationless_edges=collapse_mutationless_edges)

            # Store in obst
            self.obst[self.sc_tree_key] = cas_tree.get_tree_topology()
            logger.info(f"Reconstructed SC tree ({len(cas_tree.leaves)} leaves)")

        # Reconstruct SM trees
        if fraction in ["all", "sm"]:
            if "character_matrix_sm" not in self.layers:
                raise ValueError("No single-molecule character matrix found")

            for cell_id in self.sm_cell_ids:
                sm_char_matrix = self._get_character_matrix("sm", cell_id=cell_id)

                cas_tree = CassiopeiaTree(
                    character_matrix=sm_char_matrix, missing_state_indicator=-1
                )
                solver.solve(cas_tree, collapse_mutationless_edges=False)

                # Store in obst
                self.obst[self.sm_tree_key(cell_id)] = cas_tree.get_tree_topology()
                logger.info(
                    f"Reconstructed SM tree for {cell_id} ({len(cas_tree.leaves)} leaves)"
                )

    def _get_character_matrix(
        self,
        fraction: Literal["sc", "sm"],
        cell_id: str | None = None,
    ) -> pd.DataFrame:
        """Get character matrix for a specific fraction."""
        if "observation_type" not in self.obs.columns:
            raise ValueError("observation_type column not found in obs")

        if fraction == "sc":
            layer_key = "character_matrix_sc"
            mask = self.obs["observation_type"] == "cell"
            obs_indices = self.obs[mask].index
        elif fraction == "sm":
            if cell_id is None:
                raise ValueError("Must provide cell_id for SM fraction")
            layer_key = "character_matrix_sm"
            mask = (self.obs["observation_type"] == "molecule") & (
                self.obs["parent_cell"] == cell_id
            )
            obs_indices = self.obs[mask].index
        else:
            raise ValueError(f"Unknown fraction: {fraction}")

        char_matrix = self.layers[layer_key].loc[obs_indices]
        return char_matrix.dropna(how="all")

    # ═══════════════════════════════════════════════════════════
    # TREE ACCESSORS
    # ═══════════════════════════════════════════════════════════

    def get_sc_tree(
        self, as_cassiopeia: bool = False
    ) -> nx.DiGraph | CassiopeiaTree | None:
        """
        Get single-cell tree.

        Parameters
        ----------
        as_cassiopeia
            If True, return CassiopeiaTree; if False, return networkx DiGraph

        Returns
        -------
        Tree in requested format, or None if not reconstructed yet
        """
        if self.sc_tree_key not in self.obst:
            return None

        nx_tree = self.obst[self.sc_tree_key]

        if not as_cassiopeia:
            return nx_tree

        # Convert to CassiopeiaTree
        char_matrix = self._get_character_matrix("sc")
        cas_tree = CassiopeiaTree(
            character_matrix=char_matrix, missing_state_indicator=-1
        )
        cas_tree.populate_tree(nx_tree)
        return cas_tree

    def get_sm_tree(
        self,
        cell_id: str,
        as_cassiopeia: bool = False,
    ) -> nx.DiGraph | CassiopeiaTree | None:
        """Get single-molecule tree for specific cell."""
        tree_key = self.sm_tree_key(cell_id)
        if tree_key not in self.obst:
            return None

        nx_tree = self.obst[tree_key]

        if not as_cassiopeia:
            return nx_tree

        # Convert to CassiopeiaTree
        char_matrix = self._get_character_matrix("sm", cell_id=cell_id)
        cas_tree = CassiopeiaTree(
            character_matrix=char_matrix, missing_state_indicator=-1
        )
        cas_tree.populate_tree(nx_tree)
        return cas_tree

    def get_all_sm_trees(
        self,
        as_cassiopeia: bool = False,
    ) -> dict[str, nx.DiGraph | CassiopeiaTree | None]:
        """Get all single-molecule trees."""
        return {
            cell_id: self.get_sm_tree(cell_id, as_cassiopeia=as_cassiopeia)
            for cell_id in self.sm_cell_ids
        }

    # ═══════════════════════════════════════════════════════════
    # PYCEA INTEGRATION
    # ═══════════════════════════════════════════════════════════

    def plot_tree(self, tree: str = "sc", **kwargs: Any) -> Any:
        """
        Plot tree using pycea.

        Parameters
        ----------
        tree
            Tree key in .obst (e.g., 'sc', 'sm_cell_001')
        **kwargs
            Additional arguments for pc.pl.plot_tree

        Returns
        -------
        Plot object from pycea
        """
        import pycea as pc

        return pc.pl.plot_tree(self, tree=tree, **kwargs)

    def add_tree_metrics(self, tree: str = "sc") -> None:
        """Add tree metrics using pycea."""
        import pycea as pc

        pc.pp.add_depth(self, tree=tree)

    # ═══════════════════════════════════════════════════════════
    # I/O
    # ═══════════════════════════════════════════════════════════

    def save(self, filepath: Path | str) -> None:
        """
        Save complete LineageForest object.

        Parameters
        ----------
        filepath
            Path to save the pickled object

        Notes
        -----
        This saves the entire object including trees and configs using pickle.
        For TreeData-native formats, use `.write_h5td()` or `.write_zarr()`.
        """
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Path | str) -> LineageForest:
        """
        Load LineageForest from pickle file.

        Parameters
        ----------
        filepath
            Path to the pickled object

        Returns
        -------
        The loaded LineageForest object
        """
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)


