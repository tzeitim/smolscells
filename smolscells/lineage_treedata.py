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
        Keys: 'sc' (single-cell tree), '0', '1', '2', ... (single-molecule trees per intBC)
    layers : dict-like
        Character matrices (inherited from TreeData)
        Keys: 'character_matrix_sc', 'character_matrix_sm', etc.
    obs : DataFrame
        Observations with 'observation_type' ('cell' or 'molecule'),
        'parent_cell', and 'intbc_id' (int) columns

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
    def intbc_ids(self) -> list[int]:
        """Integration barcode IDs present in single-molecule data."""
        if "character_matrices" not in self.uns:
            return []
        # Extract intbc IDs from character matrix keys (e.g., 'character_matrix_sm_0' -> 0)
        ids = []
        for key in self.uns["character_matrices"].keys():
            if key.startswith("character_matrix_sm_"):
                intbc_id = int(key.replace("character_matrix_sm_", ""))
                ids.append(intbc_id)
        return sorted(ids)

    @property
    def sc_tree_key(self) -> str:
        """Key for single-cell tree in obst."""
        return "sc"

    def sm_tree_key(self, intbc_id: int | str) -> str:
        """
        Key for single-molecule tree in obst.

        Parameters
        ----------
        intbc_id : int or str
            Integration barcode ID (e.g., 0, 1, '0', '1')

        Returns
        -------
        str
            Tree key as string (e.g., '0', '1')
        """
        return str(intbc_id)

    # ═══════════════════════════════════════════════════════════
    # CLASS METHODS - Construction
    # ═══════════════════════════════════════════════════════════


    @classmethod
    def from_experiment(
        cls,
        adata: AnnData,
        sm_data: dict[int, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> LineageForest:
        """
        Create LineageForest from experimental data.

        Parameters
        ----------
        adata
            AnnData with single-cell data
        sm_data
            Dict of intbc_id (int) -> character matrix for single-molecule data
            Keys should be integration barcode indices (e.g., 0, 1, 2)
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

    @classmethod
    def from_cassiopeia_tree(
        cls,
        cas_tree: CassiopeiaTree,
        tree_type: Literal["sc", "sm"] = "sc",
        intbc_id: int = 0,
        tree_key: str | None = None,
        alignment: Literal["leaves", "nodes", "subset"] = "subset",
        **kwargs: Any,
    ) -> LineageForest:
        """
        Create LineageForest from a single CassiopeiaTree.

        Parameters
        ----------
        cas_tree
            CassiopeiaTree with character matrix and topology
        tree_type
            Type of tree: 'sc' for single-cell, 'sm' for single-molecule
        intbc_id
            Integration barcode ID (used only if tree_type='sm')
        tree_key
            Custom tree key in obst (if None, uses 'sc' or str(intbc_id))
        alignment
            Tree-observation alignment type
        **kwargs
            Additional TreeData parameters

        Returns
        -------
        LineageForest with single tree loaded

        Examples
        --------
        >>> # Create from single-cell tree
        >>> lf = LineageForest.from_cassiopeia_tree(sc_tree, tree_type='sc')
        >>>
        >>> # Create from single-molecule tree
        >>> lf = LineageForest.from_cassiopeia_tree(sm_tree, tree_type='sm', intbc_id=0)
        """
        # Extract character matrix and cell IDs
        char_matrix = cas_tree.character_matrix.copy()
        cell_ids = char_matrix.index.tolist()

        # Create obs DataFrame
        obs = pd.DataFrame(index=cell_ids)

        if tree_type == "sc":
            obs["observation_type"] = "cell"
            obs["parent_cell"] = obs.index
            obs["intbc_id"] = pd.NA  # SC observations don't have intbc_id
            char_matrix_key = "character_matrix_sc"
            default_tree_key = "sc"
        elif tree_type == "sm":
            obs["observation_type"] = "molecule"
            obs["parent_cell"] = ""  # Unknown for imported SM data
            obs["intbc_id"] = int(intbc_id)
            char_matrix_key = f"character_matrix_sm_{intbc_id}"
            default_tree_key = str(intbc_id)
        else:
            raise ValueError(f"Invalid tree_type: {tree_type}. Must be 'sc' or 'sm'")
        
        # Use custom tree_key or default
        final_tree_key = tree_key if tree_key is not None else default_tree_key

        # Create minimal LineageForest - just store tree and character matrix
        # TODO: implement storage for obs and for now 
        # don't populate obs to avoid AnnData shape constraints
        lf = cls(
            uns={"character_matrices": {char_matrix_key: char_matrix}},
            alignment=alignment,
            **kwargs,
        )

        # Add tree topology if available
        try:
            topology = cas_tree.get_tree_topology()
            if topology is not None:
                lf.obst[final_tree_key] = topology
                lf.add_tree_metrics(final_tree_key)
                logger.info(
                    f"Created LineageForest from CassiopeiaTree: "
                    f"{tree_type} tree '{final_tree_key}' with {len(cell_ids)} cells"
                )
        except Exception:
            # Tree has not been initialized
            logger.warning(
                f"CassiopeiaTree has no topology. Tree '{final_tree_key}' not added to obst."
            )

        return lf

    def _add_molecule_observations(self, sm_data: dict[int, pd.DataFrame]) -> None:
        """
        Add molecule-level observations to existing TreeData.

        Parameters
        ----------
        sm_data : dict
            Dict of intbc_id (int) -> character matrix DataFrame
        """
        mol_obs_list = []

        for intbc_id, char_matrix in sm_data.items():
            mol_df = pd.DataFrame(index=char_matrix.index)
            mol_df["observation_type"] = "molecule"
            mol_df["intbc_id"] = int(intbc_id)  # Ensure integer
            # Note: parent_cell should be set if known, otherwise leave empty
            mol_df["parent_cell"] = ""
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

    def add_cassiopeia_tree(
        self,
        cas_tree: CassiopeiaTree,
        tree_type: Literal["sc", "sm"] = "sc",
        intbc_id: int | None = None,
        tree_key: str | None = None,
    ) -> None:
        """
        Add a CassiopeiaTree to existing LineageForest.

        Appends observations, character matrix, and tree topology to the
        existing LineageForest. Useful for incrementally building a forest.

        Parameters
        ----------
        cas_tree
            CassiopeiaTree with character matrix and topology
        tree_type
            Type of tree: 'sc' for single-cell, 'sm' for single-molecule
        intbc_id
            Integration barcode ID (if None for SM, auto-increments from max)
        tree_key
            Custom tree key in obst (if None, uses 'sc' or str(intbc_id))

        Examples
        --------
        >>> # Create initial LineageForest
        >>> lf = LineageForest.from_cassiopeia_tree(sc_tree, tree_type='sc')
        >>>
        >>> # Add single-molecule trees
        >>> lf.add_cassiopeia_tree(sm_tree0, tree_type='sm', intbc_id=0)
        >>> lf.add_cassiopeia_tree(sm_tree1, tree_type='sm', intbc_id=1)
        """
        # Extract character matrix
        char_matrix = cas_tree.character_matrix.copy()
        cell_ids = char_matrix.index.tolist()

        # Create obs DataFrame for new observations
        new_obs = pd.DataFrame(index=cell_ids)

        if tree_type == "sc":
            new_obs["observation_type"] = "cell"
            new_obs["parent_cell"] = new_obs.index
            new_obs["intbc_id"] = pd.NA
            char_matrix_key = "character_matrix_sc"
            default_tree_key = "sc"

            # Check if SC tree already exists
            if "sc" in self.obst:
                logger.warning("SC tree already exists. Replacing with new tree.")

        elif tree_type == "sm":
            # Auto-increment intbc_id if not provided
            if intbc_id is None:
                existing_ids = self.intbc_ids
                intbc_id = max(existing_ids) + 1 if existing_ids else 0
                logger.info(f"Auto-assigned intbc_id={intbc_id}")

            new_obs["observation_type"] = "molecule"
            new_obs["parent_cell"] = ""  # Unknown for imported SM data
            new_obs["intbc_id"] = int(intbc_id)
            char_matrix_key = f"character_matrix_sm_{intbc_id}"
            default_tree_key = str(intbc_id)

            # Check if this intbc_id already exists
            if intbc_id in self.intbc_ids:
                logger.warning(f"intBC {intbc_id} already exists. Replacing with new tree.")

        else:
            raise ValueError(f"Invalid tree_type: {tree_type}. Must be 'sc' or 'sm'")

        # Use custom tree_key or default
        final_tree_key = tree_key if tree_key is not None else default_tree_key

        # Store character matrix in uns (not layers, due to dimension constraints)
        if "character_matrices" not in self.uns:
            self.uns["character_matrices"] = {}
        self.uns["character_matrices"][char_matrix_key] = char_matrix

        # Add tree topology if available
        try:
            topology = cas_tree.get_tree_topology()
            if topology is not None:
                self.obst[final_tree_key] = topology
                self.add_tree_metrics(final_tree_key)
                logger.info(
                    f"Added {tree_type} tree '{final_tree_key}' with {len(cell_ids)} cells"
                )
        except Exception:
            # Tree has not been initialized
            logger.warning(
                f"CassiopeiaTree has no topology. Tree '{final_tree_key}' not added to obst."
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
            try:
                sc_char_matrix = self._get_character_matrix("sc")
            except ValueError as e:
                raise ValueError(f"No single-cell character matrix found: {e}")

            cas_tree = CassiopeiaTree(
                character_matrix=sc_char_matrix, missing_state_indicator=-1
            )
            solver.solve(cas_tree, collapse_mutationless_edges=collapse_mutationless_edges)

            # Store in obst
            self.obst[self.sc_tree_key] = cas_tree.get_tree_topology()
            logger.info(f"Reconstructed SC tree ({len(cas_tree.leaves)} leaves)")

        # Reconstruct SM trees
        if fraction in ["all", "sm"]:
            if not self.intbc_ids:
                raise ValueError("No single-molecule data found")

            for intbc_id in self.intbc_ids:
                sm_char_matrix = self._get_character_matrix("sm", intbc_id=intbc_id)

                cas_tree = CassiopeiaTree(
                    character_matrix=sm_char_matrix, missing_state_indicator=-1
                )
                solver.solve(cas_tree, collapse_mutationless_edges=False)

                # Store in obst
                self.obst[self.sm_tree_key(intbc_id)] = cas_tree.get_tree_topology()
                logger.info(
                    f"Reconstructed SM tree for {intbc_id} ({len(cas_tree.leaves)} leaves)"
                )

    def _get_character_matrix(
        self,
        fraction: Literal["sc", "sm"],
        intbc_id: int | str | None = None,
    ) -> pd.DataFrame:
        """
        Get character matrix for a specific fraction.

        Parameters
        ----------
        fraction : {'sc', 'sm'}
            Fraction type: 'sc' for single-cell, 'sm' for single-molecule
        intbc_id : int or str, optional
            Integration barcode ID (required for SM fraction)

        Returns
        -------
        pd.DataFrame
            Character matrix for the specified fraction
        """
        # Character matrices are stored in uns, not layers (due to dimension constraints)
        if "character_matrices" not in self.uns:
            raise ValueError("No character matrices found in uns")

        if fraction == "sc":
            char_matrix_key = "character_matrix_sc"
        elif fraction == "sm":
            if intbc_id is None:
                raise ValueError("Must provide intbc_id for SM fraction")
            # Convert to int for key generation
            intbc_id_int = int(intbc_id)
            char_matrix_key = f"character_matrix_sm_{intbc_id_int}"
        else:
            raise ValueError(f"Unknown fraction: {fraction}")

        # Retrieve character matrix from uns
        if char_matrix_key not in self.uns["character_matrices"]:
            raise ValueError(f"Character matrix '{char_matrix_key}' not found")

        char_matrix = self.uns["character_matrices"][char_matrix_key]
        return char_matrix.copy()

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
        # Note: populate_tree must be called BEFORE setting character matrix
        # because populate_tree calls set_character_states_at_leaves which
        # validates that character matrix indices match tree leaves
        char_matrix = self._get_character_matrix("sc")
        cas_tree = CassiopeiaTree(
            missing_state_indicator=-1
        )
        cas_tree.populate_tree(nx_tree)
        cas_tree.character_matrix = char_matrix
        return cas_tree

    def get_sm_tree(
        self,
        intbc_id: int | str,
        as_cassiopeia: bool = False,
    ) -> nx.DiGraph | CassiopeiaTree | None:
        """
        Get single-molecule tree for specific integration barcode.

        Parameters
        ----------
        intbc_id : int or str
            Integration barcode ID (e.g., 0, 1, '0', '1')
        as_cassiopeia : bool
            If True, return CassiopeiaTree; if False, return networkx DiGraph

        Returns
        -------
        Tree in requested format, or None if not reconstructed yet
        """
        tree_key = self.sm_tree_key(intbc_id)
        if tree_key not in self.obst:
            return None

        nx_tree = self.obst[tree_key]

        if not as_cassiopeia:
            return nx_tree

        # Convert to CassiopeiaTree
        # Note: populate_tree must be called BEFORE setting character matrix
        # because populate_tree calls set_character_states_at_leaves which
        # validates that character matrix indices match tree leaves
        char_matrix = self._get_character_matrix("sm", intbc_id=intbc_id)
        cas_tree = CassiopeiaTree(
            missing_state_indicator=-1
        )
        cas_tree.populate_tree(nx_tree)
        cas_tree.character_matrix = char_matrix
        return cas_tree

    def get_all_sm_trees(
        self,
        as_cassiopeia: bool = False,
    ) -> dict[str, nx.DiGraph | CassiopeiaTree | None]:
        """
        Get all single-molecule trees.

        Returns
        -------
        dict
            Dictionary mapping intbc_id -> tree
        """
        return {
            intbc_id: self.get_sm_tree(intbc_id, as_cassiopeia=as_cassiopeia)
            for intbc_id in self.intbc_ids
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
            Tree key in .obst (e.g., 'sc', '0', '1', '2')
        **kwargs
            Additional arguments for pc.pl.plot_tree

        Returns
        -------
        Plot object from pycea
        """
        import pycea as pc

        return pc.pl.plot_tree(self, tree=tree, **kwargs)
    
    def plot_parallel(self, tree: str  | None = None, figsize=None, **kwargs: Any) -> Any:
        """
        """
        import pycea as pc
        import matplotlib.pyplot as plt
        actual_layers = self.obst_keys() if tree is None else [tree]

        if figsize is None:
            figsize=(len(actual_layers)*2.5, 50)
        fig, axes = plt.subplots(1, len(actual_layers), figsize=figsize)

        for i, tree in enumerate(actual_layers):
            pc.pl.tree(self, tree=tree, ax=axes[i], **kwargs)

        plt.tight_layout()
    
        return fig

    def add_tree_metrics(self, tree: str | None = None) -> None:
        """Add tree metrics using pycea."""
        import pycea as pc
        if tree is None:
            # iterate through all trees 
            for tree in self.obst_keys():
                pc.pp.add_depth(self, tree=tree)
        else:
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


