"""  New class
"""
from cassiopeia.data import CassiopeiaTree
from treedata import TreeData
import networkx as nx

from pathlib import Path

class LineageForest:
    """
    Container for multiple lineage trees with optional associated data.

    Architecture:
    - CassiopeiaTree objects in .trees are the source of truth
    - TreeData objects in ._tdata_cache are ephemeral visualization cache
    - Modifications should target CassiopeiaTree or its networkx topology
    - TreeData is auto-generated from CassiopeiaTree on access

    Manages:
    - Multiple CassiopeiaTree objects (topologies + character matrices)
    - Optional TreeData cache for visualization (created lazily)
    - Shared TreeData for common data across all trees
    - Unstructured metadata (uns)
    """

    def __init__(
        self,
        shared_tdata: TreeData | None = None,
    ):
        self.trees = {}  # CassiopeiaTree objects - SOURCE OF TRUTH
        self._tdata_cache = {}  # TreeData cache for visualization - EPHEMERAL
        self.uns = {}
        self.shared_tdata = shared_tdata  # Shared data across all observations
        self._sm_counter = 0 # auto-increment counter for sm trees

    def __repr__(self) -> str:
        has_tdatas = f"  TreeData cache: {len(self._tdata_cache)} ({', '.join(self._tdata_cache.keys())})\n" if self._tdata_cache else ""

        return (
            f"LineageForest\n"
            f"  Trees: {len(self.trees)} ({', '.join(self.tree_keys)})\n"
            f"{has_tdatas}"
            f"  Shared data: {self.shared_tdata is not None}"
        )

    def __getattr__(self, name: str):
        """Dynamic attribute access with slim naming convention.

        CassiopeiaTree access:
        - lf.csc → single-cell tree
        - lf.c0, lf.c1, ... → single-molecule trees

        NetworkX graph access:
        - lf.gsc → nx.DiGraph for single-cell tree
        - lf.g0, lf.g1, ... → nx.DiGraph for single-molecule trees

        TreeData access:
        - lf.tsc → single-cell TreeData
        - lf.t0, lf.t1, ... → single-molecule TreeData

        Examples:
            lf.csc       # lf.trees['sc']
            lf.c0        # lf.trees['sm_0']
            lf.gsc       # lf.get_graph('sc')
            lf.g0        # lf.get_graph('sm_0')
            lf.tsc       # lf.get_tdata('sc')
            lf.t0        # lf.get_tdata('sm_0')
        """
        # CassiopeiaTree access
        if name == 'csc':
            return self.trees.get('sc')

        if name.startswith('c') and len(name) > 1 and name[1:].isdigit():
            tree_id = name[1:]
            return self.trees.get(f'sm_{tree_id}')

        # NetworkX graph access
        if name == 'gsc':
            return self.get_graph('sc')

        if name.startswith('g') and len(name) > 1 and name[1:].isdigit():
            tree_id = name[1:]
            return self.get_graph(f'sm_{tree_id}')

        # TreeData access
        if name == 'tsc':
            return self.get_tdata('sc')

        if name.startswith('t') and len(name) > 1 and name[1:].isdigit():
            tree_id = name[1:]
            return self.get_tdata(f'sm_{tree_id}')

        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    def reset(self):
        """Clear all trees and reset counters."""
        self.trees.clear()
        self._tdata_cache.clear()
        self.uns.clear()
        self._sm_counter = 0 

    # tree accessors
    def get_tree(
            self,
            tree_key: str,
            as_cassiopeia: bool = True,
            )-> CassiopeiaTree | nx.DiGraph | None:
        """ Get tree by key """
        tree = self.trees.get(tree_key)

        if tree is None:
            return None
        if as_cassiopeia:
            return tree
        return tree.get_tree_topology()

    def get_sc_tree(self) -> CassiopeiaTree | None:
        """ Get single-cell tree """
        return self.get_tree('sc')

    def get_sm_tree(self, intbc_id: int | str) -> CassiopeiaTree | None:
        """ Get single-molecule tree by ID """
        return self.get_tree(f'sm_{intbc_id}')

    def get_graph(self, tree_key: str, copy: bool = False) -> nx.DiGraph | None:
        """Get networkx graph from CassiopeiaTree.

        By default returns reference to internal network for in-place modifications.
        Set copy=True to get a snapshot.

        Parameters
        ----------
        tree_key : str
            Tree identifier (e.g., 'sc', 'sm_0')
        copy : bool
            If True, return copy. If False, return reference to internal network.

        Returns
        -------
        nx.DiGraph or None
            NetworkX graph if tree exists, None otherwise

        Examples
        --------
        graph = lf.get_graph('sc')
        graph.nodes['node1']['color'] = 'red'  # Persists in CassiopeiaTree

        snapshot = lf.get_graph('sc', copy=True)  # Read-only snapshot
        """
        tree = self.trees.get(tree_key)
        if tree is not None:
            if copy:
                return tree.get_tree_topology()
            else:
                # Access internal network directly for in-place modifications
                return tree._CassiopeiaTree__network
        return None

    def get_tdata(self, tree_key: str, alignment: str = "subset", refresh: bool = False) -> TreeData | None:
        """Get TreeData for visualization, creating/caching on-demand.

        TreeData is ephemeral cache synced from CassiopeiaTree topology.
        Use refresh=True to rebuild TreeData after modifying graphs.

        Parameters
        ----------
        tree_key : str
            Tree identifier (e.g., 'sc', 'sm_0')
        alignment : str
            Alignment mode for TreeData
        refresh : bool
            Force refresh from CassiopeiaTree

        Returns
        -------
        TreeData or None

        Examples
        --------
        tdata = lf.get_tdata('sc')
        tdata = lf.get_tdata('sc', refresh=True)
        """
        # Return cached if exists and not refreshing
        if tree_key in self._tdata_cache and not refresh:
            return self._tdata_cache[tree_key]

        # Create from CassiopeiaTree if tree exists
        if tree_key in self.trees:
            self._sync_tdatas_from_trees(
                tree_keys=[tree_key],
                alignment=alignment,
                overwrite=True
            )
            return self._tdata_cache.get(tree_key)

        return None

    def invalidate_tdata_cache(self, tree_key: str | None = None) -> None:
        """Clear TreeData cache.

        Call after modifying graphs to force TreeData regeneration on next access.

        Parameters
        ----------
        tree_key : str or None
            Specific tree to invalidate, or None to clear all

        Examples
        --------
        lf.gsc.nodes['node1']['color'] = 'red'
        lf.invalidate_tdata_cache('sc')
        lf.invalidate_tdata_cache()  # Clear all
        """
        if tree_key:
            self._tdata_cache.pop(tree_key, None)
        else:
            self._tdata_cache.clear()

    def get_character_matrix(self, tree_key: str):
        """Get character matrix from CassiopeiaTree.

        Parameters
        ----------
        tree_key
            Tree identifier (e.g., 'sc', 'sm_0')

        Returns
        -------
        pd.DataFrame or None
            Character matrix (leaves × characters)

        Examples
        --------
        chars = lf.get_character_matrix('sc')
        chars = lf.get_character_matrix('sm_0')
        """
        tree = self.trees.get(tree_key)
        if tree is not None:
            return tree.character_matrix
        return None

    def initialize_tdatas(
        self,
        tree_keys: list[str] | None = None,
        alignment: str = "subset"
    ) -> None:
        """Initialize TreeData objects for all or specified solved trees.

        Eagerly creates TreeData objects instead of waiting for lazy loading.
        Useful before batch operations like plotting multiple trees.

        Parameters
        ----------
        tree_keys
            Specific tree keys to initialize. If None, initializes all trees.
        alignment
            Alignment mode for TreeData

        Examples
        --------
        # Initialize all TreeData objects
        lf.initialize_tdatas()

        # Initialize only SC and first SM tree
        lf.initialize_tdatas(tree_keys=['sc', 'sm_0'])

        # Now access is instant (already cached)
        lf.tsc  # No lazy creation needed
        lf.t0   # No lazy creation needed
        """
        self._sync_tdatas_from_trees(tree_keys=tree_keys, alignment=alignment)

    # tree addition with auto-indexing
    def add_tree(
            self,
            tree_key: str,
            tree: CassiopeiaTree,
            ) -> None:
        """Add a tree with explicit key (no auto-formatting).

        Parameters
        ----------
        tree_key
            Explicit tree identifier (e.g., 'sc', 'gt', 'sm_0', 'custom')
        tree
            CassiopeiaTree object

        Examples
        --------
        lf.add_tree('sc', sc_tree)
        lf.add_tree('gt', gt_tree)
        lf.add_tree('sm_ACTCG', barcode_tree)
        """
        self.trees[tree_key] = tree

    def add_sc_tree(self, tree: CassiopeiaTree) -> None:
        """Add single-cell tree (always stored as 'sc')."""
        self.trees['sc'] = tree

    def add_sm_tree(self, tree: CassiopeiaTree) -> int:
        """Add SM tree with auto-incremented index.

        Automatically assigns index based on order of addition.

        Returns
        -------
        int
            The assigned index

        Examples
        --------
        idx0 = lf.add_sm_tree(tree0)  # Returns 0, stored as 'sm_0'
        idx1 = lf.add_sm_tree(tree1)  # Returns 1, stored as 'sm_1'
        """
        tree_key = f'sm_{self._sm_counter}'
        self.trees[tree_key] = tree
        assigned_idx = self._sm_counter
        self._sm_counter += 1
        return assigned_idx

    def add_sm_tree_with_id(
        self,
        intbc_id: int | str,
        tree: CassiopeiaTree
    ) -> None:
        """Add SM tree with explicit ID.

        Parameters
        ----------
        intbc_id
            Explicit identifier (int, barcode string, etc.)
        tree
            CassiopeiaTree object

        Examples
        --------
        lf.add_sm_tree_with_id(5, tree)        # → 'sm_5'
        lf.add_sm_tree_with_id('ACTCG', tree)  # → 'sm_ACTCG'
        """
        tree_key = f'sm_{intbc_id}'
        self.trees[tree_key] = tree

        # Update counter if it's an integer ID
        if isinstance(intbc_id, int):
            self._sm_counter = max(self._sm_counter, intbc_id + 1)

    # Properties
    @property
    def tree_keys(self) -> list[str]:
        """ All tree keys """
        return sorted(list(self.trees.keys()))

    @property
    def smtrees_keys(self) -> list[int | str]:
        """ All single-molecule tree keys """
        return [
            k for k in self.trees.keys()
            if k.startswith('sm')
        ]

    @property
    def smtrees_items(self) -> list[tuple[int | str, CassiopeiaTree]]:
        """ Returns key:value pairs only for sm trees """
        return [(k, v) for k, v in self.trees.items() if k.startswith('sm')]

    @property
    def tdatas(self):
        """TreeData cache for all trees (lazy-loaded).

        Returns dict-like object with TreeData for visualization.
        """
        # Auto-create for all trees
        for tree_key in self.trees.keys():
            if tree_key not in self._tdata_cache:
                self.get_tdata(tree_key)
        return self._tdata_cache

    @property
    def tdatas_values(self):
        """Returns all TreeData objects for iteration.
        """
        return self.tdatas.values()
    
    @property
    def trees_values(self):
        """Returns all CassiopeiaTree objects for iteration.
        """
        return self.trees.values()

    @property
    def intbc_ids(self) -> list[int]:
        """ Integration barcode IDs (single-molecule trees only) """
        return sorted([int(k) for k in self.tree_keys if k != 'sc' and k != 'exp_tree'])
    
    @property
    def n_trees(self) -> int:
        """Number of trees in forest."""
        return len(self.trees)

    # I/O
    def save(self, filepath: Path | str) -> None:
        """Save LineageForest."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Path | str) -> "LineageForest":
        """Load LineageForest."""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def _sync_tdatas_from_trees(
            self,
            tree_keys: list[str] | None = None,
            alignment: str = "leaves",
            overwrite: bool = False
            ) -> None:
        """Sync TreeData cache from CassiopeiaTree topologies.

          Creates TreeData objects from CassiopeiaTree topologies for visualization.
          Character matrices are transferred to TreeData.obsm['characters'] for leaves.
          All TreeData objects use fixed "tree" key in obst.

          Parameters
          ----------
          tree_keys : list[str] or None
              Specific tree keys to sync. If None, syncs all trees.
          alignment : str
              Alignment mode for TreeData
          overwrite : bool
              If True, replace existing TreeData. If False, skip existing.

          Examples
          --------
          lf._sync_tdatas_from_trees()
          lf._sync_tdatas_from_trees(overwrite=False)
          lf._sync_tdatas_from_trees(tree_keys=['sc', 'sm_0'])
        """
        import pycea
        import pandas as pd

        if tree_keys is None:
            tree_keys = list(self.trees.keys())

        for tree_key in tree_keys:
            # Skip if already exists and not overwriting
            if tree_key in self._tdata_cache and not overwrite:
                continue

            tree = self.trees.get(tree_key)
            if tree is None:
                continue

            # Create TreeData from CassiopeiaTree topology
            # get_tree_topology() returns a copy that preserves all node/edge attributes
            try:
                topology = tree.get_tree_topology()
                if topology is not None and len(topology.nodes()) > 0:
                    tdata = TreeData(
                        obst={"tree": topology},
                        alignment=alignment
                    )
                    pycea.pp.add_depth(tdata)

                    # Transfer character matrix to TreeData if available
                    if tree.character_matrix is not None and len(tree.character_matrix) > 0:
                        # Character matrix only has leaves - reindex to match tdata.obs (all nodes)
                        # Missing nodes (internal nodes) will have NaN values
                        char_matrix_aligned = tree.character_matrix.reindex(tdata.obs.index)
                        tdata.obsm['characters'] = char_matrix_aligned

                    self._tdata_cache[tree_key] = tdata
            except Exception:
                continue

    def _clear_tdatas(self) -> None:
        """Clear all TreeData cache."""
        self._tdata_cache.clear()

    def _remove_tdata(self, tree_key: str) -> None:
        """Remove TreeData from cache."""
        if tree_key in self._tdata_cache:
            del self._tdata_cache[tree_key]
