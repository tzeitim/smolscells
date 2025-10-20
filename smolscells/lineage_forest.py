"""  New class
"""
from cassiopeia.data import CassiopeiaTree
from treedata import TreeData
import networkx as nx

from pathlib import Path

class LineageForest:
    """
    Container for multiple lineage trees with optional associated data.

    Manages:
    - Multiple CassiopeiaTree objects (topologies + character matrices)
    - Optional per-tree TreeData objects (created lazily on access)
    - Shared TreeData for common data across all trees
    - Unstructured metadata (uns)
    """

    def __init__(
        self,
        shared_tdata: TreeData | None = None,
    ):
        self.trees = {}  # Main trees
        self.tdatas = {} # gets created lazily from .trees
        self.uns = {}
        self.shared_tdata = shared_tdata  # Shared data across all observations
        self._sm_counter = 0 # auto-increment counter for sm trees

    def __repr__(self) -> str:
        has_tdatas = f"  Tdatas: {len(self.tdatas)} ({', '.join(self.tdatas.keys())})\n" if self.tdatas else ""

        return (
            f"LineageForest\n"
            f"  Trees: {len(self.trees)} ({', '.join(self.tree_keys)})\n"
            f"{has_tdatas}"
            f"  Shared data: {self.shared_tdata is not None}"
        )

    def __getattr__(self, name: str):
        """Dynamic attribute access with slim naming convention.

        Provides ergonomic access to trees and data without dict keys.

        CassiopeiaTree access:
        - lf.csc → single-cell tree
        - lf.c0, lf.c1, ..., lf.cn → single-molecule trees

        TreeData access (lazy loading):
        - lf.tsc → single-cell data
        - lf.t0, lf.t1, ..., lf.tn → single-molecule data

        Examples:
            lf.csc       # Same as lf.trees['sc']
            lf.c0        # Same as lf.trees['sm_0']
            lf.tsc       # Same as lf.get_tdata('sc')
            lf.t0        # Same as lf.get_tdata('sm_0')
        """
        # CassiopeiaTree access
        if name == 'csc':
            return self.trees.get('sc')

        if name.startswith('c') and len(name) > 1 and name[1:].isdigit():
            tree_id = name[1:]  # 'c0' → '0'
            return self.trees.get(f'sm_{tree_id}')

        # TreeData access (lazy loading)
        if name == 'tsc':
            return self.get_tdata('sc')

        if name.startswith('t') and len(name) > 1 and name[1:].isdigit():
            tree_id = name[1:]  # 't0' → '0'
            return self.get_tdata(f'sm_{tree_id}')

        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    def reset(self):
        """Clear all trees and reset counters."""
        self.trees.clear()
        self.tdatas.clear()
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

    def get_tdata(self, tree_key: str, alignment: str = "subset") -> TreeData | None:
        """Get TreeData for tree, creating on-demand if needed.

        Parameters
        ----------
        tree_key
            Tree identifier (e.g., 'sc', 'sm_0')
        alignment
            Alignment mode for TreeData if creating new

        Returns
        -------
        TreeData or None

        Examples
        --------
        tdata = lf.get_tdata('sc')  # Creates if doesn't exist
        tdata = lf.get_tdata('sm_0')
        """
        # Return if already exists
        if tree_key in self.tdatas:
            return self.tdatas[tree_key]

        # Create on-demand if tree exists
        if tree_key in self.trees:
            self._sync_tdatas_from_trees(
                tree_keys=[tree_key],
                alignment=alignment,
                overwrite=False
            )
            return self.tdatas.get(tree_key)

        return None

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
        return list(self.trees.keys())

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
            alignment: str = "subset",
            overwrite: bool = False
            ) -> None:
        """Populate tdatas from trees.

          Creates TreeData objects from CassiopeiaTree topologies.
          Note: Character matrices remain with CassiopeiaTree (use get_character_matrix).

          Parameters
          ----------
          tree_keys
              Specific tree keys to sync. If None, syncs all trees.
          alignment
              Alignment mode for TreeData ('subset', 'intersect', etc.)
          overwrite
              If True, replace existing TreeData. If False, skip existing.

          Examples
          --------
          # Sync all trees
          lf._sync_tdatas_from_trees()

          # Sync only new trees
          lf._sync_tdatas_from_trees(overwrite=False)

          # Sync specific trees
          lf._sync_tdatas_from_trees(tree_keys=['sc', 'sm_0'])
        """
        if tree_keys is None:
            tree_keys = list(self.trees.keys())

        for tree_key in tree_keys:
            # Skip if already exists and not overwriting
            if tree_key in self.tdatas and not overwrite:
                continue

            tree = self.trees.get(tree_key)
            if tree is None:
                continue

            # Create TreeData from CassiopeiaTree topology
            # Note: Character matrices stay with CassiopeiaTree (different dimensions)
            # Check if tree has been solved (has topology)
            try:
                topology = tree.get_tree_topology()
                if topology is not None and len(topology.nodes()) > 0:
                    tdata = TreeData(
                        obst={tree_key: topology},
                        alignment=alignment
                    )
                    self.tdatas[tree_key] = tdata
            except Exception as e:
                # Tree not solved yet, skip silently
                continue

    def _clear_tdatas(self) -> None:
        """Clear all TreeData objects."""
        self.tdatas.clear()

    def _remove_tdata(self, tree_key: str) -> None:
        """Remove TreeData for specific tree."""
        if tree_key in self.tdatas:
            del self.tdatas[tree_key]
