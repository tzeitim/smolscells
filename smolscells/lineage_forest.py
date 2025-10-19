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
    - Optional per-tree or shared TreeData objects
    - Simulation metadata (configs, ground truth)
    """

    def __init__(
        self,
        shared_tdata: TreeData | None = None,
    ):
        self.trees = {}  # Main trees
        self.tdatas = {} # gets created based on .trees
        self.uns = {}
        self.shared_tdata = shared_tdata  # Shared data across all observations

    def __getattr__(self, name: str):
        """Dynamic attribute access with slim naming convention.

        Provides ergonomic access to trees and data without dict keys.

        CassiopeiaTree access:
        - lf.csc → single-cell tree
        - lf.c0, lf.c1, ..., lf.cn → single-molecule trees

        TreeData access:
        - lf.tsc → single-cell data
        - lf.t0, lf.t1, ..., lf.tn → single-molecule data

        Examples:
            lf.csc       # Same as lf.trees['sc']
            lf.c0        # Same as lf.trees['0']
            lf.tsc       # Same as lf.tdata['sc']
            lf.t0        # Same as lf.tdata['0']
        """
        # CassiopeiaTree access
        if name == 'csc':
            return self.trees.get('sc')

        if name.startswith('c') and len(name) > 1 and name[1:].isdigit():
            tree_id = name[1:]  # 'c0' → '0'
            return self.trees.get(tree_id)

        # TreeData access
        if name == 'tsc':
            return self.tdatas.get('sc')

        if name.startswith('t') and len(name) > 1 and name[1:].isdigit():
            tree_id = name[1:]  # 't0' → '0'
            return self.tdatas.get(tree_id)

        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )
    
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

        return self.get_tree(str(intbc_id))
   
    def add_tree(
            self,
            tree_key: str,
            tree: CassiopeiaTree
            ) -> None:
        """ Add a tree to the forest """
        self.trees[tree_key] = tree
        # TODO add sync to tdatas
        
    @property
    def tree_keys(self) -> list[str]:
        """ All tree keys """
        return list(self.tree.keys())

    @property
    def intbc_ids(self) -> list[int]:
        """ Integration barcode IDs (single-molecule trees only) """
        return sorted([int(k) for k in self.tree_keys if k != 'sc' and k != 'exp_tree'])

    def __repr__(self) -> str:
        return (
            f"LineageForest\n"
            f"  Trees: {len(self.trees)} ({', '.join(self.tree_keys)})\n"
            f"  Observations: {self.n_obs}\n"
            f"  Shared data: {self.shared_tdata is not None}"
        )

    # I/O
    def save(self, filepath: Path | str) -> None:
        """Save LineageForest."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Path | str) -> LineageForest:
        """Load LineageForest."""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)

