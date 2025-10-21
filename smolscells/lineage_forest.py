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
        self.is_collapsed = None

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

    def collapse_to_alleles(
        self,
        tree_keys: list[str] | None = None,
        store_mappings: bool = True
    ) -> "LineageForest":
        """Collapse cells to alleles based on identical character states.

        Groups cells with identical character states into alleles. Returns a new
        LineageForest with allele-level character matrices and trees.

        Alleles are identified by their character states:
        - allele_id: comma-separated string (e.g., '1,2,0,3')
        - allele_id_hash: MD5 hash (8 chars) for compact display (e.g., 'a3f5b2c9')

        Parameters
        ----------
        tree_keys : list[str] or None
            Trees to collapse. If None, collapses all trees.
        store_mappings : bool
            If True, stores mappings and metrics in uns['allele_info']

        Returns
        -------
        LineageForest
            New LineageForest with allele-level data

        Examples
        --------
        allele_lf = lf.collapse_to_alleles()
        allele_lf = lf.collapse_to_alleles(tree_keys=['sc', 'sm_0'])

        Access mappings:
        allele_to_cells = allele_lf.uns['allele_info']['sc']['allele_to_cells']
        metrics = allele_lf.uns['allele_info']['sc']['metrics']
        """
        from cassiopeia.mixins.utilities import find_duplicate_groups
        import hashlib
        import pandas as pd

        if tree_keys is None:
            tree_keys = list(self.trees.keys())

        allele_lf = LineageForest(shared_tdata=self.shared_tdata)
        allele_lf.uns = self.uns.copy()

        if store_mappings:
            allele_lf.uns['allele_info'] = {}

        for tree_key in tree_keys:
            tree = self.trees.get(tree_key)
            if tree is None or tree.character_matrix is None:
                continue

            char_matrix = tree.character_matrix

            allele_groups = find_duplicate_groups(char_matrix)

            allele_to_cells = {}
            cell_to_allele = {}
            allele_sizes = {}
            allele_id_to_hash = {}

            allele_char_states = []
            allele_ids = []

            cells_in_groups = set()
            for group_idx, cell_group in enumerate(allele_groups.values()):
                cells = list(cell_group)

                char_state = tuple(char_matrix.loc[cells[0]].values)
                allele_id = ",".join(str(int(x)) for x in char_state)
                allele_id_hash = hashlib.md5(allele_id.encode()).hexdigest()[:8]

                allele_to_cells[allele_id] = cells
                allele_sizes[allele_id] = len(cells)
                allele_id_to_hash[allele_id] = allele_id_hash

                for cell in cells:
                    cell_to_allele[cell] = allele_id
                    cells_in_groups.add(cell)

                allele_char_states.append(char_state)
                allele_ids.append(allele_id)

            for cell in char_matrix.index:
                if cell not in cells_in_groups:
                    char_state = tuple(char_matrix.loc[cell].values)
                    allele_id = ",".join(str(int(x)) for x in char_state)
                    allele_id_hash = hashlib.md5(allele_id.encode()).hexdigest()[:8]

                    allele_to_cells[allele_id] = [cell]
                    allele_sizes[allele_id] = 1
                    allele_id_to_hash[allele_id] = allele_id_hash
                    cell_to_allele[cell] = allele_id

                    allele_char_states.append(char_state)
                    allele_ids.append(allele_id)

            allele_char_matrix = pd.DataFrame(
                allele_char_states,
                index=allele_ids,
                columns=char_matrix.columns
            )

            allele_tree = CassiopeiaTree(
                character_matrix=allele_char_matrix,
                missing_state_indicator=tree.missing_state_indicator
            )

            allele_lf.add_tree(tree_key, allele_tree)

            if store_mappings:
                total_cells = len(char_matrix)
                total_alleles = len(allele_ids)
                compression_ratio = total_cells / total_alleles if total_alleles > 0 else 0

                allele_lf.uns['allele_info'][tree_key] = {
                    'allele_to_cells': allele_to_cells,
                    'cell_to_allele': cell_to_allele,
                    'allele_sizes': allele_sizes,
                    'allele_id_to_hash': allele_id_to_hash,
                    'metrics': {
                        'total_cells': total_cells,
                        'total_alleles': total_alleles,
                        'compression_ratio': compression_ratio,
                        'avg_allele_size': total_cells / total_alleles if total_alleles > 0 else 0,
                        'max_allele_size': max(allele_sizes.values()) if allele_sizes else 0
                    }
                }

        return allele_lf

    def expand_to_cells(
        self,
        allele_lf: "LineageForest",
        tree_keys: list[str] | None = None,
        add_metadata: bool = True
    ) -> "LineageForest":
        """Expand allele-level trees back to cell-level.

        Takes solved allele trees and expands each allele leaf into its constituent
        cells as a direct polytomy. Internal nodes are preserved unchanged.

        Parameters
        ----------
        allele_lf : LineageForest
            Solved allele-level LineageForest (from collapse_to_alleles)
        tree_keys : list[str] or None
            Trees to expand. If None, expands all trees.
        add_metadata : bool
            If True, adds allele metadata to cell nodes

        Returns
        -------
        LineageForest
            New LineageForest with cell-level data

        Examples
        --------
        allele_lf = lf.collapse_to_alleles()
        # ... solve trees at allele level ...
        cell_lf = lf.expand_to_cells(allele_lf)

        Access cell metadata:
        tree = cell_lf.get_graph('sc')
        tree.nodes['cell_1']['allele_id']
        tree.nodes['cell_1']['allele_id_hash']
        tree.nodes['cell_1']['allele_size']
        tree.nodes['cell_1']['allele_siblings']
        """
        import copy

        if tree_keys is None:
            tree_keys = list(allele_lf.trees.keys())

        cell_lf = LineageForest(shared_tdata=self.shared_tdata)
        cell_lf.uns = allele_lf.uns.copy()

        for tree_key in tree_keys:
            allele_tree = allele_lf.trees.get(tree_key)
            if allele_tree is None:
                continue

            allele_info = allele_lf.uns.get('allele_info', {}).get(tree_key, {})
            allele_to_cells = allele_info.get('allele_to_cells', {})
            allele_id_to_hash = allele_info.get('allele_id_to_hash', {})
            allele_sizes = allele_info.get('allele_sizes', {})

            original_tree = self.trees.get(tree_key)
            if original_tree is None or original_tree.character_matrix is None:
                continue

            allele_graph = allele_tree.get_tree_topology()
            cell_graph = nx.DiGraph()

            for node in allele_graph.nodes():
                if node in allele_to_cells:
                    cells = allele_to_cells[node]

                    parent = list(allele_graph.predecessors(node))
                    if not parent:
                        for cell in cells:
                            cell_graph.add_node(cell, **allele_graph.nodes[node])

                            if add_metadata:
                                cell_graph.nodes[cell]['allele_id'] = node
                                cell_graph.nodes[cell]['allele_id_hash'] = allele_id_to_hash.get(node)
                                cell_graph.nodes[cell]['allele_size'] = allele_sizes.get(node, len(cells))
                                cell_graph.nodes[cell]['allele_siblings'] = [c for c in cells if c != cell]
                    else:
                        parent = parent[0]

                        edge_attrs = allele_graph.edges.get((parent, node), {})

                        for cell in cells:
                            cell_graph.add_node(cell, **allele_graph.nodes[node])
                            cell_graph.add_edge(parent, cell, **edge_attrs)

                            if add_metadata:
                                cell_graph.nodes[cell]['allele_id'] = node
                                cell_graph.nodes[cell]['allele_id_hash'] = allele_id_to_hash.get(node)
                                cell_graph.nodes[cell]['allele_size'] = allele_sizes.get(node, len(cells))
                                cell_graph.nodes[cell]['allele_siblings'] = [c for c in cells if c != cell]
                else:
                    cell_graph.add_node(node, **allele_graph.nodes[node])

                    for parent in allele_graph.predecessors(node):
                        edge_attrs = allele_graph.edges.get((parent, node), {})
                        cell_graph.add_edge(parent, node, **edge_attrs)

            cell_tree = CassiopeiaTree(
                character_matrix=original_tree.character_matrix.copy(),
                tree=cell_graph,
                missing_state_indicator=original_tree.missing_state_indicator
            )

            cell_lf.add_tree(tree_key, cell_tree)

        return cell_lf

    def get_allele_metrics(self, tree_key: str | None = None) -> dict:
        """Get allele compression metrics.

        Parameters
        ----------
        tree_key : str or None
            Specific tree to get metrics for. If None, returns metrics for all trees.

        Returns
        -------
        dict
            Metrics dictionary. If tree_key is None, returns dict[tree_key] = metrics.
            If tree_key is specified, returns metrics directly.

        Examples
        --------
        all_metrics = lf.get_allele_metrics()
        sc_metrics = lf.get_allele_metrics('sc')
        print(f"Compression: {sc_metrics['compression_ratio']:.1f}x")
        """
        allele_info = self.uns.get('allele_info', {})

        if tree_key is None:
            return {k: v.get('metrics', {}) for k, v in allele_info.items()}

        return allele_info.get(tree_key, {}).get('metrics', {})

    def get_cells_by_allele(self, tree_key: str, allele_id: str) -> list[str]:
        """Get list of cells belonging to an allele.

        Parameters
        ----------
        tree_key : str
            Tree identifier
        allele_id : str
            Allele ID (comma-separated character states)

        Returns
        -------
        list[str]
            Cell IDs in this allele

        Examples
        --------
        cells = lf.get_cells_by_allele('sc', '1,2,0,3')
        """
        allele_info = self.uns.get('allele_info', {}).get(tree_key, {})
        return allele_info.get('allele_to_cells', {}).get(allele_id, [])

    def color_by_allele_matches(
        self,
        matches: dict[str, str],
        source_tree: str,
        target_tree: str,
        color_attr: str = 'color'
    ) -> None:
        """Color nodes based on allele matches between trees.

        Takes a matching dictionary and colors both source and target nodes
        with consistent colors. Uses ColorHash for deterministic coloring.

        Parameters
        ----------
        matches : dict[str, str]
            Mapping from source allele IDs to target allele IDs
        source_tree : str
            Source tree key
        target_tree : str
            Target tree key
        color_attr : str
            Node attribute name for color (default: 'color')

        Examples
        --------
        lf.color_by_allele_matches(matches, 'sc', 'sm_0')
        lf.invalidate_tdata_cache()
        """
        from colorhash import ColorHash

        source_graph = self.get_graph(source_tree)
        target_graph = self.get_graph(target_tree)

        if source_graph is None or target_graph is None:
            return

        for source_allele, target_allele in matches.items():
            color = ColorHash(source_allele).hex

            if source_allele in source_graph.nodes:
                source_graph.nodes[source_allele][color_attr] = color

            if target_allele in target_graph.nodes:
                target_graph.nodes[target_allele][color_attr] = color

    def get_allele_match_summary(
        self,
        matches: dict[str, str],
        source_tree: str,
        target_tree: str
    ) -> dict:
        """Get summary statistics for allele matches.

        Parameters
        ----------
        matches : dict[str, str]
            Mapping from source to target allele IDs
        source_tree : str
            Source tree key
        target_tree : str
            Target tree key

        Returns
        -------
        dict
            Summary with total_matches, source_coverage, target_coverage, etc.

        Examples
        --------
        summary = lf.get_allele_match_summary(matches, 'sc', 'sm_0')
        print(f"Matched {summary['total_matches']} alleles")
        """
        source_info = self.uns.get('allele_info', {}).get(source_tree, {})
        target_info = self.uns.get('allele_info', {}).get(target_tree, {})

        source_alleles = set(source_info.get('allele_to_cells', {}).keys())
        target_alleles = set(target_info.get('allele_to_cells', {}).keys())

        matched_source = set(matches.keys())
        matched_target = set(matches.values())

        return {
            'total_matches': len(matches),
            'source_total': len(source_alleles),
            'target_total': len(target_alleles),
            'source_matched': len(matched_source),
            'target_matched': len(matched_target),
            'source_coverage': len(matched_source) / len(source_alleles) if source_alleles else 0,
            'target_coverage': len(matched_target) / len(target_alleles) if target_alleles else 0,
        }

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
