"""Tree comparison tools for SmolScells"""

import numpy as np
from typing import Dict, Any, Set, Optional, Tuple
import cassiopeia as cass


class TreeComparator:
    """Compare reconstructed trees with ground truth"""
    
    def __init__(self, ground_truth: cass.data.CassiopeiaTree):
        """
        Initialize tree comparator
        
        Args:
            ground_truth: Ground truth tree for comparison
        """
        self.ground_truth = ground_truth
        self.metrics = {}
    
    def get_leaves_under_node(self, tree: cass.data.CassiopeiaTree, node: str) -> Set[str]:
        """
        Get all leaf nodes descended from a given node
        
        Args:
            tree: CassiopeiaTree object
            node: Node identifier
            
        Returns:
            Set of leaf node names
        """
        if tree.is_leaf(node):
            return {node}
        return set(tree.leaves_in_subtree(node))
    
    def calculate_robinson_foulds(
        self,
        test_tree: cass.data.CassiopeiaTree,
        normalize: bool = True
    ) -> float:
        """
        Calculate Robinson-Foulds distance between trees
        
        Args:
            test_tree: Tree to compare with ground truth
            normalize: Whether to normalize by maximum possible distance
            
        Returns:
            Robinson-Foulds distance
        """
        # Get bipartitions for both trees
        gt_bipartitions = self._get_bipartitions(self.ground_truth)
        test_bipartitions = self._get_bipartitions(test_tree)
        
        # Calculate symmetric difference
        symmetric_diff = gt_bipartitions.symmetric_difference(test_bipartitions)
        rf_distance = len(symmetric_diff)
        
        if normalize:
            # Maximum possible RF distance is 2*(n-3) for n leaves
            n_leaves = len(self.ground_truth.leaves)
            max_distance = 2 * (n_leaves - 3) if n_leaves > 3 else 1
            rf_distance = rf_distance / max_distance
        
        return rf_distance
    
    def _get_bipartitions(self, tree: cass.data.CassiopeiaTree) -> Set[frozenset]:
        """
        Get all bipartitions (splits) in a tree
        
        Args:
            tree: CassiopeiaTree object
            
        Returns:
            Set of bipartitions
        """
        bipartitions = set()
        all_leaves = set(tree.leaves)
        
        for node in tree.nodes:
            if not tree.is_leaf(node) and node != tree.root:
                # Get leaves under this node
                leaves_under = self.get_leaves_under_node(tree, node)
                leaves_complement = all_leaves - leaves_under
                
                # Create bipartition (use frozenset for hashability)
                if len(leaves_under) > 0 and len(leaves_complement) > 0:
                    # Store the smaller partition to ensure uniqueness
                    if len(leaves_under) <= len(leaves_complement):
                        bipartitions.add(frozenset(leaves_under))
                    else:
                        bipartitions.add(frozenset(leaves_complement))
        
        return bipartitions
    
    def calculate_triplet_accuracy(
        self,
        test_tree: cass.data.CassiopeiaTree,
        sample_size: Optional[int] = None
    ) -> float:
        """
        Calculate triplet accuracy between trees
        
        Args:
            test_tree: Tree to compare with ground truth
            sample_size: Number of triplets to sample (None for all)
            
        Returns:
            Fraction of correctly resolved triplets
        """
        import itertools
        
        leaves = list(set(self.ground_truth.leaves) & set(test_tree.leaves))
        
        if len(leaves) < 3:
            return 0.0
        
        # Generate triplets
        if sample_size and sample_size < len(list(itertools.combinations(leaves, 3))):
            # Sample triplets
            triplets = []
            for _ in range(sample_size):
                triplet = np.random.choice(leaves, 3, replace=False)
                triplets.append(tuple(triplet))
        else:
            # Use all triplets
            triplets = list(itertools.combinations(leaves, 3))
        
        correct = 0
        total = 0
        
        for a, b, c in triplets:
            gt_topology = self._get_triplet_topology(self.ground_truth, a, b, c)
            test_topology = self._get_triplet_topology(test_tree, a, b, c)
            
            if gt_topology and test_topology:
                if gt_topology == test_topology:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _get_triplet_topology(
        self,
        tree: cass.data.CassiopeiaTree,
        a: str, b: str, c: str
    ) -> Optional[Tuple[str, str, str]]:
        """
        Get the topology of a triplet in a tree
        
        Args:
            tree: CassiopeiaTree object
            a, b, c: Three leaf nodes
            
        Returns:
            Tuple indicating which two leaves are closer, or None if unresolved
        """
        try:
            # Find LCA for each pair
            lca_ab = tree.get_lca(a, b)
            lca_ac = tree.get_lca(a, c)
            lca_bc = tree.get_lca(b, c)
            
            # Get depths
            depth_ab = tree.depth(lca_ab) if hasattr(tree, 'depth') else 0
            depth_ac = tree.depth(lca_ac) if hasattr(tree, 'depth') else 0
            depth_bc = tree.depth(lca_bc) if hasattr(tree, 'depth') else 0
            
            # Determine topology based on LCA depths
            if depth_ab > depth_ac and depth_ab > depth_bc:
                return (a, b, c)  # a and b are closer
            elif depth_ac > depth_ab and depth_ac > depth_bc:
                return (a, c, b)  # a and c are closer
            elif depth_bc > depth_ab and depth_bc > depth_ac:
                return (b, c, a)  # b and c are closer
            else:
                return None  # Unresolved
        except:
            return None
    
    def compare_tree(
        self,
        test_tree: cass.data.CassiopeiaTree,
        metrics: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Compute multiple comparison metrics
        
        Args:
            test_tree: Tree to compare with ground truth
            metrics: List of metrics to compute (default: all)
            
        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ['robinson_foulds', 'triplet_accuracy', 'common_leaves']
        
        results = {}
        
        if 'robinson_foulds' in metrics:
            results['robinson_foulds'] = self.calculate_robinson_foulds(test_tree)
        
        if 'triplet_accuracy' in metrics:
            results['triplet_accuracy'] = self.calculate_triplet_accuracy(test_tree)
        
        if 'common_leaves' in metrics:
            gt_leaves = set(self.ground_truth.leaves)
            test_leaves = set(test_tree.leaves)
            results['common_leaves'] = len(gt_leaves & test_leaves)
            results['gt_leaves'] = len(gt_leaves)
            results['test_leaves'] = len(test_leaves)
        
        return results