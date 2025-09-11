"""Fitness analysis tools for SmolScells"""

import numpy as np
from typing import Dict, Any, List, Optional
import cassiopeia as cass


class FitnessAnalyzer:
    """Analyze fitness effects in reconstructed trees"""
    
    def __init__(self, trees: Dict[str, cass.data.CassiopeiaTree]):
        """
        Initialize fitness analyzer
        
        Args:
            trees: Dictionary of reconstructed trees
        """
        self.trees = trees
        self.fitness_scores = {}
    
    def calculate_lbi(self, tree_name: str, tau: float = 0.1) -> Dict[str, float]:
        """
        Calculate Local Branching Index (LBI) for a tree
        
        Args:
            tree_name: Name of the tree to analyze
            tau: LBI parameter controlling local neighborhood size
            
        Returns:
            Dictionary mapping node names to LBI scores
        """
        if tree_name not in self.trees:
            raise ValueError(f"Tree {tree_name} not found")
        
        tree = self.trees[tree_name]
        
        # Use Cassiopeia's LBIJungle tool
        from cassiopeia.tools import LBIJungle
        
        lbi_tool = LBIJungle()
        lbi_tool.setup(tree, tau=tau)
        lbi_tool.fit(tree)
        
        # Extract LBI scores
        lbi_scores = {}
        for node in tree.nodes:
            if hasattr(tree, 'nodes') and node in tree.nodes:
                node_data = tree.nodes[node]
                if 'LBI' in node_data:
                    lbi_scores[node] = node_data['LBI']
        
        return lbi_scores
    
    def compare_fitness_detection(
        self,
        ground_truth_tree: str = 'recorded_ground_truth',
        test_trees: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare fitness detection across different tree reconstructions
        
        Args:
            ground_truth_tree: Name of ground truth tree
            test_trees: List of tree names to test
            
        Returns:
            Dictionary with comparison metrics
        """
        if test_trees is None:
            test_trees = [name for name in self.trees.keys() 
                         if name != ground_truth_tree]
        
        results = {}
        
        # Calculate LBI for ground truth
        gt_lbi = self.calculate_lbi(ground_truth_tree)
        
        # Calculate LBI for test trees and compare
        for tree_name in test_trees:
            try:
                test_lbi = self.calculate_lbi(tree_name)
                
                # Find common nodes
                common_nodes = set(gt_lbi.keys()) & set(test_lbi.keys())
                
                if common_nodes:
                    # Calculate correlation
                    gt_values = [gt_lbi[node] for node in common_nodes]
                    test_values = [test_lbi[node] for node in common_nodes]
                    
                    correlation = np.corrcoef(gt_values, test_values)[0, 1]
                    
                    results[tree_name] = {
                        'correlation': correlation,
                        'common_nodes': len(common_nodes),
                        'gt_nodes': len(gt_lbi),
                        'test_nodes': len(test_lbi)
                    }
                else:
                    results[tree_name] = {
                        'correlation': 0.0,
                        'common_nodes': 0,
                        'gt_nodes': len(gt_lbi),
                        'test_nodes': len(test_lbi)
                    }
                    
            except Exception as e:
                results[tree_name] = {'error': str(e)}
        
        return results
    
    def aggregate_bulk_fitness(
        self,
        single_cell_tree: str = 'single_cell',
        bulk_trees: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Aggregate fitness information from bulk trees to enhance single-cell tree
        
        Args:
            single_cell_tree: Name of single-cell tree
            bulk_trees: List of bulk tree names
            
        Returns:
            Enhanced fitness scores for single-cell tree nodes
        """
        if bulk_trees is None:
            bulk_trees = [name for name in self.trees.keys() 
                         if name.startswith('single_molecule_')]
        
        # Get single-cell tree LBI
        sc_lbi = self.calculate_lbi(single_cell_tree)
        
        # Aggregate bulk tree LBI scores
        bulk_lbi_scores = []
        for tree_name in bulk_trees:
            try:
                bulk_lbi = self.calculate_lbi(tree_name)
                bulk_lbi_scores.append(bulk_lbi)
            except:
                continue
        
        # Simple aggregation: average bulk LBI scores for matching nodes
        enhanced_scores = {}
        for node in sc_lbi:
            scores = [sc_lbi[node]]
            
            # Add bulk scores if available
            for bulk_lbi in bulk_lbi_scores:
                if node in bulk_lbi:
                    scores.append(bulk_lbi[node])
            
            # Use weighted average (higher weight for single-cell)
            if len(scores) > 1:
                weights = [2.0] + [1.0] * (len(scores) - 1)
                enhanced_scores[node] = np.average(scores, weights=weights)
            else:
                enhanced_scores[node] = scores[0]
        
        return enhanced_scores