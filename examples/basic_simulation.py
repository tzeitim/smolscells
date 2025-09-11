#!/usr/bin/env python3
"""
Basic SmolScells simulation example

This script demonstrates how to:
1. Run a complete simulation
2. Analyze the results
3. Compare reconstruction accuracy
"""

from smolscells import SmolScellsSimulator
from smolscells.analysis import FitnessAnalyzer, TreeComparator
import numpy as np


def main():
    """Run a basic SmolScells simulation"""
    
    print("="*60)
    print("SmolScells Basic Simulation Example")
    print("="*60)
    
    # Create simulator with default configuration
    print("\n1. Initializing simulator...")
    sim = SmolScellsSimulator(config_path="configs/default.yaml")
    
    # Run complete simulation
    print("\n2. Running simulation pipeline...")
    results = sim.run_complete_simulation()
    
    # Extract trees
    recorded_gt = results['recorded_gt_tree']
    single_cell_tree = results['trees'].get('single_cell')
    bulk_trees = {k: v for k, v in results['trees'].items() 
                  if 'single_molecule' in k}
    
    print(f"\nGenerated trees:")
    print(f"  - Ground truth: {len(recorded_gt.leaves)} leaves")
    if single_cell_tree:
        print(f"  - Single-cell: {len(single_cell_tree.leaves)} leaves")
    print(f"  - Bulk trees: {len(bulk_trees)} integration barcodes")
    
    # Analyze fitness detection
    print("\n3. Analyzing fitness detection...")
    analyzer = FitnessAnalyzer(results['trees'])
    
    # Calculate LBI for ground truth
    gt_lbi = analyzer.calculate_lbi('recorded_ground_truth', tau=0.1)
    print(f"  Ground truth LBI computed for {len(gt_lbi)} nodes")
    
    # Compare fitness detection across modalities
    fitness_comparison = analyzer.compare_fitness_detection()
    
    print("\nFitness detection correlation with ground truth:")
    for tree_name, metrics in fitness_comparison.items():
        if 'correlation' in metrics:
            print(f"  {tree_name}: r={metrics['correlation']:.3f} "
                  f"({metrics['common_nodes']} common nodes)")
    
    # Compare tree reconstruction accuracy
    print("\n4. Evaluating reconstruction accuracy...")
    comparator = TreeComparator(recorded_gt)
    
    if single_cell_tree:
        sc_metrics = comparator.compare_tree(single_cell_tree)
        print(f"\nSingle-cell tree metrics:")
        print(f"  Robinson-Foulds: {sc_metrics.get('robinson_foulds', 0):.3f}")
        print(f"  Triplet accuracy: {sc_metrics.get('triplet_accuracy', 0):.3f}")
        print(f"  Common leaves: {sc_metrics.get('common_leaves', 0)}/{sc_metrics.get('gt_leaves', 0)}")
    
    # Test bulk-enhanced fitness detection
    print("\n5. Testing bulk-enhanced fitness detection...")
    enhanced_scores = analyzer.aggregate_bulk_fitness()
    print(f"  Enhanced fitness scores computed for {len(enhanced_scores)} nodes")
    
    # Calculate improvement
    if single_cell_tree and 'single_cell' in fitness_comparison:
        original_corr = fitness_comparison['single_cell'].get('correlation', 0)
        
        # Compare enhanced scores with ground truth
        common_nodes = set(gt_lbi.keys()) & set(enhanced_scores.keys())
        if common_nodes:
            gt_values = [gt_lbi[node] for node in common_nodes]
            enhanced_values = [enhanced_scores[node] for node in common_nodes]
            enhanced_corr = np.corrcoef(gt_values, enhanced_values)[0, 1]
            
            improvement = (enhanced_corr - original_corr) / original_corr * 100
            print(f"\n  Original correlation: {original_corr:.3f}")
            print(f"  Enhanced correlation: {enhanced_corr:.3f}")
            print(f"  Improvement: {improvement:.1f}%")
    
    # Save results
    print("\n6. Saving results...")
    sim.save_results(save_trees=True, save_matrices=True)
    print(f"  Results saved to: {sim.output_dirs['base']}")
    
    print("\n" + "="*60)
    print("Simulation complete!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()