#!/usr/bin/env python3
"""
Batch experiments for SmolScells

This script demonstrates how to:
1. Run multiple simulations with different parameters
2. Compare results across parameter sweeps
3. Generate summary statistics
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from smolscells import SmolScellsSimulator
from smolscells.analysis import FitnessAnalyzer, TreeComparator


def run_single_experiment(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single experiment with given parameters"""
    
    # Create simulator
    sim = SmolScellsSimulator(config=params['config'], run_name=params['run_name'])
    
    # Run simulation
    results = sim.run_complete_simulation()
    
    # Analyze results
    recorded_gt = results['recorded_gt_tree']
    single_cell_tree = results['trees'].get('single_cell')
    
    # Calculate metrics
    metrics = {
        'run_name': params['run_name'],
        'sc_rate': params['config']['sc_sampling_rate'],
        'sm_rate': params['config']['sm_sampling_rate'],
        'dropout_rate': params['config'].get('dropout', {}).get('rate', 0),
        'num_intbc': params['config']['tree_config']['k'],
    }
    
    # Tree reconstruction accuracy
    if single_cell_tree:
        comparator = TreeComparator(recorded_gt)
        tree_metrics = comparator.compare_tree(single_cell_tree)
        metrics.update({
            'robinson_foulds': tree_metrics.get('robinson_foulds', 1.0),
            'triplet_accuracy': tree_metrics.get('triplet_accuracy', 0.0),
            'common_leaves': tree_metrics.get('common_leaves', 0),
        })
    
    # Fitness detection
    analyzer = FitnessAnalyzer(results['trees'])
    fitness_comparison = analyzer.compare_fitness_detection()
    
    if 'single_cell' in fitness_comparison:
        metrics['sc_fitness_corr'] = fitness_comparison['single_cell'].get('correlation', 0)
    
    # Average bulk fitness correlation
    bulk_corrs = []
    for tree_name, tree_metrics in fitness_comparison.items():
        if 'single_molecule' in tree_name and 'correlation' in tree_metrics:
            bulk_corrs.append(tree_metrics['correlation'])
    
    if bulk_corrs:
        metrics['bulk_fitness_corr'] = np.mean(bulk_corrs)
    
    # Enhanced fitness correlation
    enhanced_scores = analyzer.aggregate_bulk_fitness()
    gt_lbi = analyzer.calculate_lbi('recorded_ground_truth')
    
    common_nodes = set(gt_lbi.keys()) & set(enhanced_scores.keys())
    if common_nodes:
        gt_values = [gt_lbi[node] for node in common_nodes]
        enhanced_values = [enhanced_scores[node] for node in common_nodes]
        metrics['enhanced_fitness_corr'] = np.corrcoef(gt_values, enhanced_values)[0, 1]
    
    return metrics


def generate_parameter_sweep() -> List[Dict[str, Any]]:
    """Generate parameter combinations for sweep"""
    
    # Base configuration
    with open('configs/default.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    experiments = []
    exp_id = 0
    
    # Vary sampling rates
    sc_rates = [0.05, 0.1, 0.2, 0.3]
    sm_rates = [0.2, 0.3, 0.5, 0.7]
    
    for sc_rate in sc_rates:
        for sm_rate in sm_rates:
            config = base_config.copy()
            config['sc_sampling_rate'] = sc_rate
            config['sm_sampling_rate'] = sm_rate
            
            experiments.append({
                'config': config,
                'run_name': f'exp_{exp_id:03d}_sc{int(sc_rate*100)}_sm{int(sm_rate*100)}'
            })
            exp_id += 1
    
    # Vary dropout rates
    dropout_rates = [0.0, 0.1, 0.2, 0.3]
    for dropout_rate in dropout_rates:
        config = base_config.copy()
        config['dropout'] = {
            'enabled': dropout_rate > 0,
            'rate': dropout_rate,
            'pattern': 'uniform'
        }
        
        experiments.append({
            'config': config,
            'run_name': f'exp_{exp_id:03d}_dropout{int(dropout_rate*100)}'
        })
        exp_id += 1
    
    # Vary number of integration barcodes
    num_intbcs = [2, 4, 8, 16]
    for k in num_intbcs:
        config = base_config.copy()
        config['tree_config']['k'] = k
        
        experiments.append({
            'config': config,
            'run_name': f'exp_{exp_id:03d}_intbc{k}'
        })
        exp_id += 1
    
    return experiments


def run_batch_experiments(
    experiments: List[Dict[str, Any]],
    n_workers: int = 4,
    output_dir: str = "batch_results"
) -> pd.DataFrame:
    """Run batch experiments in parallel"""
    
    print(f"Running {len(experiments)} experiments with {n_workers} workers...")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all experiments
        future_to_exp = {
            executor.submit(run_single_experiment, exp): exp 
            for exp in experiments
        }
        
        # Process completed experiments
        with tqdm(total=len(experiments)) as pbar:
            for future in as_completed(future_to_exp):
                exp = future_to_exp[future]
                try:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                except Exception as e:
                    print(f"Experiment {exp['run_name']} failed: {e}")
                    pbar.update(1)
    
    # Create results dataframe
    df = pd.DataFrame(results)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    csv_path = output_path / "batch_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    return df


def analyze_batch_results(df: pd.DataFrame):
    """Analyze and summarize batch experiment results"""
    
    print("\n" + "="*60)
    print("Batch Experiment Summary")
    print("="*60)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(df[['robinson_foulds', 'triplet_accuracy', 'sc_fitness_corr', 
              'bulk_fitness_corr', 'enhanced_fitness_corr']].describe())
    
    # Group by sampling rates
    print("\nEffect of Sampling Rates:")
    grouped = df.groupby(['sc_rate', 'sm_rate']).agg({
        'robinson_foulds': 'mean',
        'triplet_accuracy': 'mean',
        'enhanced_fitness_corr': 'mean'
    }).round(3)
    print(grouped)
    
    # Effect of dropout
    print("\nEffect of Dropout Rate:")
    dropout_grouped = df.groupby('dropout_rate').agg({
        'robinson_foulds': 'mean',
        'triplet_accuracy': 'mean',
        'sc_fitness_corr': 'mean',
        'enhanced_fitness_corr': 'mean'
    }).round(3)
    print(dropout_grouped)
    
    # Effect of integration barcodes
    print("\nEffect of Number of Integration Barcodes:")
    intbc_grouped = df.groupby('num_intbc').agg({
        'robinson_foulds': 'mean',
        'bulk_fitness_corr': 'mean',
        'enhanced_fitness_corr': 'mean'
    }).round(3)
    print(intbc_grouped)
    
    # Improvement from bulk integration
    df['fitness_improvement'] = (
        (df['enhanced_fitness_corr'] - df['sc_fitness_corr']) / 
        df['sc_fitness_corr'].abs() * 100
    )
    
    print(f"\nAverage Fitness Detection Improvement: {df['fitness_improvement'].mean():.1f}%")
    print(f"Maximum Improvement: {df['fitness_improvement'].max():.1f}%")
    
    # Best configuration
    best_idx = df['enhanced_fitness_corr'].idxmax()
    best_config = df.loc[best_idx]
    print(f"\nBest Configuration:")
    print(f"  SC rate: {best_config['sc_rate']:.0%}")
    print(f"  SM rate: {best_config['sm_rate']:.0%}")
    print(f"  Dropout: {best_config['dropout_rate']:.0%}")
    print(f"  Integration barcodes: {best_config['num_intbc']}")
    print(f"  Enhanced fitness correlation: {best_config['enhanced_fitness_corr']:.3f}")


def main():
    """Run batch experiments"""
    
    # Generate parameter sweep
    experiments = generate_parameter_sweep()
    print(f"Generated {len(experiments)} experiment configurations")
    
    # Run experiments
    df = run_batch_experiments(experiments, n_workers=4)
    
    # Analyze results
    analyze_batch_results(df)
    
    return df


if __name__ == "__main__":
    results_df = main()