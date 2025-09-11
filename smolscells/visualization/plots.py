"""Visualization tools for SmolScells"""

import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import cassiopeia as cass


class TreeVisualizer:
    """Visualize trees and simulation results"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize tree visualizer
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
    
    def plot_tree(
        self,
        tree: cass.data.CassiopeiaTree,
        title: str = "Tree",
        figsize: tuple = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot a tree using Cassiopeia's built-in plotting
        
        Args:
            tree: CassiopeiaTree to plot
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        try:
            import cassiopeia.plotting as cplot
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Use Cassiopeia's plotting if available
            cplot.plot_matplotlib(tree, ax=ax)
            ax.set_title(title)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            print("Cassiopeia plotting not available")
    
    def plot_comparison(
        self,
        metrics: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """
        Plot comparison metrics
        
        Args:
            metrics: Dictionary of metrics to plot
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract tree names and values
        trees = list(metrics.keys())
        
        # Plot different metrics
        metric_types = set()
        for tree_metrics in metrics.values():
            if isinstance(tree_metrics, dict):
                metric_types.update(tree_metrics.keys())
        
        # Create bar plot for numeric metrics
        numeric_metrics = ['correlation', 'robinson_foulds', 'triplet_accuracy']
        available_metrics = [m for m in numeric_metrics if m in metric_types]
        
        if available_metrics:
            x_pos = range(len(trees))
            width = 0.8 / len(available_metrics)
            
            for i, metric in enumerate(available_metrics):
                values = []
                for tree in trees:
                    if isinstance(metrics[tree], dict) and metric in metrics[tree]:
                        values.append(metrics[tree][metric])
                    else:
                        values.append(0)
                
                x_offset = (i - len(available_metrics)/2) * width
                ax.bar([p + x_offset for p in x_pos], values, width, label=metric)
            
            ax.set_xlabel('Tree')
            ax.set_ylabel('Metric Value')
            ax.set_title('Tree Reconstruction Metrics')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(trees, rotation=45, ha='right')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()