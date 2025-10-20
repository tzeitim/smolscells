"""
SimulatedLineageForest - Simulation workflow for lineage tracing experiments.

This module provides a dedicated class for simulating lineage tracing data,
keeping simulation concerns separate from the observational LineageForest class.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import yaml
from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator import Cas9LineageTracingDataSimulator, BirthDeathFitnessSimulator

from .lineage_treedata import LineageForest as LFold
from .lineage_forest import LineageForest
from .sampling import (
    mutually_exclusive_sampling,
    split_single_molecule_data,
    apply_single_cell_dropout,
    compute_single_cell_dropout,
)
from . import solvers


# ═══════════════════════════════════════════════════════════
# DEFAULT CONFIGURATIONS
# ═══════════════════════════════════════════════════════════


def _birth_waiting_distribution(scale):
    """Default birth waiting distribution for ground truth simulation."""
    return np.random.exponential(1 / scale)


def _state_generating_distribution():
    """Default state generating distribution for experimental simulation."""
    return np.random.exponential(1e-5)


def return_default_conf_gt(random_seed:int|None=None):
    """Return default ground truth simulation configuration."""
    if random_seed is None:
        random_seed = 1717
    else:
        assert isinstance(random_seed, int), "Parameter random_seed must be an integer"

    return {
        "birth_waiting_distribution": _birth_waiting_distribution,
        "initial_birth_scale": 2,
        "num_extant": 1000,
        "random_seed": random_seed,
    }


def return_default_conf_exp(missing_data=False):
    """Return default experimental recording configuration."""
    return {
        "number_of_cassettes": 4,
        "size_of_cassette": 10,
        "mutation_rate": 0.1,
        "state_generating_distribution": _state_generating_distribution,
        "number_of_states": 50,
        "state_priors": None,
        "collapse_sites_on_cassette": False,
        "heritable_silencing_rate": 0,
        "stochastic_silencing_rate": 0,
        "heritable_missing_data_state": -1,
        "stochastic_missing_data_state": -1,
    }


def return_default_conf_dropout(missing_data=False):
    """Return default dropout configuration."""
    return {
        "enabled": missing_data,
        "pattern": "per_intbc",
        "intbc_variability": 0.1,
        "cell_variability": 0.2,
    }


def return_default_conf_solver():
    """Return default solver."""
    return solvers.get_solver_class("nj")


# ═══════════════════════════════════════════════════════════
# SIMULATED LINEAGE FOREST
# ═══════════════════════════════════════════════════════════

class SimulatedLineageForest:
    """
    Complete simulation workflow and data container for lineage tracing.

    Separates simulation concerns (ground truth, experimental artifacts) from
    observational data (LineageForest). Contains all simulation metadata and
    populates a LineageForest instance with observational layer data.

    Attributes
    ----------
    conf_gt : dict
        Ground truth simulation configuration
    conf_exp : dict
        Experimental recording configuration
    conf_dropout : dict
        Dropout configuration
    gt_tree : CassiopeiaTree | None
        Ground truth tree
    exp_tree : CassiopeiaTree | None
        Experimental tree with character matrix
    sc_matrix : pd.DataFrame | None
        Single-cell character matrix (pre-dropout)
    sc_matrix_masked : pd.DataFrame | None
        Single-cell character matrix (post-dropout)
    sc_matrix_mask : pd.DataFrame | None
        Dropout mask for single-cell data
    sm_matrix : pd.DataFrame | None
        Single-molecule character matrix
    sc_cell_ids : list | None
        Cell IDs in single-cell fraction
    sm_cell_ids : list | None
        Cell IDs in single-molecule fraction
    sm_mats : dict | None
        Single-molecule matrices by intBC
    lf : LineageForest
        Observational data (solved trees)
    solver : Any
        Cassiopeia solver instance

    Examples
    --------
    >>> # Create simulation
    >>> sim = SimulatedLineageForest()
    >>> sim.simulate(sc_rate=0.1, sm_rate=0.5)
    >>>
    >>> # Access observational data
    >>> lf = sim.lf
    >>> lf.plot_tree(tree='sc')
    >>>
    >>> # Compare to ground truth
    >>> print(sim.gt_tree)
    >>>
    >>> # From config files
    >>> sim = SimulatedLineageForest(
    ...     conf_gt='configs/gt.yaml',
    ...     conf_exp='configs/exp.yaml'
    ... )
    >>> # Or from master config
    >>> sim = SimulatedLineageForest.from_config('configs/master.yaml')
    """

    @staticmethod
    def _load_config(config: Path | dict | None, default: dict | None = None) -> dict:
        """Load configuration from file or dict.

        Parameters
        ----------
        config
            Configuration as dict or path to YAML file
        default
            Default configuration if config is None

        Returns
        -------
        Configuration dict
        """
        if config is None:
            return default if default is not None else {}

        if isinstance(config, dict):
            return config

        # Load from file
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            loaded = yaml.safe_load(f)

        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a dict, got {type(loaded)}")

        return loaded

    def __init__(
        self,
        conf_gt: Path | dict | None = None,
        conf_exp: Path | dict | None = None,
        conf_dropout: Path | dict | None = None,
        conf_solver: Path | dict | None = None,
        missing_data: bool = False,
    ):
        """
        Initialize SimulatedLineageForest.

        Parameters
        ----------
        conf_gt
            Ground truth config (dict or path to YAML file)
        conf_exp
            Experimental recording config (dict or path to YAML file)
        conf_dropout
            Dropout config (dict or path to YAML file)
        conf_solver
            Solver config (dict or path to YAML file)
        missing_data
            Whether to enable missing data in default configs
        """
        # Load and store configurations
        self.conf_gt = self._load_config(conf_gt, return_default_conf_gt())
        self.conf_exp = self._load_config(conf_exp, return_default_conf_exp(missing_data))
        self.conf_dropout = self._load_config(
            conf_dropout, return_default_conf_dropout(missing_data)
        )
        self.solver = (
            self._load_config(conf_solver, None)
            if conf_solver is not None
            else return_default_conf_solver()
        )

        # Ground truth
        self.gt_tree: CassiopeiaTree | None = None
        self.exp_tree: CassiopeiaTree | None = None

        # Sampling artifacts
        self.sc_matrix: pd.DataFrame | None = None
        self.sc_matrix_masked: pd.DataFrame | None = None
        self.sc_matrix_mask: pd.DataFrame | None = None
        self.sm_matrix: pd.DataFrame | None = None
        self.sc_cell_ids: list | None = None
        self.sm_cell_ids: list | None = None
        self.sm_mats: dict | None = None

        # Observational data container
        #self.lf = LineageForest(alignment="subset")
        self.lf = LineageForest()

        # CassiopeiaTree objects for solving
        self._sc_tree: CassiopeiaTree | None = None
        self._sm_trees: dict[str, CassiopeiaTree] = {}

    def __repr__(self) -> str:
        gt_tree = self.gt_tree 
        exp_tree = self.lf.get_tree('exp_tree', as_cassiopeia=True)
          
        # Simulation status
        has_gt =  gt_tree is not None
        has_exp = exp_tree is not None
        has_samples = self.sc_cell_ids is not None
        has_lf = self.lf is not None and self.lf.n_trees > 0

        if not has_gt:
            status = "Not simulated"
        elif not has_exp:
            status = "Ground truth only"
        elif not has_samples:
            status = "Recording done"
        elif not has_lf:
            status = "Sampled (not solved)"
        else:
            status = "Complete"

        lines = [f"SimulatedLineageForest [{status}]"]

        # Ground truth info
        if has_gt:
            n_cells = gt_tree.n_cell if hasattr(gt_tree, 'n_cell') else len(gt_tree.leaves)
            lines.append(f"  Ground truth: {n_cells} cells")

        # Sampling info
        if has_samples:
            n_sc = len(self.sc_cell_ids) if self.sc_cell_ids is not None else 0
            n_sm = len(self.sm_cell_ids) if self.sm_cell_ids is not None else 0
            n_intbcs = len(self.sm_mats) if self.sm_mats is not None else 0
            lines.append(f"  Sampled: {n_sc} SC cells, {n_sm} SM cells ({n_intbcs} intBCs)")

        # LineageForest info
        if has_lf and self.lf.n_trees > 0:
            n_solved = len([k for k in self.lf.tree_keys if k not in ('exp_tree')])
            lines.append(f"  Solved trees: {n_solved} ({', '.join(self.lf.tree_keys)})")

        # Config summary
        if self.conf_gt:
            n_extant = self.conf_gt.get('num_extant', '?')
            lines.append(f"  Config: {n_extant} target cells")

        return '\n'.join(lines)

    @classmethod
    def from_config(
        cls,
        config_file: Path | str,
        missing_data: bool = False
    ) -> 'SimulatedLineageForest':
        """Create from master configuration file.

        Parameters
        ----------
        config_file
            Path to master YAML file containing all configs
        missing_data
            Whether to enable missing data in default configs

        Returns
        -------
        SimulatedLineageForest instance

        Examples
        --------
        Master config file structure:

        .. code-block:: yaml

            conf_gt:
              num_extant: 1000
              initial_birth_scale: 2
            conf_exp:
              number_of_cassettes: 4
              mutation_rate: 0.1
            conf_dropout:
              enabled: true
              pattern: "per_intbc"

        >>> sim = SimulatedLineageForest.from_config('configs/master.yaml')
        """
        config_path = Path(config_file)
        with open(config_path, 'r') as f:
            master = yaml.safe_load(f)

        if not isinstance(master, dict):
            raise ValueError("Master config must be a dict")

        return cls(
            conf_gt=master.get('conf_gt'),
            conf_exp=master.get('conf_exp'),
            conf_dropout=master.get('conf_dropout'),
            conf_solver=master.get('conf_solver'),
            missing_data=missing_data
        )

    def simulate_gt(self) -> None:
        """Simulate ground truth tree."""
        if self.gt_tree is None:
            simulator = BirthDeathFitnessSimulator(**self.conf_gt)

            #self.lf.add_tree(tree = simulator.simulate_tree(), tree_key="gt")
            self.gt_tree = simulator.simulate_tree() 

            logger.info(f"Simulated GT tree with {self.gt_tree.n_cell} cells")
        else:
            logger.warning("GT tree already exists")

    def simulate_recording(self) -> None:
        """Simulate lineage recording on ground truth."""
        gt_tree = self.gt_tree
        if gt_tree is None:
            raise ValueError("Must simulate ground truth first (call simulate_gt)")

        exp_simulator = Cas9LineageTracingDataSimulator(**self.conf_exp)
        exp_simulator.overlay_data(gt_tree)

        self.lf.add_tree(tree_key="exp_tree", 
                         tree =CassiopeiaTree(
                                character_matrix=gt_tree.character_matrix, 
                                missing_state_indicator=-1
                                )
         )
 
        logger.info("Simulated lineage recording")


    def sample_fractions(self, sc_rate: float = 0.1, sm_rate: float = 0.5) -> None:
        """
        Sample single-cell and single-molecule fractions.

        Parameters
        ----------
        sc_rate
            Fraction of cells to sample for single-cell
        sm_rate
            Fraction of cells to sample for single-molecule
        """
        if self.lf.trees['exp_tree'] is None:
            raise ValueError("Must simulate recording first (call simulate_recording)")

        # Sample from experimental tree
        sc_matrix, sm_matrix, sc_cell_ids, sm_cell_ids = mutually_exclusive_sampling(
            character_matrix=self.lf.get_tree('exp_tree').character_matrix,
            sc_rate=sc_rate,
            sm_rate=sm_rate,
        )

        self.sc_matrix = sc_matrix
        self.sm_matrix = sm_matrix
        self.sc_cell_ids = sc_cell_ids
        self.sm_cell_ids = sm_cell_ids

        # Split single-molecule data by intBC
        self.sm_mats = split_single_molecule_data(
            character_matrix=self.sm_matrix, sm_cell_ids=self.sm_cell_ids, **self.conf_exp
        )

        # Create CassiopeiaTree objects for single-molecule data
        for matrix in self.sm_mats.values():
            self.lf.add_sm_tree(tree=CassiopeiaTree(character_matrix=matrix))

        logger.info(f"Created {len(self.lf.smtrees_keys)} single-molecule trees")

        # Apply dropout to single-cell data if enabled
        if self.conf_dropout.get('enabled', True):
            # Compute dropout stats
            cmultipliers, intdbrates = compute_single_cell_dropout(
                character_matrix=self.sc_matrix,
                dropout_config=self.conf_dropout,
                number_of_cassettes=self.conf_exp.get("number_of_cassettes")
            )

            # Apply dropout to single-cell data
            self.sc_matrix_masked, self.sc_matrix_mask = apply_single_cell_dropout(
                character_matrix=self.sc_matrix,
                cell_multipliers=cmultipliers,
                intbc_dropout_rates=intdbrates,
                sites_per_intbc=self.conf_exp["size_of_cassette"],
            )
            logger.info(f"Applied dropout to single-cell data")
        else:
            # No dropout - use original matrix
            self.sc_matrix_masked = self.sc_matrix.copy()
            self.sc_matrix_mask = None
            logger.info(f"Dropout disabled - using original single-cell data")

        # Create CassiopeiaTree object for single-cell data
        self.lf.add_sc_tree(tree = CassiopeiaTree(character_matrix=self.sc_matrix_masked))

        logger.info(
            f"Sampled {len(self.sc_cell_ids)} cells (SC) and "
            f"{len(self.sm_cell_ids)} cells with molecules (SM)"
        )

    def solve_fractions(
        self,
        fraction: str = "all",
        solver: str | None = None,
        collapse_mutationless_edges: bool = True,
    ) -> None:
        """
        Solve (reconstruct) trees for sampled fractions.

        Populates the lf with solved trees and character matrices.

        Parameters
        ----------
        fraction
            Which fraction to solve ('all', 'sc', 'sm', 'exp_tree')
        solver
            Solver name (if None, uses self.solver)
        collapse_mutationless_edges
            Whether to collapse mutationless edges
        """
        if solver is not None:
            self.solver = solvers.get_solver_class(solver)

        # Solve single-cell tree
        if fraction in ['all', 'sc']:
            logger.info("Solving single-cell tree")
            self.solver.solve(self.lf.get_sc_tree(), collapse_mutationless_edges=collapse_mutationless_edges)
            logger.info("Added single-cell tree")

        # Solve single-molecule trees
        if fraction in ['all', 'sm']:
            for intbc_key, sm_tree in self.lf.smtrees_items:
                logger.info(f"Solving single-molecule tree {intbc_key}")
                self.solver.solve(sm_tree, collapse_mutationless_edges=collapse_mutationless_edges)
                logger.info(f"Solved and added SM tree {intbc_key}")

        # Solve experimental tree (full tree before sampling)
        if fraction in ['exp_tree']:
            logger.info("Solving experimental tree")
            self.solver.solve(self.lf.get_tree('exp_tree'), collapse_mutationless_edges=collapse_mutationless_edges)
            logger.info("Solved and added experimental tree")

    def simulate(
        self,
        sc_rate: float = 0.1,
        sm_rate: float = 0.5,
        solver: str | None = None,
        collapse_mutationless_edges: bool = True,
    ) -> None:
        """
        Run complete simulation workflow.

        Executes: ground truth → recording → sampling → solving

        Parameters
        ----------
        sc_rate
            Fraction of cells to sample for single-cell
        sm_rate
            Fraction of cells to sample for single-molecule
        solver
            Solver name (if None, uses default)
        collapse_mutationless_edges
            Whether to collapse mutationless edges
        """
        self.lf.reset()
        self.simulate_gt()
        self.simulate_recording()
        self.lf.initialize_tdatas()
        self.sample_fractions(sc_rate=sc_rate, sm_rate=sm_rate)
        self.solve_fractions(solver=solver, collapse_mutationless_edges=collapse_mutationless_edges)
        self.lf._sync_tdatas_from_trees()
        print(self.lf)

    # ═══════════════════════════════════════════════════════════
    # I/O
    # ═══════════════════════════════════════════════════════════

    def save(self, filepath: Path | str) -> None:
        """
        Save complete simulation object.

        Parameters
        ----------
        filepath
            Path to save the pickled object
        """
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved simulation to {filepath}")

    @staticmethod
    def load(filepath: Path | str) -> SimulatedLineageForest:
        """
        Load simulation object from pickle file.

        Parameters
        ----------
        filepath
            Path to the pickled object

        Returns
        -------
        The loaded simulation object
        """
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════


def simulate_lineage_experiment(
    conf_gt: dict[str, Any] | None = None,
    conf_exp: dict[str, Any] | None = None,
    conf_dropout: dict[str, Any] | None = None,
    sc_rate: float = 0.1,
    sm_rate: float = 0.5,
    solver: str = "nj",
    return_simulation: bool = False,
) -> LineageForest | SimulatedLineageForest:
    """
    One-shot simulation function.

    Creates a complete simulated lineage tracing experiment with ground truth,
    recording, sampling, and tree reconstruction.

    Parameters
    ----------
    conf_gt
        Ground truth simulation configuration
    conf_exp
        Recording simulation configuration
    conf_dropout
        Dropout configuration
    sc_rate
        Fraction of cells to sample for single-cell
    sm_rate
        Fraction of cells to sample for single-molecule
    solver
        Solver name (e.g., 'nj', 'upgma')
    return_simulation
        If True, return SimulatedLineageForest; if False, return LineageForest

    Returns
    -------
    LineageForest (if return_simulation=False) or SimulatedLineageForest (if True)

    Examples
    --------
    >>> # Get observational data only
    >>> lf = simulate_lineage_experiment(sc_rate=0.1, sm_rate=0.5)
    >>> lf.plot_tree(tree='sc')
    >>>
    >>> # Get full simulation object
    >>> sim = simulate_lineage_experiment(sc_rate=0.1, sm_rate=0.5, return_simulation=True)
    >>> print(sim.gt_tree)
    >>> lf = sim.lf
    """
    sim = SimulatedLineageForest(conf_gt=conf_gt, conf_exp=conf_exp, conf_dropout=conf_dropout)
    sim.simulate(sc_rate=sc_rate, sm_rate=sm_rate, solver=solver)

    if return_simulation:
        return sim
    else:
        return sim.lf



