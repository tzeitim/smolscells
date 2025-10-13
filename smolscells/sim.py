from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import CassiopeiaTreeError
from cassiopeia.simulator import Cas9LineageTracingDataSimulator, BirthDeathFitnessSimulator
import numpy as np
from convexml import convexml
from pathlib import Path
import logging

from .sampling import mutually_exclusive_sampling, split_single_molecule_data, apply_single_cell_dropout, compute_single_cell_dropout
from . import solvers

logger = logging.getLogger(__name__)

def _birth_waiting_distribution(scale):
    """Default birth waiting distribution for ground truth simulation."""
    return np.random.exponential(1/scale)

def _state_generating_distribution():
    """Default state generating distribution for experimental simulation."""
    return np.random.exponential(1e-5)

def return_default_conf_gt():
    return {
        'birth_waiting_distribution': _birth_waiting_distribution,
        'initial_birth_scale': 2,
        'num_extant': 1000
    }

def return_defalt_conf_exp(missing_data=False):
    return {
        'number_of_cassettes': 4,
        'size_of_cassette': 10,
        'mutation_rate': 0.1,
        'state_generating_distribution': _state_generating_distribution,
        'number_of_states': 50,
        'state_priors': None,
        'heritable_silencing_rate': 0, #9e-4 if missing_data else 0,
        'stochastic_silencing_rate': 0, #0.1 if missing_data else 0,
        'heritable_missing_data_state': -1,
        'stochastic_missing_data_state': -1,
    }

def return_default_conf_dropout(missing_data=False):
    #pattern == 'uniform':'per_intbc': 'per_cell'
    return {
            'enabled': missing_data,
            'pattern': 'per_intbc',
            'intbc_variability': 0.1,
            'cell_variability': 0.2,
            }
def return_default_conf_solver():
    return solvers.get_solver_class('nj')
        
#
#solver = cas.solver.NeighborJoiningSolver(
#    dissimilarity_function=cas.solver.dissimilarity.weighted_hamming_distance,
#    add_root=True
#    )

#solver.solve(exp_tree, collapse_mutationless_edges=True)

class Simsmolscells():
    conf_gt:None|dict
    conf_exp:None|dict
    gt_tree:None|CassiopeiaTree=None
    sm_trees:dict={}

    def __init__(self, conf_gt:Path|dict|None=None, 
                 conf_xp:Path|dict|None=None, 
                 conf_dropout:Path|dict|None=None,  
                 conf_solver:Path|dict|None=None,  
                 missing_data=False):
        if conf_gt is None:
            self.conf_gt = return_default_conf_gt()

        if conf_xp is None:
            self.conf_exp = return_defalt_conf_exp(missing_data)

        if conf_dropout is None:
            self.conf_dropout = return_default_conf_dropout(missing_data)

        if conf_solver is None:
            self.solver = return_default_conf_solver()
        
    def simulate_gt(self):
        if self.gt_tree is None:
            simulator = BirthDeathFitnessSimulator(**self.conf_gt)
            self.gt_tree = simulator.simulate_tree()
            logger.info("Simulated GT Tree")
        else:
            logger.warning("GT tree already exists")

    def simulate_recording(self):
        exp_simulator = Cas9LineageTracingDataSimulator(**self.conf_exp)
        exp_simulator.overlay_data(self.gt_tree)

        self.exp_tree = CassiopeiaTree(
                character_matrix = self.gt_tree.character_matrix, 
                missing_state_indicator = -1)


    def sample_fractions(self, sc_rate=0.1, sm_rate=0.5):
        """ samples bulk and single-cell fractions from the GT tree 
        """
        sc_matrix, sm_matrix, sc_cell_ids, sm_cell_ids = mutually_exclusive_sampling(
                character_matrix= self.exp_tree.character_matrix,
                sc_rate=sc_rate, 
                sm_rate=sm_rate
                ) 
        self.sc_matrix = sc_matrix
        self.sm_matrix = sm_matrix
        self.sc_cell_ids = sc_cell_ids
        self.sm_cell_ids = sm_cell_ids


        # populate single molecule objects
        self.sm_mats = split_single_molecule_data(
                character_matrix=self.sm_matrix,
                sm_cell_ids=self.sm_cell_ids,
                **self.conf_exp)
       

        for k,v in self.sm_mats.items():
            self.sm_trees[k] = CassiopeiaTree(character_matrix=v) 
        logger.info(f"Created single-molecule trees {len(self.sm_trees)} total")

        # compute dropout stats
        cmultipliers, intdbrates =  compute_single_cell_dropout(character_matrix=self.sc_matrix, dropout_config={})

        # populate single cell objects
        self.sc_matrix_masked, self.sc_matrix_mask = apply_single_cell_dropout(
            character_matrix=self.sc_matrix,
            cell_multipliers=cmultipliers, 
            intbc_dropout_rates=intdbrates,
            sites_per_intbc=self.conf_exp["size_of_cassette"]
            )

        self.sc_tree = CassiopeiaTree(character_matrix=self.sc_matrix_masked)
        logger.info("Created single-cell tree")

        # solve trees

    def solve_fractions(self, fraction:str="all", solver:str|None=None):
        if solver is not None:
            self.solver = solvers.get_solver_class(solver)

        match fraction:
            case "all":
                logger.info("Solving all sub-trees")
                self.solver.solve(self.sc_tree)
                logger.info("Solved single-cell")

                for k,v in self.sm_trees.items():
                    self.solver.solve(v)
                    logger.info(f"Solved {k}")
            case "sc":
                self.solver.solve(self.sc_tree)


    def simulate(self, sc_rate=0.1, sm_rate=0.5):
        self.simulate_gt()
        self.simulate_recording()
        self.sample_fractions(sc_rate=sc_rate, sm_rate=sm_rate)

        self.solve_fractions()

    def save(self, filepath):
        """Save the simulation object to a file using pickle.

        Parameters
        ----------
        filepath : str or Path
            Path to save the pickled object
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load a simulation object from a pickle file.

        Parameters
        ----------
        filepath : str or Path
            Path to the pickled object

        Returns
        -------
        Simsmolscells
            The loaded simulation object
        """
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)


