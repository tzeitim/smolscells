from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator import Cas9LineageTracingDataSimulator, BirthDeathFitnessSimulator
import numpy as np
from convexml import convexml
from pathlib import Path
from .sampling import mutually_exclusive_sampling, split_single_molecule_data, apply_single_cell_dropout, compute_single_cell_dropout
import logging

logger = logging.getLogger(__name__)

def return_default_conf_gt():
    return {
        'birth_waiting_distribution': lambda scale: np.random.exponential(1/scale),
        'initial_birth_scale': 2,
        'num_extant': 1000
    }

def return_defalt_conf_exp(missing_data=False):
    return {
        'number_of_cassettes': 4,
        'size_of_cassette': 10,
        'mutation_rate': 0.1,
        'state_generating_distribution': lambda: np.random.exponential(1e-5),
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

    def __init__(self, conf_gt:Path|dict|None=None, conf_xp:Path|dict|None=None, conf_dropout:Path|dict|None=None,  missing_data=False):
        if conf_gt is None:
            self.conf_gt = return_default_conf_gt()

        if conf_xp is None:
            self.conf_exp = return_defalt_conf_exp(missing_data)

        if conf_dropout is None:
            self.conf_dropout = return_default_conf_dropout(missing_data)
        
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
        
        
        # compute dropout stats
        cmultipliers, intdbrates =  compute_single_cell_dropout(character_matrix=self.sc_matrix, dropout_config={})

        # populate single cell objects
        masked, self.sc_matrix_mask = apply_single_cell_dropout(
            character_matrix=self.sc_matrix,
            cell_multipliers=cmultipliers, 
            intbc_dropout_rates=intdbrates,
            sites_per_intbc=self.conf_exp['size_of_cassette']
            )


    def simulate(self, sc_rate=0.1, sm_rate=0.5):
        self.simulate_gt()
        self.simulate_recording()
        self.sample_fractions(sc_rate=sc_rate, sm_rate=sm_rate)

        
