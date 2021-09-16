"""
Run a pre-solver on the BBOB benchmark suite.
"""
import numpy as np
import cocoex  # only experimentation module
import sys
import logging
import os
import random
import click
import shutil

from scipy.optimize import minimize
# from scipy.optimize import fmin_slsqp
from smac.facade.func_facade import fmin_smac

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LogFunction():
    """
    Log a solution x and its objective value f(x) found by a pre-solver

    Attributes
    ----------
    fun : callable object
        A function to be minimized.
    sample_data_file_path: path
        A directory path.
    max_fevals: int
        The maximum number of function evaluations.
    """
    def __init__(self, fun, sample_data_file_path, max_fevals):
        self.fun = fun
        self.max_fevals = max_fevals
        self.fevals = 0
        self.sample_data_file_path = sample_data_file_path
        fh = open(self.sample_data_file_path, 'w')
        fh.close()
        
    def eval(self, x):
        # SLSQP rarely generates a solution that slightly violates the lower bound or upper bound like x = -5.000000000000001. I believe that the following clipping operation does not influence the behavior of SLSQP.
        x = np.clip(x, -5, 5)        
        obj_val = self.fun(x)

        if self.fevals < self.max_fevals:
            data_str = '{},'.format(obj_val)                
            data_str += ','.join([str(y) for y in x])
            with open(self.sample_data_file_path, 'a') as fh:
                fh.write(data_str + '\n')
            self.fevals += 1
                
        return obj_val
    
def run_presolver(bbob_suite='bbob', budget_multiplier=50, presolver='slsqp', sample_dir_path='./presolver_data', sample_id=0, run_iids=[1,2,3,4,5], run_dims=[2,3,5,10]):
    """
    Run a pre-solver on the BBOB functions.

    This function is based on https://github.com/numbbo/coco/blob/master/code-experiments/build/python/example_experiment_for_beginners.py

    Parameters
    ----------
    bbob_suite: string
        The name of a bbob suite (bbob, bbob-noisy, bbob-biobj, bbob-largescale, bbob-mixint, bbob-biobj-mixint)
    budget_multiplier : int
        A multiplier x for determining a budget of function evaluations (= x * dimension).
    presolver: string
        A pre-solver.
    sample_dir_path: path
        A directory path.
    sample_id: int
        Run ID.
    run_iids: list 
        Instance IDs to run. 
    run_dims: list
        A list of dimensions to run.

    Returns
    -------    

    """    
    sample_dir_path = os.path.join('./presolver_data', '{}_multiplier{}_sid{}'.format(presolver, budget_multiplier, sample_id))
    os.makedirs(sample_dir_path, exist_ok=True)
        
    ### input
    output_folder = 'tmp'

    ### prepare
    suite = cocoex.Suite(bbob_suite, "", "")
    observer = cocoex.Observer(bbob_suite, "result_folder: " + output_folder)
    minimal_print = cocoex.utilities.MiniPrint()

    # This is used only when naming a file with a different instance ID insted of the original instance ID.    
    count_instance_id = 1
    
    ### go
    for problem in suite:  # this loop will take several minutes or longer
        problem.observe_with(observer)  # generates the data for cocopp post-processing

        if problem.dimension not in run_dims:
            minimal_print(problem, final=problem.index == len(suite) - 1)   
            continue
        
        # COCO automatically perform experiments on 2, 3, 5, 10, 20, and 40 for 'bbob'. The following terminates experiments on 'stop_dim' dimension
        if bbob_suite == 'bbob' and problem.dimension > max(run_dims):
            break        
                
        fun_id = int(problem.info.split('_f')[1].split('_')[0])
        instance_id = int(problem.info.split('_i')[1].split('_')[0])
        #instance_id = count_instance_id

        if instance_id not in run_iids:
            minimal_print(problem, final=problem.index == len(suite) - 1)
            continue
        
        # Recode each pair of x and f(x) in a csv file
        presolver_data_file_path = os.path.join(sample_dir_path, 'presolver_x_f_data_{}_f{}_DIM{}_i{}.csv'.format(bbob_suite, fun_id, problem.dimension, instance_id))
        
        # The inital search point is [0, ..., 0].
        x0 = problem.initial_solution
        bbob_bounds = [(-5, 5)] * problem.dimension
        # lbounds = np.full(problem.dimension, -5)
        # ubounds = np.full(problem.dimension, 5)
        lbounds = [-5] * problem.dimension
        ubounds = [5] * problem.dimension
        
        my_fun = LogFunction(problem, presolver_data_file_path, max_fevals=budget_multiplier*problem.dimension)

        if 'slsqp' in presolver:
            res = minimize(my_fun.eval, x0, method='SLSQP', options={'ftol':1e-6, 'disp':False}, bounds=bbob_bounds)
            #res = fmin_slsqp(my_fun.eval, x0, bounds=((-5,5),)*problem.dimension, disp=False)
        elif 'smac' in presolver:
            x, cost, _ = fmin_smac(func=my_fun.eval, x0=x0, bounds=bbob_bounds, maxfun=budget_multiplier*problem.dimension, rng=sample_id)
        else:
            logger.error("%s is not defined.", presolver)
            exit(1)                    
                    
        minimal_print(problem, final=problem.index == len(suite) - 1)        
        
@click.command()
@click.option('--budget_multiplier', '-bm', required=True, default=50, type=int, help='A multiplier to determine a budget of function evaluations.')
@click.option('--presolver', '-p', required=True, default='slsqp', type=str, help='A pre-solver.')
@click.option('--sample_id', '-sid', required=True, default=0, type=int, help='The run ID of a pre-solver')
def run(budget_multiplier, presolver, sample_id):
    """Run a pre-solver

    Parameters
    ----------
    budget_multiplier : int
        A multiplier x for determining a budget of function evaluations (= x * dimension).
    presolver: string
        A pre-solver.
    sample_id: int
        Run ID.

    Returns
    -------

    """
    np.random.seed(seed=sample_id)
    random.seed(sample_id)

    run_presolver(bbob_suite='bbob', budget_multiplier=budget_multiplier, presolver=presolver, sample_dir_path='./sample_data', sample_id=sample_id, run_iids=[1,2,3,4,5], run_dims=[2, 3, 5, 10])
        
if __name__ == '__main__':
    run()
