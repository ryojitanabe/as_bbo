"""
Creating a sample on a BBOB benchmark suite.
"""
import numpy as np
from pflacco.pflacco import create_initial_sample
import cocoex  # only experimentation module
import sobol_seq # this works for up to 80 dimensions
from pyDOE import lhs
import sys
import logging
import os
import random
import click
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
    
def create_sample(fun, sampling_method, sample_size):
    """Create the sample

    Parameters
    ----------
    fun : callable object
        A function to be minimized.
    sampling_method: string
        A method for making the initial sample.
    sample_size: int
        The sample size.

    Returns

    -------
    sample: 2-d ndarray
        Sample (a set of solutions).
    obj_values: list
        The objective values of the sampl
    """
    dim = fun.dimension

    # Each solution is generated in the range [0,1]^dim.    
    if sampling_method == 'ihs':     
        sample = create_initial_sample(n_obs=sample_size, dim=dim, type='lhs')
    elif sampling_method == 'random':     
        sample = np.random.random_sample((sample_size, dim))
    elif sampling_method == 'sobol':     
        # sobol_seq is available for <= 80 dimensions
        sample = sobol_seq.i4_sobol_generate(dim, sample_size)
    elif sampling_method == 'lhs':     
        sample = lhs(dim, sample_size, criterion='center')
    else:
        error_msg = "Error: %s is not defined." % (sampling_method)
        logger.error(error_msg)
        exit(1)    

    # Linearly map each solution from [0,1]^dim to [-5,5]^dim
    lbound = np.full(dim, -5.)
    ubound = np.full(dim, 5.)
    sample = (ubound - lbound) * sample + lbound

    # Evaluate each solution in the sample
    obj_values = []
    for x in sample:
        obj_values.append(fun(x))

    return sample, obj_values

def create_sample_bbob(bbob_suite='bbob', sample_multiplier=50, sampling_method='lhs', sample_dir_path='./sample_data', sample_id=0, stop_dim=10):
    """
    Create the initial sample.

    This function is based on https://github.com/numbbo/coco/blob/master/code-experiments/build/python/example_experiment_for_beginners.py

    Parameters
    ----------
    bbob_suite: string
        The name of a bbob suite (bbob, bbob-noisy, bbob-biobj, bbob-largescale, bbob-mixint, bbob-biobj-mixint)
    sample_multiplier : int
        A multiplier x for determining the sample size (= x * dimension).
    sampling_method: string
        A method for making the initial sample.
    sample_dir_path: path
        A directory path.
    sample_id: int
        Run ID.
    stop_dim: int
        A dimension to terminate the sampling procedure. For example, when stop_dim=10, the sampling procedure is performed only for 2, 3, 5, and 10 dimensions.

    Returns
    -------    

    """
    sample_dir_path = os.path.join(sample_dir_path, '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sample_id))
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

        # COCO automatically perform experiments on 2, 3, 5, 10, 20, and 40 for 'bbob'. The following terminates experiments on 'stop_dim' dimension
        if bbob_suite == 'bbob' and problem.dimension > stop_dim:
            break

        sample_size = sample_multiplier * problem.dimension        
        sample, obj_values = create_sample(problem, sampling_method, sample_size)

        fun_id = int(problem.info.split('_f')[1].split('_')[0])
        instance_id = int(problem.info.split('_i')[1].split('_')[0])
        #instance_id = count_instance_id
        
        # Save each pair of x and f(x) in a csv file 
        sample_data_file_path = os.path.join(sample_dir_path, 'x_f_data_{}_f{}_DIM{}_i{}.csv'.format(bbob_suite, fun_id, problem.dimension, instance_id))
        with open(sample_data_file_path, 'w') as fh:
            for x, obj_val in zip(sample, obj_values):
                data_str = '{},'.format(obj_val)                
                data_str += ','.join([str(y) for y in x])
                fh.write(data_str + '\n')
                
        minimal_print(problem, final=problem.index == len(suite) - 1)
        
@click.command()
@click.option('--sample_multiplier', '-smulti', required=True, default=50, type=int, help='A multiplier to determine the sample size.')
@click.option('--sampling_method', '-smethod', required=True, default='ihs', type=str, help='A sampling method.')
@click.option('--sample_id', '-sid', required=True, default=0, type=int, help='The run ID of a sampler')
def run(sample_multiplier, sampling_method, sample_id):
    """Make the initial sample

    Parameters
    ----------
    sample_multiplier : int
        A multiplier x for determining the sample size (= x * dimension).
    sampling_method: string
        A method for making the initial sample.
    sample_id: int
        Run ID.
    Returns
    -------

    """
    np.random.seed(seed=sample_id)
    random.seed(sample_id)

    if sampling_method in ['ihs', 'random', 'sobol', 'lhs']:
        create_sample_bbob(bbob_suite='bbob', sample_multiplier=sample_multiplier, sampling_method=sampling_method, sample_dir_path='./sample_data', sample_id=sample_id)
    else:
        logger.error("%s is not defined.", sampling_method)
        exit(1)            
        
if __name__ == '__main__':
    run()
