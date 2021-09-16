"""
This is to obtain the optimal solution of each noiseless BBOB function for postprocessing of results of algorithm selection systems. This is based on generate_aRTA_plot.py in the COCO software (https://github.com/numbbo/coco/tree/master/code-postprocessing/aRTAplots).
"""
import os
import numpy as np
import cocoex
# Note that bbobbenchmarks is the old COCO Python file. The cocoex module and bbobbenchmarks are independent from each other.
import bbobbenchmarks as bm

def bbob_fopt():
    ### input
    output_folder = 'tmp'
    # 'bbob', 'bbob-largescale', 'bbob-biobj', 'bbob-mixint', 'bbob-biobj-mixint']
    suite_name = 'bbob'
    # # the maximum number of function evaluations is "budget_multiplier * dimensionality"
    # budget_multiplier = 100

    ### prepare
    suite = cocoex.Suite(suite_name, "", "")
    observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
    minimal_print = cocoex.utilities.MiniPrint()

    ### go
    for problem in suite:  # this loop will take several minutes or longer
        problem.observe_with(observer)  # generates the data for cocopp post-processing

        fun_id = int(problem.info.split('_f')[1].split('_')[0])
        instance_id = int(problem.info.split('_i')[1].split('_')[0])
        f, fopt = bm.instantiate(fun_id, iinstance=instance_id)
        
        os.makedirs('./bbob_fopt_data', exist_ok=True)        
        fopt_file_path =os.path.join('bbob_fopt_data', 'fopt_f{}_DIM{}_i{}.csv'.format(fun_id, problem.dimension, instance_id))    
        with open(fopt_file_path, 'w') as fh:
            fh.write("{}".format(fopt))
        minimal_print(problem, final=problem.index == len(suite) - 1)
        
if __name__ == '__main__':
    bbob_fopt()
