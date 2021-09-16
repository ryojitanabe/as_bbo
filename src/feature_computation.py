"""
Computing features based on a sample.
"""
import numpy as np
from pflacco.pflacco import create_feature_object, calculate_feature_set, calculate_features
import sys
import os
import logging
import click

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def compute_features(ela_feature_class, sample_data_file_path, feature_file_path):
    bbob_lower_bound = -5
    bbob_upper_bound = 5
    
    n_cell_blocks = None
    if ela_feature_class in ['cm_angle', 'cm_conv', 'cm_grad', 'gcm']:
        n_cell_blocks = 3
    
    data_set = np.loadtxt(sample_data_file_path, delimiter=",", comments="#", dtype=np.float)
    sample_f = data_set[:, 0]
    sample_x = data_set[:, 1:]
            
    feat_object = create_feature_object(x=sample_x, y=sample_f, minimize=True, lower=bbob_lower_bound, upper=bbob_upper_bound, blocks=n_cell_blocks)    

    try:
        # The calculate_feature_set function returns a dictionary object 
        feature_dict = calculate_feature_set(feat_object, ela_feature_class)
    except rpy2.rinterface_lib.embedded.RRuntimeError as e:
        logger.error(e)

    with open(feature_file_path, 'w') as fh:
        for key, value in feature_dict.items():
            fh.write('{},{}\n'.format(key, value))            

@click.command()
@click.option('--dir_sampling_method', '-dsample', required=True, default='ihs_multiplier50_sid0', type=str, help='Directory of a sampling method.')
@click.option('--ela_feature_class', '-f', required=False, default='basic', type=str, help='A feature class to be computed.')
@click.option('--dim', '-d', required=False, default=2, type=int, help='Dimension.')
@click.option('--fun_id', '-fid', required=False, default=1, type=int, help='Function ID.')
def run(dir_sampling_method, ela_feature_class, dim, fun_id):
    sample_dir_path = os.path.join('./sample_data', dir_sampling_method)    
    bbob_suite = 'bbob'
    
    # all_feature_classes = ['basic', 'ela_distr', 'pca', 'limo', 'ic', 'disp', 'nbc', 'ela_level', 'ela_meta']    
    # dims = [2, 3, 5, 10]    
    # all_fun_ids = range(1, 24+1)
    # if bbob_suite == 'bbob-noisy':
    #     all_fun_ids = range(101, 130+1)    

    # for ela_feature_class in all_feature_classes:
    #     for dim in dims:            
    #         for fun_id in all_fun_ids:
    #             for instance_id in range(1, 5+1):                    
    #                 sample_data_file_path = os.path.join(sample_dir_path, 'x_f_data_{}_f{}_DIM{}_i{}.csv'.format(bbob_suite, fun_id, dim, instance_id))
    #                 feature_dir_path = os.path.join('./ela_feature_dataset', dir_sampling_method) 
    #                 os.makedirs(feature_dir_path, exist_ok=True)
    #                 feature_file_path = os.path.join(feature_dir_path, '{}_{}_f{}_DIM{}_i{}.csv'.format(ela_feature_class, bbob_suite, fun_id, dim, instance_id))

    #                 compute_features(ela_feature_class, sample_data_file_path, feature_file_path)
    #                 logger.info("Done: Feature=%s, dimension=%d,  f=%d, instance ID=%d", ela_feature_class, dim, fun_id, instance_id)

    for instance_id in range(1, 5+1):                    
        sample_data_file_path = os.path.join(sample_dir_path, 'x_f_data_{}_f{}_DIM{}_i{}.csv'.format(bbob_suite, fun_id, dim, instance_id))

        feature_dir_path = os.path.join('./ela_feature_dataset', dir_sampling_method)                    
        os.makedirs(feature_dir_path, exist_ok=True)
        feature_file_path = os.path.join(feature_dir_path, '{}_{}_f{}_DIM{}_i{}.csv'.format(ela_feature_class, bbob_suite, fun_id, dim, instance_id))

        compute_features(ela_feature_class, sample_data_file_path, feature_file_path)
        logger.info("Done: Feature=%s, dimension=%d,  f=%d, instance ID=%d", ela_feature_class, dim, fun_id, instance_id)
    
if __name__ == '__main__':
    run()
