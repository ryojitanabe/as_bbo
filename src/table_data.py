"""
Make table data by aggregating features
"""
import numpy as np
import pandas as pd
import csv
import sys
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def create_feature_table_data(bbob_suite, table_file_path, feature_dir_path, ap_dir_path, all_feature_classes, dims, per_metric):    
    all_fun_ids = range(1, 24+1)
    if bbob_suite == 'bbob-noisy':
        all_fun_ids = range(101, 130+1)
    all_instance_ids = range(1, 5+1)
    all_instance_ids_plus0 = range(0, 5+1)
    
    ap_config_file_path = os.path.join(ap_dir_path, 'ap_config.csv')
    ap_algs = np.loadtxt(ap_config_file_path, delimiter=",", comments="#", dtype=str)
   
    # 1. Set the name of each column
    column_names = []
    my_basic_feature_names = ['dim', 'fun', 'instance']
    column_names.extend(my_basic_feature_names)
    # column_names.extend(my_high_level_prop_names)

    # 'best_alg' is for multiclass classfication
    for left_instance_id in all_instance_ids_plus0:
        column_names.append('best_alg_liid{}'.format(left_instance_id))

    # for instance_id in all_instance_ids:    
    #     column_names.append('best_alg_i{}'.format(instance_id))        
    #column_names.extend(ap_algs)
    for alg in ap_algs:
        for left_instance_id in all_instance_ids_plus0:    
            column_names.append('{}_liid{}'.format(alg, left_instance_id))
        
    # Extract the name of all the features
    dim = dims[0]
    instance_id = all_instance_ids[0]
    fun_id = all_fun_ids[0]
    
    for ela_feature_class in all_feature_classes:
        feature_file_path = os.path.join(feature_dir_path, '{}_{}_f{}_DIM{}_i{}.csv'.format(ela_feature_class, bbob_suite, fun_id, dim, instance_id))        
        feature_data_set = np.loadtxt(feature_file_path, delimiter=",", comments="#", dtype=str)
        feature_names = feature_data_set[:, 0].tolist()
        column_names.extend(feature_names)   
        
    # 2. Make table data
    table_df = pd.DataFrame(columns=column_names)

    for dim in dims:
        for fun_id in all_fun_ids:
            data_dict = {}

            # Save the relative metric value and the best algorithm
            for left_instance_id in all_instance_ids_plus0:                
                tmp_relmetric_dict = {}
                for alg in ap_algs:
                    rel_metric_file_path = os.path.join(ap_dir_path, 'rel'+per_metric, '{}_f{}_DIM{}_liid{}.csv'.format(alg, fun_id, dim, left_instance_id))
                    with open(rel_metric_file_path, 'r') as fh:
                        rel_metric_value = fh.read()
                        alg_name = '{}_liid{}'.format(alg, left_instance_id)
                        data_dict[alg_name] = float(rel_metric_value)
                        tmp_relmetric_dict[alg] = float(rel_metric_value)
                        
                data_dict['best_alg_liid{}'.format(left_instance_id)] = min(tmp_relmetric_dict, key=tmp_relmetric_dict.get)
            
            data_dict['dim'] = dim
            data_dict['fun'] = fun_id           
                    
            # For each instance, recode the feature values
            for instance_id in all_instance_ids:
                data_dict['instance'] = instance_id                
                for ela_feature_class in all_feature_classes:
                    feature_file_path = os.path.join(feature_dir_path, '{}_{}_f{}_DIM{}_i{}.csv'.format(ela_feature_class, bbob_suite, fun_id, dim, instance_id))
                    feature_data_set = np.loadtxt(feature_file_path, delimiter=",", comments="#", dtype=str)
                    for key, value in feature_data_set:
                        data_dict[key] = value
                table_df = table_df.append(pd.Series(data_dict), ignore_index=True)

    table_df.to_csv(table_file_path, index=False)

def run(dir_sampling_method='ihs_multiplier50_sid0', ap_name='kt_ecj19', target='-2.0', per_metric='sp1'):
    bbob_suite = 'bbob'
    all_feature_classes = ['basic', 'ela_distr', 'pca', 'limo', 'ic', 'disp', 'nbc', 'ela_level', 'ela_meta']
    dims = [2, 3, 5, 10]

    feature_dir_path = os.path.join('./ela_feature_dataset', dir_sampling_method)
    ap_dir_path = os.path.join('./alg_portfolio', ap_name)
    
    table_dir_path = os.path.join(ap_dir_path, 'feature_table_data')
    os.makedirs(table_dir_path, exist_ok=True)

    features_str = '_'.join(all_feature_classes)
    dims_str = '_'.join([str(d) for d in dims])
    table_file_path = os.path.join(table_dir_path, '{}_{}_{}_dims{}.csv'.format(dir_sampling_method, per_metric, features_str, dims_str))

    create_feature_table_data(bbob_suite, table_file_path, feature_dir_path, ap_dir_path, all_feature_classes, dims, per_metric)
        
if __name__ == '__main__':
    sampling_method = 'ihs'
    sample_multiplier = 50
    portfolio_list = ['kt', 'dlvat', 'jped', 'bmtp', 'mk', 'ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']
    
    for ap_name in portfolio_list:
        for sid in range(0, 31):
            dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)
            run(dir_sampling_method, ap_name, target='-2.0')                

