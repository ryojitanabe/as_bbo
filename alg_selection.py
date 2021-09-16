"""
Perform algorithm selection
"""
import numpy as np
import pandas as pd
import statistics as stat
import csv
import os
import sys
import logging
import click
import random
import json
from statistics import multimode, mode

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

#from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

from pyclustering.cluster.gmeans import gmeans

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(filename='{}.log'.format(__file__), level=logging.DEBUG)

def ap_column_list(ap_algs):
    l = []    
    for i in range(0, 5+1):
        s = 'best_alg_liid{}'.format(i)
        l.append(s)
    for alg in ap_algs:
        for i in range(0, 5+1):
            s = '{}_liid{}'.format(alg, i)
            l.append(s)
    return l

def preprocessing_table(table_data_file_path, dim, cross_valid_type, ap_columns, problem_info_columns):
    table_df = pd.read_csv(table_data_file_path, header=0)
    # All dimensions are considered in LOPOAD-CV
    if cross_valid_type == 'lopo_cv' or cross_valid_type == 'loio_cv':
        table_df = table_df[table_df['dim'] == dim]
        
    # Remove duplicated columns, where all elements take the same value (e.g., nan, nan, ..., nan and 2, 2, ..., 2)
    table_df = table_df.dropna(how='all', axis=1)
    dup_columns = []
    # The first len(problem_info_columns) + len(ap_columns) columns should be ignored
    n_misc = len(problem_info_columns) + len(ap_columns)
    all_columns =  table_df.columns.values
    for column in all_columns[n_misc:]:
        unum = table_df[column].nunique()
        if unum == 1:
            dup_columns.append(column)
    table_df = table_df.drop(dup_columns, axis=1)           

    # Impute inf and nan values.
    table_df = table_df.replace([np.inf, -np.inf], np.nan)
    missing_columns = table_df.columns[table_df.isnull().any()]
    if len(missing_columns) > 0:
        logger.warning('Missing values (NaN or INF) have been imputed with the mean of all values in the corresponding column: %s', missing_columns)
        for missing_column in missing_columns:
            table_df[[missing_column]] = imp_mean.fit_transform(table_df[[missing_column]])
            
    return table_df

def feature_selection(estimator, feature_selector, n_features_to_select, X_train, y_train, X_test, all_feature_columns, fs_res_file_path):
    if feature_selector == 'rfe':
        fselector = RFE(estimator, n_features_to_select=n_features_to_select)
    elif feature_selector == 'sffs':
        fselector = SequentialFeatureSelector(estimator, direction='forward', n_features_to_select=n_features_to_select)
    elif feature_selector == 'sfbs':
        fselector = SequentialFeatureSelector(estimator, direction='backward', n_features_to_select=n_features_to_select)
    else:
        logger.error('%s is not defined', fselector)
        exit(1)
    fselector.fit(X_train, y_train)
    X_train = fselector.transform(X_train)
    X_test = fselector.transform(X_test)                

    selected_columns = all_feature_columns[fselector.support_]        
    with open(fs_res_file_path, 'w') as fh:
        fh.write(','.join(selected_columns))

    return X_train, X_test            

def multiclass_classification_lopo_cv(as_result_dir_path, table_data_file_path, dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):    
    table_df = preprocessing_table(table_data_file_path, dim, cross_valid_type, ap_columns, problem_info_columns)
        
    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[table_df['fun'] == left_fun_id]
    test_instance_ids = test_df['instance'].values
    y_test = test_df['best_alg_liid0'].values
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values

    # train datasets
    train_df = table_df[table_df['fun'] != left_fun_id]
    y_train = train_df['best_alg_liid0'].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values

    estimator = RandomForestClassifier(random_state=ml_seed)

    # Feature selection
    if feature_selector != 'none':
        fs_res_file_path = os.path.join(as_result_dir_path, 'fs_res_f{}_DIM{}_ix.csv'.format(left_fun_id, dim))
        X_train, X_test = feature_selection(estimator, feature_selector, n_features_to_select, X_train, y_train, X_test, train_df.columns.values, fs_res_file_path)
        
    estimator.fit(X_train, y_train)
    pred_algs = estimator.predict(X_test)
        
    for instance_id, pred_alg in zip(test_instance_ids, pred_algs):
        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(left_fun_id, dim, instance_id))
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)

def multiclass_classification_lopoad_cv(as_result_dir_path, table_data_file_path, left_dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):
    table_df = preprocessing_table(table_data_file_path, None, cross_valid_type, ap_columns, problem_info_columns)
        
    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[(table_df['fun'] == left_fun_id) & (table_df['dim'] == left_dim)]
    test_instance_ids = test_df['instance'].values
    y_test = test_df['best_alg_liid0'].values
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values

    # train datasets
    train_df = table_df[(table_df['fun'] != left_fun_id) | (table_df['dim'] != left_dim)]    
    y_train = train_df['best_alg_liid0'].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values
    
    estimator = RandomForestClassifier(random_state=ml_seed)

    # Feature selection
    if feature_selector != 'none':
        fs_res_file_path = os.path.join(as_result_dir_path, 'fs_res_f{}_DIM{}_ix.csv'.format(left_fun_id, left_dim))
        X_train, X_test = feature_selection(estimator, feature_selector, n_features_to_select, X_train, y_train, X_test, train_df.columns.values, fs_res_file_path)

    estimator.fit(X_train, y_train)
    pred_algs = estimator.predict(X_test)

    for instance_id, pred_alg in zip(test_instance_ids, pred_algs):
        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(left_fun_id, left_dim, instance_id))        
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)

def multiclass_classification_loio_cv(as_result_dir_path, table_data_file_path, dim, left_instance_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):
    table_df = preprocessing_table(table_data_file_path, dim, cross_valid_type, ap_columns, problem_info_columns)
       
    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[table_df['instance'] == left_instance_id]
    test_fun_ids = test_df['fun'].values
    y_test = test_df['best_alg_liid{}'.format(left_instance_id)].values
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values
    
    # train datasets
    train_df = table_df[table_df['instance'] != left_instance_id]    
    y_train = train_df['best_alg_liid{}'.format(left_instance_id)].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values
    
    estimator = RandomForestClassifier(random_state=ml_seed)

    # Feature selection
    if feature_selector != 'none':
        fs_res_file_path = os.path.join(as_result_dir_path, 'fs_res_fx_DIM{}_i{}.csv'.format(dim, left_instance_id))
        X_train, X_test = feature_selection(estimator, feature_selector, n_features_to_select, X_train, y_train, X_test, train_df.columns.values, fs_res_file_path)

    estimator.fit(X_train, y_train)    
    pred_algs = estimator.predict(X_test)

    for fun_id, pred_alg in zip(test_fun_ids, pred_algs):
        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(fun_id, dim, left_instance_id))
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)

def hiearchical_regression_lopo_cv(as_result_dir_path, table_data_file_path, dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):
    table_df = preprocessing_table(table_data_file_path, dim, cross_valid_type, ap_columns, problem_info_columns)
    
    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[table_df['fun'] == left_fun_id]
    test_instance_ids = test_df['instance'].values
    # The relERT values of all algorithms are necessary for the hiearchical regression-based selection method. 
    y_test_dict = {}
    for alg in ap_algs:    
        alg_data_name = '{}_liid0'.format(alg)
        y_test_dict[alg] = test_df[alg_data_name].values
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values

    # train datasets
    train_df = table_df[table_df['fun'] != left_fun_id]
    y_train_dict = {}
    for alg in ap_algs:
        alg_data_name = '{}_liid0'.format(alg)
        y_train_dict[alg] = train_df[alg_data_name].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values
    
    # train and test
    pred_relert_dict = {}
    for alg in ap_algs:
        estimator = RandomForestRegressor(random_state=ml_seed)

        # Feature selection
        if feature_selector != 'none':
            # X_train and X_test should be initialized for each trial
            X_train = train_df.values
            X_test = test_df.values                                                    
            fs_res_file_path = os.path.join(as_result_dir_path, 'fs_res_f{}_DIM{}_ix_{}.csv'.format(left_fun_id, dim, alg))
            X_train, X_test = feature_selection(estimator, feature_selector, n_features_to_select, X_train, y_train_dict[alg], X_test, train_df.columns.values, fs_res_file_path)

        estimator.fit(X_train, y_train_dict[alg])
        pred_relert_dict[alg] = estimator.predict(X_test)
    
    # The use of dummy_id is confusable.
    # When test_instance_ids include sequential instance ids like [1, 2, 3, 4, 5], dummy_id is not necessary
    # When test_instance_ids include unsequential instance ids such as [1, 2, 3, 4, 5, 20, 21, 23, 30, 79], dummy_id is necessary
    for dummy_id, instance_id in enumerate(test_instance_ids):
        # For each instance, the best algorithm is selected from the algorithm portfolio in terms of their predicted relERT values.
        tmp_relert_dict = {}
        for alg in ap_algs:
            tmp_relert_dict[alg] = pred_relert_dict[alg][dummy_id]
        pred_alg = min(tmp_relert_dict, key=tmp_relert_dict.get)

        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(left_fun_id, dim, instance_id))
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)

def hiearchical_regression_lopoad_cv(as_result_dir_path, table_data_file_path, left_dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):    
    table_df = preprocessing_table(table_data_file_path, None, cross_valid_type, ap_columns, problem_info_columns)

    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[(table_df['fun'] == left_fun_id) & (table_df['dim'] == left_dim)]    
    test_instance_ids = test_df['instance'].values
    # The relERT values of all algorithms are necessary for the hiearchical regression-based selection method. 
    y_test_dict = {}
    for alg in ap_algs:    
        alg_data_name = '{}_liid0'.format(alg)
        y_test_dict[alg] = test_df[alg_data_name].values
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values

    # train datasets
    train_df = table_df[(table_df['fun'] != left_fun_id) | (table_df['dim'] != left_dim)]
    y_train_dict = {}
    for alg in ap_algs:
        alg_data_name = '{}_liid0'.format(alg)
        y_train_dict[alg] = train_df[alg_data_name].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values
    
    # train and test
    pred_relert_dict = {}
    for alg in ap_algs:
        estimator = RandomForestRegressor(random_state=ml_seed)

        # Feature selection
        if feature_selector != 'none':
            # X_train and X_test should be initialized for each trial
            X_train = train_df.values
            X_test = test_df.values                                                    
            fs_res_file_path = os.path.join(as_result_dir_path, 'fs_res_f{}_DIM{}_ix_{}.csv'.format(left_fun_id, left_dim, alg))
            X_train, X_test = feature_selection(estimator, feature_selector, n_features_to_select, X_train, y_train_dict[alg], X_test, train_df.columns.values, fs_res_file_path)
            
        estimator.fit(X_train, y_train_dict[alg])
        pred_relert_dict[alg] = estimator.predict(X_test)
    
    # The use of dummy_id is confusable.
    # When test_instance_ids include sequential instance ids like [1, 2, 3, 4, 5], dummy_id is not necessary
    # When test_instance_ids include unsequential instance ids such as [1, 2, 3, 4, 5, 20, 21, 23, 30, 79], dummy_id is necessary
    for dummy_id, instance_id in enumerate(test_instance_ids):
        # For each instance, the best algorithm is selected from the algorithm portfolio in terms of their predicted relERT values.
        tmp_relert_dict = {}
        for alg in ap_algs:
            tmp_relert_dict[alg] = pred_relert_dict[alg][dummy_id]
        pred_alg = min(tmp_relert_dict, key=tmp_relert_dict.get)

        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(left_fun_id, left_dim, instance_id))
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)        

def hiearchical_regression_loio_cv(as_result_dir_path, table_data_file_path, dim, left_instance_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):    
    table_df = preprocessing_table(table_data_file_path, dim, cross_valid_type, ap_columns, problem_info_columns)

    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[table_df['instance'] == left_instance_id]
    test_fun_ids = test_df['fun'].values
    # The relERT values of all algorithms are necessary for the hiearchical regression-based selection method. 
    y_test_dict = {}
    for alg in ap_algs:
        # Observed values to be predicted are the relERT value on all the five instances (IIDs: 1, 2, 3, 4, and 5)
        alg_data_name = '{}_liid0'.format(alg)        
        y_test_dict[alg] = test_df[alg_data_name].values        
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values

    # train datasets
    train_df = table_df[table_df['instance'] != left_instance_id]    
    y_train_dict = {}
    for alg in ap_algs:
        # The relERT value on four instances (except for left_instance_id) is used for the training phase.
        # For example, if left_instance_id = 2, IIDs = 1, 3, 4, and 5. 
        alg_data_name = '{}_liid{}'.format(alg, left_instance_id)
        y_train_dict[alg] = train_df[alg_data_name].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values

    # train and test
    pred_relert_dict = {}
    for alg in ap_algs:
        estimator = RandomForestRegressor(random_state=ml_seed)

        # Feature selection
        if feature_selector != 'none':
            # X_train and X_test should be initialized for each trial
            X_train = train_df.values
            X_test = test_df.values                                                    
            fs_res_file_path = os.path.join(as_result_dir_path, 'fs_res_fx_DIM{}_i{}_{}.csv'.format(dim, left_instance_id, alg))
            X_train, X_test = feature_selection(estimator, feature_selector, n_features_to_select, X_train, y_train_dict[alg], X_test, train_df.columns.values, fs_res_file_path)
            
        estimator.fit(X_train, y_train_dict[alg])
        pred_relert_dict[alg] = estimator.predict(X_test)
        
    for i, fun_id in enumerate(test_fun_ids):
        # For each function instance (f=fun_id, IID=left_instance_id), the best algorithm is selected from the algorithm portfolio in terms of their predicted relERT values.
        tmp_relert_dict = {}
        for alg in ap_algs:
            tmp_relert_dict[alg] = pred_relert_dict[alg][i]
        pred_alg = min(tmp_relert_dict, key=tmp_relert_dict.get)
        
        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(fun_id, dim, left_instance_id))
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)        
    
def pairwise_classification_lopo_cv(as_result_dir_path, table_data_file_path, dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):        
    table_df = preprocessing_table(table_data_file_path, dim, cross_valid_type, ap_columns, problem_info_columns)

    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[table_df['fun'] == left_fun_id]
    test_instance_ids = test_df['instance'].values
    y_test_dict = {}
    for alg in ap_algs:    
        alg_data_name = '{}_liid0'.format(alg)
        y_test_dict[alg] = test_df[alg_data_name].values
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values

    # train datasets
    train_df = table_df[table_df['fun'] != left_fun_id]
    y_train_dict = {}
    for alg in ap_algs:
        alg_data_name = '{}_liid0'.format(alg)
        y_train_dict[alg] = train_df[alg_data_name].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values
    
    vote_df = pd.DataFrame(index=test_instance_ids, columns=ap_algs)
    vote_df.fillna(0, inplace=True)

    # Perform a binary classification for all pairs of optimizers in ap_algs. Select the optimizer with the most votes.
    for i, alg_i in enumerate(ap_algs):
        for j, alg_j in enumerate(ap_algs[i:]):            
            if i != j:
                # Train
                y_train_binary = np.empty(len(y_train_dict[alg_i]), dtype=object)
                for k, (rel_ert_i, rel_ert_j) in enumerate(zip(y_train_dict[alg_i], y_train_dict[alg_j])):
                    if rel_ert_i < rel_ert_j:
                        y_train_binary[k] = alg_i
                    else:
                        y_train_binary[k] = alg_j
                estimator = RandomForestClassifier(random_state=ml_seed)

                # Feature selection
                if feature_selector != 'none':
                    # X_train and X_test should be initialized for each trial
                    X_train = train_df.values
                    X_test = test_df.values                                        
                    fs_res_file_path = os.path.join(as_result_dir_path, 'fs_res_f{}_DIM{}_ix_{}_{}.csv'.format(left_fun_id, dim, alg_i, alg_j))
                    X_train, X_test = feature_selection(estimator, feature_selector, n_features_to_select, X_train, y_train_binary, X_test, train_df.columns.values, fs_res_file_path)
                
                estimator.fit(X_train, y_train_binary)
                # Test
                pred_algs = estimator.predict(X_test)
                for test_id, p_alg in zip(test_instance_ids, pred_algs):                
                    vote_df.loc[test_id, p_alg] += 1

    # # Ties are broken by lexical order
    # pred_algs = list(vote_df.idxmax(axis=1))

    # Ties are broken randomly
    pred_algs = []
    for i in test_instance_ids:
        most_votes_algs = list(vote_df.loc[i][vote_df.loc[i] == vote_df.loc[i].max()].index)
        pred_algs.append(random.choice(most_votes_algs))     
    
    for instance_id, pred_alg in zip(test_instance_ids, pred_algs):
        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(left_fun_id, dim, instance_id))
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)

def pairwise_classification_lopoad_cv(as_result_dir_path, table_data_file_path, left_dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):    
    table_df = preprocessing_table(table_data_file_path, None, cross_valid_type, ap_columns, problem_info_columns)

    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[(table_df['fun'] == left_fun_id) & (table_df['dim'] == left_dim)]
    test_instance_ids = test_df['instance'].values
    y_test_dict = {}
    for alg in ap_algs:    
        alg_data_name = '{}_liid0'.format(alg)
        y_test_dict[alg] = test_df[alg_data_name].values
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values

    # train datasets
    train_df = table_df[(table_df['fun'] != left_fun_id) | (table_df['dim'] != left_dim)]    
    y_train_dict = {}
    for alg in ap_algs:
        alg_data_name = '{}_liid0'.format(alg)
        y_train_dict[alg] = train_df[alg_data_name].values    
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values

    vote_df = pd.DataFrame(index=test_instance_ids, columns=ap_algs)
    vote_df.fillna(0, inplace=True)

    # Perform a binary classification for all pairs of optimizers in ap_algs. Select the optimizer with the most votes.    
    for i, alg_i in enumerate(ap_algs):
        for j, alg_j in enumerate(ap_algs[i:]):            
            if alg_i != alg_j:
                # Train
                y_train_binary = np.empty(len(y_train_dict[alg_i]), dtype=object)
                for k, (rel_ert_i, rel_ert_j) in enumerate(zip(y_train_dict[alg_i], y_train_dict[alg_j])):
                    if rel_ert_i < rel_ert_j:
                        y_train_binary[k] = alg_i
                    else:
                        y_train_binary[k] = alg_j                                    
                estimator = RandomForestClassifier(random_state=ml_seed)

                # Feature selection
                if feature_selector != 'none':
                    # X_train and X_test should be initialized for each trial
                    X_train = train_df.values
                    X_test = test_df.values                                        
                    fs_res_file_path = os.path.join(as_result_dir_path, 'fs_res_f{}_DIM{}_ix_{}_{}.csv'.format(left_fun_id, left_dim, alg_i, alg_j))
                    X_train, X_test = feature_selection(estimator, feature_selector, n_features_to_select, X_train, y_train_binary, X_test, train_df.columns.values, fs_res_file_path)
                
                estimator.fit(X_train, y_train_binary)
                # Test                
                pred_algs = estimator.predict(X_test)
                for test_id, p_alg in zip(test_instance_ids, pred_algs):                
                    vote_df.loc[test_id, p_alg] += 1

    # # Ties are broken by lexical order
    # pred_algs = list(vote_df.idxmax(axis=1))

    # Ties are broken randomly
    pred_algs = []
    for i in test_instance_ids:
        most_votes_algs = list(vote_df.loc[i][vote_df.loc[i] == vote_df.loc[i].max()].index)
        pred_algs.append(random.choice(most_votes_algs))     
    
    for instance_id, pred_alg in zip(test_instance_ids, pred_algs):
        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(left_fun_id, left_dim, instance_id))        
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)

def pairwise_classification_loio_cv(as_result_dir_path, table_data_file_path, dim, left_instance_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):
    table_df = preprocessing_table(table_data_file_path, dim, cross_valid_type, ap_columns, problem_info_columns)

    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[table_df['instance'] == left_instance_id]
    test_fun_ids = test_df['fun'].values
    y_test_dict = {}
    for alg in ap_algs:    
        alg_data_name = '{}_liid0'.format(alg)
        y_test_dict[alg] = test_df[alg_data_name].values   
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values

    # train datasets
    train_df = table_df[table_df['instance'] != left_instance_id]    
    y_train_dict = {}
    for alg in ap_algs:
        alg_data_name = '{}_liid{}'.format(alg, left_instance_id)
        y_train_dict[alg] = train_df[alg_data_name].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values

    vote_df = pd.DataFrame(index=test_fun_ids, columns=ap_algs)
    vote_df.fillna(0, inplace=True)

    # Perform a binary classification for all pairs of optimizers in ap_algs. Select the optimizer with the most votes.    
    for i, alg_i in enumerate(ap_algs):
        for j, alg_j in enumerate(ap_algs):            
            if i != j:
                # Train
                y_train_binary = np.empty(len(y_train_dict[alg_i]), dtype=object)
                for k, (rel_ert_i, rel_ert_j) in enumerate(zip(y_train_dict[alg_i], y_train_dict[alg_j])):
                    if rel_ert_i < rel_ert_j:
                        y_train_binary[k] = alg_i
                    else:
                        y_train_binary[k] = alg_j                                    
                estimator = RandomForestClassifier(random_state=ml_seed)

                # Feature selection
                if feature_selector != 'none':
                    # X_train and X_test should be initialized for each trial
                    X_train = train_df.values
                    X_test = test_df.values                                        
                    fs_res_file_path = os.path.join(as_result_dir_path, 'fs_res_fx_DIM{}_i{}_{}_{}.csv'.format(dim, left_instance_id, alg_i, alg_j))
                    X_train, X_test = feature_selection(estimator, feature_selector, n_features_to_select, X_train, y_train_binary, X_test, train_df.columns.values, fs_res_file_path)
                
                estimator.fit(X_train, y_train_binary)
                # Test                
                pred_algs = estimator.predict(X_test)
                for test_id, p_alg in zip(test_fun_ids, pred_algs):                
                    vote_df.loc[test_id, p_alg] += 1

    # # Ties are broken by lexical order
    # pred_algs = list(vote_df.idxmax(axis=1))

    # Ties are broken randomly
    pred_algs = []
    for i in test_fun_ids:
        most_votes_algs = list(vote_df.loc[i][vote_df.loc[i] == vote_df.loc[i].max()].index)
        pred_algs.append(random.choice(most_votes_algs))     
    
    for fun_id, pred_alg in zip(test_fun_ids, pred_algs):
        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(fun_id, dim, left_instance_id))
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)

def pairwise_regression_lopo_cv(as_result_dir_path, table_data_file_path, dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):
    table_df = preprocessing_table(table_data_file_path, dim, cross_valid_type, ap_columns, problem_info_columns)

    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[table_df['fun'] == left_fun_id]
    test_instance_ids = test_df['instance'].values
    y_test_dict = {}
    
    for alg in ap_algs:    
        alg_data_name = '{}_liid0'.format(alg)
        y_test_dict[alg] = test_df[alg_data_name].values
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values

    # train datasets
    train_df = table_df[table_df['fun'] != left_fun_id]
    y_train_dict = {}
    for alg in ap_algs:
        alg_data_name = '{}_liid0'.format(alg)
        y_train_dict[alg] = train_df[alg_data_name].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values
    
    vote_df = pd.DataFrame(index=test_instance_ids, columns=ap_algs)
    vote_df.fillna(0, inplace=True)

    # Build a regression model for all pairs of optimizers in ap_algs. Select the optimizer with the lowest prediction value.
    for i, alg_i in enumerate(ap_algs):
        for j, alg_j in enumerate(ap_algs[i:]):            
            if i != j:
                # Train
                # The difference between two relERT values
                y_train_diff = y_train_dict[alg_i] - y_train_dict[alg_j]
                estimator = RandomForestRegressor(random_state=ml_seed)
                
                # Feature selection
                if feature_selector != 'none':
                    # X_train and X_test should be initialized for each trial
                    X_train = train_df.values
                    X_test = test_df.values                                        
                    fs_res_file_path = os.path.join(as_result_dir_path, 'fs_res_f{}_DIM{}_ix_{}_{}.csv'.format(left_fun_id, dim, alg_i, alg_j))
                    X_train, X_test = feature_selection(estimator, feature_selector, n_features_to_select, X_train, y_train_diff, X_test, train_df.columns.values, fs_res_file_path)
                
                estimator.fit(X_train, y_train_diff)
                # Test
                pred_diff_relert_values = estimator.predict(X_test)

                for test_id, pred_diff_relert in zip(test_instance_ids, pred_diff_relert_values):
                    vote_df.loc[test_id, alg_i] += pred_diff_relert
                    # Invert the sign for the second optimizer. 
                    vote_df.loc[test_id, alg_j] -= pred_diff_relert

    # # Ties are broken by lexical order
    # pred_algs = list(vote_df.idxmax(axis=1))

    # Ties are broken randomly, but this operation is almost never performed in pairwise regression-based selection.
    pred_algs = []
    for i in test_instance_ids:
        most_votes_algs = list(vote_df.loc[i][vote_df.loc[i] == vote_df.loc[i].min()].index)
        pred_algs.append(random.choice(most_votes_algs))     
        
    for instance_id, pred_alg in zip(test_instance_ids, pred_algs):
        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(left_fun_id, dim, instance_id))
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)

def pairwise_regression_lopoad_cv(as_result_dir_path, table_data_file_path, left_dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):        
    table_df = preprocessing_table(table_data_file_path, None, cross_valid_type, ap_columns, problem_info_columns)    

    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[(table_df['fun'] == left_fun_id) & (table_df['dim'] == left_dim)]    
    test_instance_ids = test_df['instance'].values
    y_test_dict = {}    
    for alg in ap_algs:    
        alg_data_name = '{}_liid0'.format(alg)
        y_test_dict[alg] = test_df[alg_data_name].values
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values

    # train datasets
    train_df = table_df[(table_df['fun'] != left_fun_id) | (table_df['dim'] != left_dim)]        
    y_train_dict = {}
    for alg in ap_algs:
        alg_data_name = '{}_liid0'.format(alg)
        y_train_dict[alg] = train_df[alg_data_name].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values
    
    vote_df = pd.DataFrame(index=test_instance_ids, columns=ap_algs)
    vote_df.fillna(0, inplace=True)

    # Build a regression model for all pairs of optimizers in ap_algs. Select the optimizer with the lowest prediction value.
    for i, alg_i in enumerate(ap_algs):
        for j, alg_j in enumerate(ap_algs[i:]):            
            if i != j:
                # Train
                # The difference between two relERT values
                y_train_diff = y_train_dict[alg_i] - y_train_dict[alg_j]
                estimator = RandomForestRegressor(random_state=ml_seed)
                
                # Feature selection
                if feature_selector != 'none':
                    # X_train and X_test should be initialized for each trial
                    X_train = train_df.values
                    X_test = test_df.values                                        
                    fs_res_file_path = os.path.join(as_result_dir_path, 'fs_res_f{}_DIM{}_ix_{}_{}.csv'.format(left_fun_id, left_dim, alg_i, alg_j))                    
                    X_train, X_test = feature_selection(estimator, feature_selector, n_features_to_select, X_train, y_train_diff, X_test, train_df.columns.values, fs_res_file_path)
                
                estimator.fit(X_train, y_train_diff)
                # Test
                pred_diff_relert_values = estimator.predict(X_test)

                for test_id, pred_diff_relert in zip(test_instance_ids, pred_diff_relert_values):
                    vote_df.loc[test_id, alg_i] += pred_diff_relert
                    # Invert the sign for the second optimizer. 
                    vote_df.loc[test_id, alg_j] -= pred_diff_relert

    # # Ties are broken by lexical order
    # pred_algs = list(vote_df.idxmax(axis=1))

    # Ties are broken randomly, but this operation is almost never performed in pairwise regression-based selection.
    pred_algs = []
    for i in test_instance_ids:
        most_votes_algs = list(vote_df.loc[i][vote_df.loc[i] == vote_df.loc[i].min()].index)
        pred_algs.append(random.choice(most_votes_algs))     
        
    for instance_id, pred_alg in zip(test_instance_ids, pred_algs):
        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(left_fun_id, left_dim, instance_id))
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)

def pairwise_regression_loio_cv(as_result_dir_path, table_data_file_path, dim, left_instance_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):
    table_df = preprocessing_table(table_data_file_path, dim, cross_valid_type, ap_columns, problem_info_columns)

    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[table_df['instance'] == left_instance_id]
    test_fun_ids = test_df['fun'].values
    y_test_dict = {}
    for alg in ap_algs:    
        alg_data_name = '{}_liid0'.format(alg)
        y_test_dict[alg] = test_df[alg_data_name].values   
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values
    
    # train datasets
    train_df = table_df[table_df['instance'] != left_instance_id]    
    y_train_dict = {}
    for alg in ap_algs:
        alg_data_name = '{}_liid{}'.format(alg, left_instance_id)
        y_train_dict[alg] = train_df[alg_data_name].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values
    
    vote_df = pd.DataFrame(index=test_fun_ids, columns=ap_algs)
    vote_df.fillna(0, inplace=True)

    # Build a regression model for all pairs of optimizers in ap_algs. Select the optimizer with the lowest prediction value.
    for i, alg_i in enumerate(ap_algs):
        for j, alg_j in enumerate(ap_algs[i:]):            
            if i != j:
                # Train
                # The difference between two relERT values
                y_train_diff = y_train_dict[alg_i] - y_train_dict[alg_j]
                estimator = RandomForestRegressor(random_state=ml_seed)
                
                # Feature selection
                if feature_selector != 'none':
                    # X_train and X_test should be initialized for each trial
                    X_train = train_df.values
                    X_test = test_df.values                                        
                    fs_res_file_path = os.path.join(as_result_dir_path, 'fs_res_fx_DIM{}_i{}_{}_{}.csv'.format(dim, left_instance_id, alg_i, alg_j))                    
                    X_train, X_test = feature_selection(estimator, feature_selector, n_features_to_select, X_train, y_train_diff, X_test, train_df.columns.values, fs_res_file_path)
                
                estimator.fit(X_train, y_train_diff)
                # Test
                pred_diff_relert_values = estimator.predict(X_test)

                for test_id, pred_diff_relert in zip(test_fun_ids, pred_diff_relert_values):            
                    vote_df.loc[test_id, alg_i] += pred_diff_relert
                    # Invert the sign for the second optimizer. 
                    vote_df.loc[test_id, alg_j] -= pred_diff_relert

    # # Ties are broken by lexical order
    # pred_algs = list(vote_df.idxmax(axis=1))

    # Ties are broken randomly, but this operation is almost never performed in pairwise regression-based selection.
    pred_algs = []
    for i in test_fun_ids:
        most_votes_algs = list(vote_df.loc[i][vote_df.loc[i] == vote_df.loc[i].min()].index)
        pred_algs.append(random.choice(most_votes_algs))     

    for fun_id, pred_alg in zip(test_fun_ids, pred_algs):
        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(fun_id, dim, left_instance_id))
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)

def clustering_lopo_cv(as_result_dir_path, table_data_file_path, dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):    
    table_df = preprocessing_table(table_data_file_path, dim, cross_valid_type, ap_columns, problem_info_columns)

    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[table_df['fun'] == left_fun_id]
    test_instance_ids = test_df['instance'].values
    # The relERT values of all algorithms are necessary for the hiearchical regression-based selection method. 
    y_test_dict = {}
    for alg in ap_algs:    
        alg_data_name = '{}_liid0'.format(alg)
        y_test_dict[alg] = test_df[alg_data_name].values
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values

    # train datasets
    train_df = table_df[table_df['fun'] != left_fun_id]
    y_train_dict = {}
    for alg in ap_algs:
        alg_data_name = '{}_liid0'.format(alg)
        y_train_dict[alg] = train_df[alg_data_name].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values
    
    # For each feature, normalize feature values into the range [-1, 1].
    scaler =  MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    # Perform a clustering by g-means.
    gmeans_instance = gmeans(X_train, repeat=10, random_state=ml_seed).process()
    clusters = gmeans_instance.get_clusters()
    centers = gmeans_instance.get_centers()
    n_clusters = len(centers)

    # Determine the best solver for each cluster. 
    cluster_best_algs = []    
    for cluster in clusters:
        # print("cluster:", cluster)
        vote_df = pd.DataFrame(index=cluster, columns=ap_algs)
        vote_df.fillna(0, inplace=True)
        for iid in cluster:            
            for alg in ap_algs:
                vote_df.loc[iid, alg] = y_train_dict[alg][iid]                
        tmp_df = vote_df.rank(method='min', axis=1)
        tmp_df = tmp_df.mean(numeric_only=True)
        l = tmp_df[tmp_df == tmp_df.min()].index
        cluster_best_algs.append(random.choice(l))        
    cluster_best_algs = np.array(cluster_best_algs)
    
    # Test
    X_test = scaler.transform(X_test)    
    pred_cluster_labels = gmeans_instance.predict(X_test)        
    pred_algs = cluster_best_algs[pred_cluster_labels]
        
    for instance_id, pred_alg in zip(test_instance_ids, pred_algs):
        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(left_fun_id, dim, instance_id))
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)

def clustering_lopoad_cv(as_result_dir_path, table_data_file_path, left_dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):
    table_df = preprocessing_table(table_data_file_path, dim, cross_valid_type, ap_columns, problem_info_columns)

    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[(table_df['fun'] == left_fun_id) & (table_df['dim'] == left_dim)]    
    test_instance_ids = test_df['instance'].values
    # The relERT values of all algorithms are necessary for the hiearchical regression-based selection method. 
    y_test_dict = {}
    for alg in ap_algs:    
        alg_data_name = '{}_liid0'.format(alg)
        y_test_dict[alg] = test_df[alg_data_name].values
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values

    # train datasets
    train_df = table_df[(table_df['fun'] != left_fun_id) | (table_df['dim'] != left_dim)]
    y_train_dict = {}
    for alg in ap_algs:
        alg_data_name = '{}_liid0'.format(alg)
        y_train_dict[alg] = train_df[alg_data_name].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values
    
    # For each feature, normalize feature values into the range [-1, 1].
    scaler =  MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    # Perform a clustering by g-means.
    gmeans_instance = gmeans(X_train, repeat=10, random_state=ml_seed).process()
    clusters = gmeans_instance.get_clusters()
    centers = gmeans_instance.get_centers()
    n_clusters = len(centers)

    # Determine the best solver for each cluster. 
    cluster_best_algs = []    
    for cluster in clusters:
        # print("cluster:", cluster)
        vote_df = pd.DataFrame(index=cluster, columns=ap_algs)
        vote_df.fillna(0, inplace=True)
        for iid in cluster:            
            for alg in ap_algs:
                vote_df.loc[iid, alg] = y_train_dict[alg][iid]                
        tmp_df = vote_df.rank(method='min', axis=1)
        tmp_df = tmp_df.mean(numeric_only=True)
        l = tmp_df[tmp_df == tmp_df.min()].index
        cluster_best_algs.append(random.choice(l))        
    cluster_best_algs = np.array(cluster_best_algs)
    
    # Test
    X_test = scaler.transform(X_test)    
    pred_cluster_labels = gmeans_instance.predict(X_test)        
    pred_algs = cluster_best_algs[pred_cluster_labels]
        
    for instance_id, pred_alg in zip(test_instance_ids, pred_algs):
        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(left_fun_id, left_dim, instance_id))        
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)

def clustering_loio_cv(as_result_dir_path, table_data_file_path, dim, left_instance_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select):    
    table_df = preprocessing_table(table_data_file_path, dim, cross_valid_type, ap_columns, problem_info_columns)

    # Split data sets into train and test datasets
    # Test data
    test_df = table_df[table_df['instance'] == left_instance_id]
    test_fun_ids = test_df['fun'].values
    # The relERT values of all algorithms are necessary for the hiearchical regression-based selection method. 
    y_test_dict = {}
    for alg in ap_algs:
        # Observed values to be predicted are the relERT value on all the five instances (IIDs: 1, 2, 3, 4, and 5)
        alg_data_name = '{}_liid0'.format(alg)        
        y_test_dict[alg] = test_df[alg_data_name].values        
    test_df = test_df.drop(columns=problem_info_columns+ap_columns)
    X_test = test_df.values

    # train datasets
    train_df = table_df[table_df['instance'] != left_instance_id]    
    y_train_dict = {}
    for alg in ap_algs:
        # The relERT value on four instances (except for left_instance_id) is used for the training phase.
        # For example, if left_instance_id = 2, IIDs = 1, 3, 4, and 5. 
        alg_data_name = '{}_liid{}'.format(alg, left_instance_id)
        y_train_dict[alg] = train_df[alg_data_name].values
    train_df = train_df.drop(columns=problem_info_columns+ap_columns)
    X_train = train_df.values
    
    # For each feature, normalize feature values into the range [-1, 1].
    scaler =  MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    # Perform a clustering by g-means.
    gmeans_instance = gmeans(X_train, repeat=10, random_state=ml_seed).process()
    clusters = gmeans_instance.get_clusters()
    centers = gmeans_instance.get_centers()
    n_clusters = len(centers)

    # Determine the best solver for each cluster. 
    cluster_best_algs = []    
    for cluster in clusters:
        vote_df = pd.DataFrame(index=cluster, columns=ap_algs)
        vote_df.fillna(0, inplace=True)
        for iid in cluster:
            for alg in ap_algs:
                vote_df.loc[iid, alg] = y_train_dict[alg][iid]                
        tmp_df = vote_df.rank(method='min', axis=1)
        tmp_df = tmp_df.mean(numeric_only=True)
        l = tmp_df[tmp_df == tmp_df.min()].index
        cluster_best_algs.append(random.choice(l))
    cluster_best_algs = np.array(cluster_best_algs)
    
    # Test
    X_test = scaler.transform(X_test)    
    pred_cluster_labels = gmeans_instance.predict(X_test)        
    pred_algs = cluster_best_algs[pred_cluster_labels]
    
    for fun_id, pred_alg in zip(test_fun_ids, pred_algs):
        pred_alg_file_path = os.path.join(as_result_dir_path, 'selected_alg_f{}_DIM{}_i{}.csv'.format(fun_id, dim, left_instance_id))        
        with open(pred_alg_file_path, 'w') as fh:
            fh.write(pred_alg)
    
@click.command()
@click.option('--ap_name', '-ap', required=False, default='kt_ecj19', type=str, help='Name of an algorithm portfolio')
@click.option('--dir_sampling_method', '-dsample', required=False, default='ihs_multiplier50_sid0', type=str, help='Directory of a sampling method.')
@click.option('--ela_feature_classes', '-f', required=False, default='basic', type=str, help='All feature classes')
@click.option('--dims', '-ds', required=False, default='dims2_3_5_10', type=str, help='All dimensions considered')
@click.option('--selector', '-sel', required=False, default='hiearchical_regression', type=str, help='Algorithm selector')
@click.option('--cross_valid_type', '-cv', required=False, default='lopo_cv', type=str, help='Crossvalidation method.')
@click.option('--dim', '-d', required=False, default=2, type=int, help='Dimension.')
@click.option('--left_dim', '-ldim', required=False, default=1, type=int, help='Dimension to be left in LOPO-CV-AD.')
@click.option('--left_fun_id', '-lfid', required=False, default=1, type=int, help='Function ID to be left in LOPO-CV and LOPO-CV-AD.')
@click.option('--left_instance_id', '-liid', required=False, default=1, type=int, help='Instance ID to be left in LOIO-CV.')
@click.option('--as_run_id', '-run_id', required=False, default=1, type=int, help='Run ID.')
@click.option('--feature_selector', '-fs', required=False, default='none', type=str, help='Feature selector.')
@click.option('--n_features_to_select', '-nf', required=False, default=0, type=int, help='Number of features to select.')
@click.option('--per_metric', '-pm', required=False, default='sp1', type=str, help='A performance metric.')
def run(ap_name, dir_sampling_method, as_run_id, ela_feature_classes, dims, selector, cross_valid_type, dim, left_dim, left_fun_id, left_instance_id, feature_selector, n_features_to_select, per_metric):
    np.random.seed(seed=as_run_id)
    random.seed(as_run_id)
    # ml_seed determines the seed of random state in a sklearn model
    ml_seed = as_run_id
    
    bbob_suite = 'bbob'
    all_fun_ids = range(1, 24+1)
    if bbob_suite == 'bbob-noisy':
        all_fun_ids = range(101, 130+1)
        
    ap_dir_path = os.path.join('./alg_portfolio', ap_name)
    sample_dir_path = os.path.join('./sample_data', dir_sampling_method)
    table_data_name = dir_sampling_method + '_' + per_metric + '_' + ela_feature_classes + '_' + dims
    table_data_file_path = os.path.join(ap_dir_path, 'feature_table_data', '{}.csv'.format(table_data_name))

    as_result_dir_path = os.path.join('as_results', '{}_{}_{}_{}_{}'.format(ap_name, selector, cross_valid_type, table_data_name, feature_selector))
    if feature_selector != 'none':
        as_result_dir_path += '_nfs{}'.format(n_features_to_select)    
    os.makedirs(as_result_dir_path, exist_ok=True)
    
    config_file_path = os.path.join(as_result_dir_path, 'config.json')
    if os.path.isfile(config_file_path) == False:    
        config_dict = {}
        config_dict['ap_name'] = ap_name
        config_dict['selector'] = selector
        config_dict['feature_selector'] = feature_selector
        config_dict['n_features_to_select'] = n_features_to_select
        config_dict['cross_valid_type'] = cross_valid_type
        config_dict['ela_feature_classes'] =ela_feature_classes
        config_dict['dir_sampling_method'] = dir_sampling_method
        config_dict['sample_dir_path'] = sample_dir_path
        config_dict['ap_dir_path'] = ap_dir_path
        config_dict['per_metric'] = per_metric
        with open(config_file_path, 'w') as fh:
            json.dump(config_dict, fh, indent=4)
    
    problem_info_columns = ['dim', 'fun', 'instance']
    ap_config_file_path = os.path.join(ap_dir_path, 'ap_config.csv')
    ap_algs = np.loadtxt(ap_config_file_path, delimiter=",", comments="#", dtype=np.str)    
    ap_columns = ap_column_list(ap_algs)
    
    if selector == 'multiclass_classification' and cross_valid_type == 'lopo_cv':
        multiclass_classification_lopo_cv(as_result_dir_path, table_data_file_path, dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)
    elif selector == 'multiclass_classification' and cross_valid_type == 'lopoad_cv':
        multiclass_classification_lopoad_cv(as_result_dir_path, table_data_file_path, left_dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)
    elif selector == 'multiclass_classification' and cross_valid_type == 'loio_cv':
        multiclass_classification_loio_cv(as_result_dir_path, table_data_file_path, dim, left_instance_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)
    elif selector == 'hiearchical_regression' and cross_valid_type == 'lopo_cv':
        hiearchical_regression_lopo_cv(as_result_dir_path, table_data_file_path, dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)        
    elif selector == 'hiearchical_regression' and cross_valid_type == 'lopoad_cv':
        hiearchical_regression_lopoad_cv(as_result_dir_path, table_data_file_path, left_dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)        
    elif selector == 'hiearchical_regression' and cross_valid_type == 'loio_cv':
        hiearchical_regression_loio_cv(as_result_dir_path, table_data_file_path, dim, left_instance_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)        
    elif selector == 'pairwise_classification' and cross_valid_type == 'lopo_cv':
        pairwise_classification_lopo_cv(as_result_dir_path, table_data_file_path, dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)
    elif selector == 'pairwise_classification' and cross_valid_type == 'lopoad_cv':
        pairwise_classification_lopoad_cv(as_result_dir_path, table_data_file_path, left_dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)    
    elif selector == 'pairwise_classification' and cross_valid_type == 'loio_cv':
        pairwise_classification_loio_cv(as_result_dir_path, table_data_file_path, dim, left_instance_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)
    elif selector == 'pairwise_regression' and cross_valid_type == 'lopo_cv':
        pairwise_regression_lopo_cv(as_result_dir_path, table_data_file_path, dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)
    elif selector == 'pairwise_regression' and cross_valid_type == 'lopoad_cv':
        pairwise_regression_lopoad_cv(as_result_dir_path, table_data_file_path, left_dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)
    elif selector == 'pairwise_regression' and cross_valid_type == 'loio_cv':
        pairwise_regression_loio_cv(as_result_dir_path, table_data_file_path, dim, left_instance_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)
    elif selector == 'clustering' and cross_valid_type == 'lopo_cv':
        clustering_lopo_cv(as_result_dir_path, table_data_file_path, dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)        
    elif selector == 'clustering' and cross_valid_type == 'lopoad_cv':
        clustering_lopo_cv(as_result_dir_path, table_data_file_path, left_dim, left_fun_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)
    elif selector == 'clustering' and cross_valid_type == 'loio_cv':
        clustering_loio_cv(as_result_dir_path, table_data_file_path, dim, left_instance_id, ml_seed, cross_valid_type, problem_info_columns, ap_columns, ap_algs, feature_selector, n_features_to_select)
    else:
        logger.error('The selector %s is not defined for the CV %s', selector, cross_valid_type)
        exit(1)
        
if __name__ == '__main__':
    run()
