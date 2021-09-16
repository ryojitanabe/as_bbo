"""
Postprocessing results of algorithm selection.
"""
import os
import numpy as np
import pandas as pd
import pylab as plt
import matplotlib.ticker as mtick
import seaborn as sns
import csv
import logging
import statistics as stat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True

mat_markers = ['o',  '^', 's', 'd', 'p', 'h', '8', '*', 'X', 'D']
                       
def plot_box_metric_funcs(fig_file_path, as_result_dir_path, dim, sids=range(0,31), all_fun_ids=range(1,24+1), per_metric='sp1'):
    fig = plt.figure(figsize=(17, 3))
    ax = plt.subplot(111)
    ax.set_rasterization_zorder(1)

    all_relert_values = []

    for fun_id in all_fun_ids:
        relert_values = []
        for sid in sids:
            relert_dir_path = os.path.join(as_result_dir_path.replace('sid', 'sid{}'.format(sid)), 'rel'+per_metric, 'f{}_DIM{}_liid0.csv'.format(fun_id, dim))
            with open(relert_dir_path, 'r') as fh:
                relert_values.append(float(fh.read()))
        all_relert_values.append(relert_values)
        
    ax.boxplot(all_relert_values, labels=all_fun_ids)    

    plt.yscale('log')
    plt.grid(True, which="both", ls="-", color='0.85')    
    ax.set_axisbelow(True)

    plt.ylim(ymin=0.1, ymax=1e5)    
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel("Function IDs", fontsize=34)            
    plt.ylabel("Mean rel"+per_metric.upper(), fontsize=34)            
    plt.savefig(fig_file_path, bbox_inches='tight')   
    plt.close()

def plot_box_mean_metric_val(fig_file_path, as_result_dir_path, sids=range(0,31), all_fun_ids=range(1, 24+1), dims=[2, 3, 5, 10], per_metric='sp1', cv_type='loio_cv'):
    fig = plt.figure()
    ax = plt.subplot(111)

    mean_relert_list_dims = []
    for dim in dims:
        mean_relert_list = []
        for sid in sids:
            as_result_dir_path_ = os.path.join(as_result_dir_path.replace('sid', 'sid{}'.format(sid)))             
            
            relert_values = []
            for fun_id in all_fun_ids:
                rel_ert_file_path = os.path.join(as_result_dir_path_, 'rel'+per_metric, 'f{}_DIM{}_liid0.csv'.format(fun_id, dim))
                with open(rel_ert_file_path, 'r') as fh:
                    relert_values.append(float(fh.read()))
            mean_relert_list.append(stat.mean(relert_values))
        mean_relert_list_dims.append(mean_relert_list)
            
    sns.boxplot(data=mean_relert_list_dims)    

    ax.set_xticklabels(dims)
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", color='0.85')
    ax.set_axisbelow(True)
    plt.ylim(ymin=1, ymax=10000)
    if cv_type == 'loio_cv':
        plt.ylim(ymin=1, ymax=1000)
    
    plt.xticks(fontsize=31)
    plt.yticks(fontsize=31)
    plt.xlabel("Dimensions", fontsize=34)            
    plt.ylabel("Mean rel"+per_metric.upper(), fontsize=34)            
    plt.savefig(fig_file_path, bbox_inches='tight') # , rasterized=True
    plt.close()

def plot_box_mean_metric_val_3cv(fig_file_path, as_result_dir_path, sids=range(0,31), all_fun_ids=range(1, 24+1), dims=[2, 3, 5, 10], per_metric='sp1'):
    fig = plt.figure()
    ax = plt.subplot(111)
    metric_df = pd.DataFrame(columns=['dim', 'metric', 'cv'])

    for cv_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:        
        for dim in dims:
            for sid in sids:
                as_result_dir_path_ = as_result_dir_path.replace('cv', cv_type)
                as_result_dir_path_ = as_result_dir_path_.replace('sid', 'sid{}'.format(sid))

                relert_values = []
                for fun_id in all_fun_ids:
                    rel_ert_file_path = os.path.join(as_result_dir_path_, 'rel'+per_metric, 'f{}_DIM{}_liid0.csv'.format(fun_id, dim))
                    with open(rel_ert_file_path, 'r') as fh:
                        relert_values.append(float(fh.read()))

                cv = cv_type.upper().replace('_CV', '')
                metric_df = metric_df.append({'dim':dim, 'metric':stat.mean(relert_values), 'cv':cv},  ignore_index=True)
            
    sns.boxplot(x='dim', y='metric', data=metric_df, hue='cv', ax=ax)

    ax.set_xticklabels(dims)    
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", color='0.85')
    ax.set_axisbelow(True)    
    plt.ylim(ymin=1, ymax=10000)    
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel("Dimensions", fontsize=35)            
    plt.ylabel("Mean rel"+per_metric.upper(), fontsize=35)            
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1.05), fontsize=28)
    plt.savefig(fig_file_path, bbox_inches='tight') # , rasterized=True
    plt.close()    
    
if __name__ == '__main__':    
    bbob_suite = 'bbob'
    all_fun_ids = range(1, 24+1)
    if bbob_suite == 'bbob-noisy':
        all_fun_ids = range(101, 130+1)    

    ela_feature_classes = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    dims = 'dims2_3_5_10'

    per_metric = 'sp1'
    #per_metric = 'ert'
    sampling_method = 'ihs'
    sample_multiplier = 50
           
    ap_name = 'kt'
    feature_selector = 'none'
    n_features_to_select = 0
    
    dir_sampling_method = '{}_multiplier{}_sid_{}'.format(sampling_method, sample_multiplier, per_metric)
    table_data_name = dir_sampling_method + '_' + ela_feature_classes + '_' + dims

    for selector in ['multiclass_classification', 'hiearchical_regression', 'pairwise_classification', 'pairwise_regression', 'clustering']:
        as_result_dir_path = os.path.join('pp_as_results', '{}_{}_cv_{}_{}'.format(ap_name, selector, table_data_name, feature_selector))
        if feature_selector != 'none':
            as_result_dir_path += '_nfs{}'.format(n_features_to_select)    
        # For a box plot of mean measure values
        fig_file_path = os.path.join('pp_figs', 'dist_mean_rel'+per_metric+'_3cv')
        os.makedirs(fig_file_path, exist_ok=True)

        fig_file_path = os.path.join(fig_file_path, '{}_{}_{}_{}'.format(ap_name, selector, table_data_name, feature_selector))
        if feature_selector != 'none':
            fig_file_path += '_nfs{}'.format(n_features_to_select)    
        fig_file_path += '.pdf'

        plot_box_mean_metric_val_3cv(fig_file_path, as_result_dir_path, sids=range(0,31), all_fun_ids=all_fun_ids, dims=[2, 3, 5, 10], per_metric=per_metric)
        logger.info("A figure was generated: %s", fig_file_path)
