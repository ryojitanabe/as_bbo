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
                       
def calc_n_sbs(ap, n_sbs_dir_path, as_result_dir_path, cv_type, sids=range(0,31), all_fun_ids=range(1, 24+1), dims=[2, 3, 5, 10], per_metric='sp1'):
    
    for dim in dims:
        # Read the metric value of the SBS
        relert_values = []
        for fun_id in all_fun_ids:
            rel_ert_file_path = os.path.join('pp_as_results/sbs_{}_{}_DIM{}'.format(ap, per_metric, dim), 'rel'+per_metric, 'f{}_DIM{}_liid0.csv'.format(fun_id, dim))
            with open(rel_ert_file_path, 'r') as fh:
                relert_values.append(float(fh.read()))                    
        sbs_mean_relmetric = stat.mean(relert_values)
        
        n_beat_sbs = 0        
        for sid in sids:    
            as_result_dir_path_ = as_result_dir_path.replace('sid', 'sid{}'.format(sid))
            relert_values = []
            for fun_id in all_fun_ids:
                rel_ert_file_path = os.path.join(as_result_dir_path_, 'rel'+per_metric, 'f{}_DIM{}_liid0.csv'.format(fun_id, dim))
                with open(rel_ert_file_path, 'r') as fh:
                    relert_values.append(float(fh.read()))

            mean_relmetric = stat.mean(relert_values)
            if mean_relmetric < sbs_mean_relmetric:
                n_beat_sbs += 1
                    
        n_sbs_file_path = os.path.join(n_sbs_dir_path, '{}_DIM{}_{}.csv'.format(ap, dim, cv_type))
        with open(n_sbs_file_path, 'w') as fh:
            fh.write(str(n_beat_sbs))

def plot_n_beat_sbs(fig_file_path, n_sbs_dir_path, dim, all_aps, all_ap_symbols):
    # fig = plt.figure()
    fig = plt.figure(figsize=(9, 3))
    ax = plt.subplot(111)
    ax.set_rasterization_zorder(1)

    for cv_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:                   
        n_beat_sbs_list = []

        for ap in all_aps:
            n_sbs_file_path = os.path.join(n_sbs_dir_path, '{}_DIM{}_{}.csv'.format(ap, dim, cv_type))
            print(n_sbs_file_path)
            with open(n_sbs_file_path, 'r') as fh:
                n_beat_sbs_list.append(float(fh.read()))

        ax.plot(all_ap_symbols, n_beat_sbs_list, lw=2, marker=mat_markers[0], markersize=10, markerfacecolor="None", markeredgewidth=2, label=cv_type.upper().replace('_', '-'))

    plt.grid(True, which="both", ls="-", color='0.85')    
        
    plt.ylim(ymin=0, ymax=31)    
    plt.yticks(fontsize=27)
    ax.set_xticklabels(all_ap_symbols, rotation='vertical', fontsize=27)
    plt.xlabel("Algorithm portfolios", fontsize=30)
    plt.ylabel("$N^{\mathrm{SBS}}$", fontsize=30)        
    plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.4), ncol=3, fontsize=25, columnspacing=0.3)    
    plt.savefig(fig_file_path, bbox_inches='tight')
    plt.close()
        
if __name__ == '__main__':    
    bbob_suite = 'bbob'
    all_fun_ids = range(1, 24+1)
    if bbob_suite == 'bbob-noisy':
        all_fun_ids = range(101, 130+1)    

    ela_feature_classes = 'basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta'
    dims = 'dims2_3_5_10'

    per_metric = 'sp1'
    sampling_method = 'ihs'
    sample_multiplier = 50

    ap_nickname_dict =  {'mk_21':'mk', 'jped_gecco21':'jped', 'dlvat_foga19':'dlvat', 'fixed_dlvat_foga19':'dlvat', 'kt_ecj19':'kt', 'bmtp_gecco12':'bmtp', 'lsap_k2_sp1_target-2.0_dims2_3_5_10':'ls2', 'lsap_k4_sp1_target-2.0_dims2_3_5_10':'ls4', 'lsap_k6_sp1_target-2.0_dims2_3_5_10':'ls6', 'lsap_k8_sp1_target-2.0_dims2_3_5_10':'ls8', 'lsap_k10_sp1_target-2.0_dims2_3_5_10':'ls10', 'lsap_k12_sp1_target-2.0_dims2_3_5_10':'ls12', 'lsap_k14_sp1_target-2.0_dims2_3_5_10':'ls14', 'lsap_k16_sp1_target-2.0_dims2_3_5_10':'ls16', 'lsap_k18_sp1_target-2.0_dims2_3_5_10':'ls18'}            

    all_aps = ['kt_ecj19', 'fixed_dlvat_foga19', 'jped_gecco21', 'bmtp_gecco12', 'mk_21', 'lsap_k2_sp1_target-2.0_dims2_3_5_10', 'lsap_k4_sp1_target-2.0_dims2_3_5_10', 'lsap_k6_sp1_target-2.0_dims2_3_5_10', 'lsap_k8_sp1_target-2.0_dims2_3_5_10', 'lsap_k10_sp1_target-2.0_dims2_3_5_10', 'lsap_k12_sp1_target-2.0_dims2_3_5_10', 'lsap_k14_sp1_target-2.0_dims2_3_5_10', 'lsap_k16_sp1_target-2.0_dims2_3_5_10', 'lsap_k18_sp1_target-2.0_dims2_3_5_10']
    common_ap_dir_path = './pp_14_ap'

    dir_sampling_method = '{}_multiplier{}_sid_{}'.format(sampling_method, sample_multiplier, per_metric)
    table_data_name = dir_sampling_method + '_' + ela_feature_classes + '_' + dims
    
    for selector in ['multiclass_classification', 'hiearchical_regression', 'pairwise_classification', 'pairwise_regression', 'clustering']:    
        for ap in all_aps:
            for cv_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:               
                as_result_dir_path = os.path.join('pp_as_results', '{}_{}_{}_{}_none_slsqp_multiplier50'.format(ap, selector, cv_type, table_data_name))

                n_sbs_dir_path = os.path.join(common_ap_dir_path, 'n_sbs', selector)
                os.makedirs(n_sbs_dir_path, exist_ok=True)            
                calc_n_sbs(ap, n_sbs_dir_path, as_result_dir_path, cv_type, sids=range(0,31), all_fun_ids=all_fun_ids, dims=[2, 3, 5, 10], per_metric=per_metric)                        

    all_ap_symbols = []
    for ap in all_aps:
        all_ap_symbols.append(ap_nickname_dict[ap])        
    
    for selector in ['multiclass_classification', 'hiearchical_regression', 'pairwise_classification', 'pairwise_regression', 'clustering']:
        for dim in [2, 3, 5, 10]:
            n_sbs_dir_path = os.path.join(common_ap_dir_path, 'n_sbs', selector)
            
            fig_dir_path = os.path.join('pp_figs', 'n_beat_sbs')
            os.makedirs(fig_dir_path, exist_ok=True)
            fig_file_path = os.path.join(fig_dir_path, '{}_DIM{}.pdf'.format(selector, dim))        

            plot_n_beat_sbs(fig_file_path, n_sbs_dir_path, dim, all_aps, all_ap_symbols)
            logger.info("A figure was generated: %s", fig_file_path)
