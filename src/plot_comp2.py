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
#logging.basicConfig(filename='{}.log'.format(__file__), level=logging.DEBUG)

plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True

mat_markers = ["o","v","^","<",">","s","p","P","*","h","H", "D"]

def plot_comp2(fig_file_path, pair_selectors, dim, sids=range(0,31), all_fun_ids=range(1, 24+1), per_metric='sp1'):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_rasterization_zorder(1)
    ax.set_aspect('equal')

    median_relmetric_values = []    

    for as_id, as_dict in enumerate(pair_selectors):
        print(as_dict)
        
        mean_relmetric_values = []
        for sid in sids:
            relmetric_values = []
            for fun_id in all_fun_ids:
                relmetric_file_path = os.path.join(as_dict['pp_res_dir'], 'rel'+per_metric, 'f{}_DIM{}_liid0.csv'.format(fun_id, dim))                    
                relmetric_file_path_ = relmetric_file_path.replace('sid', 'sid{}'.format(sid))
                with open(relmetric_file_path_, 'r') as fh:
                    relmetric_values.append(float(fh.read()))     
            mean_relmetric = stat.mean(relmetric_values)
            mean_relmetric_values.append(mean_relmetric)

        if as_id == 0:
            median_run_id = np.argsort(np.array(mean_relmetric_values))[len(mean_relmetric_values)//2]
            
        relmetric_values = []
        for fun_id in all_fun_ids:
            relmetric_file_path = os.path.join(as_dict['pp_res_dir'], 'rel'+per_metric, 'f{}_DIM{}_liid0.csv'.format(fun_id, dim))                    
            relmetric_file_path_ = relmetric_file_path.replace('sid', 'sid{}'.format(median_run_id))
            with open(relmetric_file_path_, 'r') as fh:
                relmetric_values.append(float(fh.read()))     

        median_relmetric_values.append(relmetric_values)
        
    cmap = plt.get_cmap("tab20")
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    cmap = plt.get_cmap("tab20_r")
    color_list.extend(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    color_list.append('k')
    color_list.append('y')
    color_list.append('m')
    color_list.append('c')
            
    for i, (x, y) in enumerate(zip(median_relmetric_values[1], median_relmetric_values[0])):        
        if i < 12:
            ax.scatter(x, y, s=100, c=color_list[i], marker=mat_markers[i%12], label='{}'.format(i+1))
        else:
            ax.scatter(x, y, s=100, facecolor='None', edgecolors=color_list[i], linewidth=2, marker=mat_markers[i%12], label='{}'.format(i+1))            
                
    # plt.xscale('log')
    # plt.yscale('log')
    plt.grid(True, which="both", ls="-", color='0.85')    
    plt.xlim(xmin=0, xmax=30)
    plt.ylim(ymin=0, ymax=30)

    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="-", c=".3")
    plt.xticks(fontsize=31)
    plt.yticks(fontsize=31)
    plt.xlabel('RelSP1 of '+ pair_selectors[1]['nickname'], fontsize=30)  
    plt.ylabel('RelSP1 of '+ pair_selectors[0]['nickname'], fontsize=30)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.08), ncol=2, fontsize=20, columnspacing=0.3)

    # For the zoom-up subfig
    sub_axes = plt.axes([.52, .18, .28, .28])
    sub_axes.set_aspect('equal')
    
    sub_axes.tick_params(which='both', direction='in', top=bool, right=bool, labelbottom=True)
    sub_axes.tick_params(labelsize=18)
    
    sub_axes.set_xlim(0, 5)
    sub_axes.set_ylim(0, 5)
    sub_axes.set_xticks([0, 1, 2, 3, 4, 5])
    sub_axes.set_yticks([0, 1, 2, 3, 4, 5])
    
    sub_axes.plot(sub_axes.get_xlim(), sub_axes.get_ylim(), ls="-", c=".3", zorder=-1)

    for i, (x, y) in enumerate(zip(median_relmetric_values[1], median_relmetric_values[0])):        
        if i < 12:
            sub_axes.scatter(x, y, s=100, c=color_list[i], marker=mat_markers[i%12], label='{}'.format(i+1))
        else:
            sub_axes.scatter(x, y, s=100, facecolor='None', edgecolors=color_list[i], linewidth=2, marker=mat_markers[i%12], label='{}'.format(i+1))
            
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
    ap_name = 'kt'
    dim = 5
    
    selector = 'multiclass_classification'
    #selector = 'hiearchical_regression'    
    #selector = 'pairwise_classification'
    #selector = 'pairwise_regression'
    #selector = 'clustering'

    cv_type = 'loio_cv'
    #cv_type = 'lopo_cv'    
    #cv_type = 'lopoad_cv'
            
    pair_selectors = []    
    # AS1
    d = {}
    d['pp_res_dir'] = 'pp_as_results/{}_{}_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none'.format(ap_name, selector, cv_type)
    d['nickname'] = 'AS ($50 \\times n$)'
    pair_selectors.append(d)

    # AS2
    d = {}
    d['pp_res_dir'] = 'pp_as_results/{}_{}_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none_slsqp_multiplier50'.format(ap_name, selector, cv_type)
    d['nickname'] = 'SLSQP-AS'
    pair_selectors.append(d)

    fig_file_path = os.path.join('pp_figs', 'presolver_comp2'+per_metric)    
    os.makedirs(fig_file_path, exist_ok=True)

    fig_file_path = os.path.join(fig_file_path, '{}_{}_{}'.format(ap_name, selector, cv_type))
    fig_file_path += '_DIM{}.pdf'.format(dim)

    plot_comp2(fig_file_path, pair_selectors, dim, sids=range(0,31), all_fun_ids=all_fun_ids, per_metric=per_metric)
