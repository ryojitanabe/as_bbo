"""
Postprocessing results of algorithm selection.
"""
import os
import numpy as np
import csv
import statistics as stat
import shutil
from scipy import stats

def summarized_table_rel_ert(all_optimizer_info, dims, all_fun_ids, sids, cv_type, per_metric, ap_symbol='A'):
    sbs_mean_rel_ert_dict = {}
    for dim in dims:        
        relert_values = []
        for fun_id in all_fun_ids:
            rel_ert_file_path = os.path.join('pp_as_results/sbs_{}_{}_DIM{}'.format(ap_name, per_metric, dim), 'rel'+per_metric, 'f{}_DIM{}_liid0.csv'.format(fun_id, dim))
            with open(rel_ert_file_path, 'r') as fh:
                relert_values.append(float(fh.read()))                    
        sbs_mean_rel_ert_dict[dim] = stat.mean(relert_values)                    

    print("%%%%%%%%%%%")
    print("\\subfloat[{} ({})]{{".format(cv_type.upper().replace('_', '-'), ap_symbol))
    print("\\begin{tabular}{lllllll}")
    print("\\toprule")

    print("System & $n=2$ & $n=3$ & $n=5$ & $n=10$\\\\")
    print("\\midrule")                    

    base_rel_ert = {}
    for optimizer_info in all_optimizer_info:
        if optimizer_info['nickname'] == 'SBS':            
            print("\\midrule")                    

        print("{}".format(optimizer_info['nickname']), end='')

        for dim in dims:
            # the median value of the mean relERT over 31 runs.
            all_stat_rel_ert = []    
            
            for sid in sids:
                relert_values = []
                for fun_id in all_fun_ids:
                    rel_ert_file_path = os.path.join(optimizer_info['pp_res_dir'], 'rel'+per_metric, 'f{}_DIM{}_liid0.csv'.format(fun_id, dim))                    
                    s_rel_ert_file_path = rel_ert_file_path.replace('sid', 'sid{}'.format(sid))
                    with open(s_rel_ert_file_path, 'r') as fh:
                        relert_values.append(float(fh.read()))
                mean_relert = stat.mean(relert_values)
                all_stat_rel_ert.append(mean_relert)

            stat_value = stat.median(all_stat_rel_ert)
            stat_sign = ''            
            #stat_sign = '$\\approx$'
            
            # I assume that the first algorithm is the base line.
            if optimizer_info['stat_base'] == True:
                base_rel_ert[dim] = all_stat_rel_ert
            else:                
                stat_res = stats.mannwhitneyu(base_rel_ert[dim], all_stat_rel_ert, alternative='two-sided')

                if stat_res.pvalue < 0.05 and stat_value < stat.median(base_rel_ert[dim]):
                    stat_sign = '$+$'
                elif stat_res.pvalue < 0.05 and stat_value > stat.median(base_rel_ert[dim]):
                    stat_sign = '$-$'                

            if stat_value < sbs_mean_rel_ert_dict[dim]:
                print(" & \\cellcolor{{c1}}{:.2f}{}".format(stat_value, stat_sign), end='')
            else:
                print(" & {:.2f}{}".format(stat_value, stat_sign), end='')

        print("\\\\")

    print("\\midrule")        
    print("SBS", end='')
    for dim in dims:
        print(" & {:.2f}".format(sbs_mean_rel_ert_dict[dim]), end='')
    print("\\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
                               
if __name__ == '__main__':
    bbob_suite = 'bbob'
    all_fun_ids = range(1, 24+1)
    if bbob_suite == 'bbob-noisy':
        all_fun_ids = range(101, 130+1)

    dims = [2, 3, 5, 10]
    sids = range(0, 31)
    
    selector = 'multiclass_classification'
    #selector = 'hiearchical_regression'    
    #selector = 'pairwise_classification' 
    #selector = 'pairwise_regression'
    #selector = 'clustering'

    ap_nickname_dict =  {'mk_21':'$\mathcal{A}_{\mathrm{mk}}$', 'jped_gecco21':'$\mathcal{A}_{\mathrm{jped}}$', 'dlvat_foga19':'$\mathcal{A}_{\mathrm{dlvat}}$', 'fixed_dlvat_foga19':'$\mathcal{A}_{\mathrm{dlvat}}$', 'kt_ecj19':'$\mathcal{A}_{\mathrm{kt}}$', 'bmtp_gecco12':'$\mathcal{A}_{\mathrm{bmtp}}$', 'lsap_k2_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}2}$', 'lsap_k4_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}4}$', 'lsap_k6_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}6}$', 'lsap_k8_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}8}$', 'lsap_k10_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}10}$', 'lsap_k12_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}12}$', 'lsap_k14_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}14}$', 'lsap_k16_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}16}$', 'lsap_k18_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}18}$'}
    
    per_metric = 'sp1'    
    
    for ap_name in ['kt_ecj19']:                                           
        for cv_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:    
            all_optimizer_info = []
            # 
            d = {}
            d['pp_res_dir'] = 'pp_as_results/{}_{}_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none'.format(ap_name, selector, cv_type)
            d['nickname'] = 'AS ($50 \\times n$)'
            d['stat_base'] = True
            all_optimizer_info.append(d)

            # # 
            # d = {}
            # d['pp_res_dir'] = 'pp_as_results/{}_{}_{}_ihs_multiplier100_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none'.format(ap_name, selector, cv_type)
            # d['nickname'] = 'AS ($100 \\times n$)'
            # d['stat_base'] = False
            # all_optimizer_info.append(d)

            # 
            d = {}
            d['pp_res_dir'] = 'pp_as_results/{}_{}_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none_slsqp_multiplier50'.format(ap_name, selector, cv_type)
            d['nickname'] = 'SLSQP-AS'
            d['stat_base'] = False
            all_optimizer_info.append(d)

            summarized_table_rel_ert(all_optimizer_info, dims, all_fun_ids, sids, cv_type, per_metric, ap_symbol=ap_nickname_dict[ap_name])
        print("\\\\")
