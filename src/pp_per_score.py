"""
Postprocessing results of algorithm selection.
"""
import os
import numpy as np
import csv
import statistics as stat
import shutil
from scipy import stats

def calc_ps_mean_metric(res_dir_path, all_optimizer_info, dim, all_fun_ids, sids, cv_type, per_metric):

    for opt1 in all_optimizer_info:
        sum_lost = 0
        
        opt1_metric_list = []        
        for sid in sids:
            fun_metric_list = []            
            for fun_id in all_fun_ids:                   
                metric_file_path = os.path.join(opt1['pp_res_dir'], 'rel'+per_metric, 'f{}_DIM{}_liid0.csv'.format(fun_id, dim))                    
                metric_file_path = metric_file_path.replace('sid', 'sid{}'.format(sid))
                with open(metric_file_path, 'r') as fh:
                    fun_metric_list.append(float(fh.read()))

            mean_metric_value = stat.mean(fun_metric_list)
            opt1_metric_list.append(mean_metric_value)                    
            
        for opt2 in all_optimizer_info:                
            if opt1['nickname'] == opt2['nickname']:
                continue

            opt2_metric_list = []
            for sid in sids:
                fun_metric_list = []            
                for fun_id in all_fun_ids:                   
                    metric_file_path = os.path.join(opt2['pp_res_dir'], 'rel'+per_metric, 'f{}_DIM{}_liid0.csv'.format(fun_id, dim))
                    metric_file_path = metric_file_path.replace('sid', 'sid{}'.format(sid))
                    with open(metric_file_path, 'r') as fh:
                        fun_metric_list.append(float(fh.read()))

                mean_metric_value = stat.mean(fun_metric_list)
                opt2_metric_list.append(mean_metric_value)                    

            # print(opt1_metric_list, opt2_metric_list)
            # exit()
            if opt1_metric_list != opt2_metric_list:                
                stat_res = stats.mannwhitneyu(opt1_metric_list, opt2_metric_list, alternative='two-sided')
                if stat_res.pvalue < 0.05 and stat.median(opt1_metric_list) > stat.median(opt2_metric_list):
                        sum_lost += 1

        aps = float(sum_lost)
        print(dim, opt1['nickname'], aps, sum_lost)        

        res_file_path = os.path.join(res_dir_path, '{}_DIM{}.csv'.format(opt1['nickname'], dim))
        with open(res_file_path, 'w') as fh:
            fh.write(str(aps))

def run(all_fun_ids, dims, sids, per_metric, ap_name):
    for cv_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:    
        res_dir_path = os.path.join('./pp_as_results', 'aps_comp_selectors_mean_sp1', '{}_{}'.format(ap_name, cv_type))
        os.makedirs(res_dir_path, exist_ok=True)

        all_optimizer_info = []
        # 
        d = {}
        d['pp_res_dir'] = 'pp_as_results/{}_multiclass_classification_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none_slsqp_multiplier50'.format(ap_name, cv_type)
        d['nickname'] = 'Classification'
        all_optimizer_info.append(d)
        
        # 
        d = {}
        d['pp_res_dir'] = 'pp_as_results/{}_hiearchical_regression_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none_slsqp_multiplier50'.format(ap_name, cv_type)
        d['nickname'] = 'Regression'
        all_optimizer_info.append(d)

        # 
        d = {}
        d['pp_res_dir'] = 'pp_as_results/{}_pairwise_classification_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none_slsqp_multiplier50'.format(ap_name, cv_type)
        d['nickname'] = 'P-classification'
        all_optimizer_info.append(d)
        
        # 
        d = {}
        d['pp_res_dir'] = 'pp_as_results/{}_pairwise_regression_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none_slsqp_multiplier50'.format(ap_name, cv_type)
        d['nickname'] = 'P-regression'
        all_optimizer_info.append(d)

        #
        d = {}
        d['pp_res_dir'] = 'pp_as_results/{}_clustering_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none_slsqp_multiplier50'.format(ap_name, cv_type)
        d['nickname'] = 'Clustering'
        all_optimizer_info.append(d)
        
        for dim in dims:
            calc_ps_mean_metric(res_dir_path, all_optimizer_info, dim, all_fun_ids, sids, cv_type, per_metric)

def print_aps(ap_name, ap_symbol, cv_type):
    opt_list = ['Classification', 'Regression', 'P-classification', 'P-regression', 'Clustering']    

    #nickname_dict = {'Classification':'Cla.', 'Regression':'Reg.', 'P-classification':'P-Cla.', 'P-regression':'P-Reg.', 'Clustering':'Clu.'}
    nickname_dict = {'Classification':'Classification.', 'Regression':'Regression', 'P-classification':'P-classification', 'P-regression':'P-regression', 'Clustering':'Clustering'}
        
    best_dict = {}
    second_best_dict = {}
    best_value_dict = {}
    second_best_value_dict = {}

    for dim in dims:
        aps_list = []
        for opt_name in opt_list:
            res_file_path = os.path.join('./pp_as_results', 'aps_comp_selectors_mean_sp1', '{}_{}'.format(ap_name, cv_type), '{}_DIM{}.csv'.format(opt_name, dim))            
            with open(res_file_path, 'r') as fh:
                aps = float(fh.read())
            aps_list.append(aps)

        l = np.argsort(np.array(aps_list))
        best_dict[dim] = opt_list[l[0]]
        second_best_dict[dim] = opt_list[l[1]]

        aps_list = list(set(aps_list))
        aps_list = np.sort(np.array(aps_list))

        best_value_dict[dim] = aps_list[0]

        if len(aps_list) > 1:
            second_best_value_dict[dim] = aps_list[1]        
        else:
            second_best_value_dict[dim] = -1
            
    print("%%%%%%%%%%%")
    print("\\subfloat[{} ({})]{{".format(cv_type.upper().replace('_', '-'), ap_symbol))
    print("\\begin{tabular}{lC{1em}C{1em}C{1em}C{1em}}")
    print("\\toprule")

    print(" & \\multicolumn{4}{c}{$n$}\\\\")
    print("\cmidrule{2-5}")
    print(" & $2$ & $3$ & $5$ & $10$\\\\")    
    print("\\midrule")                    

    for opt_name in opt_list:
        print("{}".format(nickname_dict[opt_name]), end='')

        for dim in dims:
            res_file_path = os.path.join('./pp_as_results', 'aps_comp_selectors_mean_sp1', '{}_{}'.format(ap_name, cv_type), '{}_DIM{}.csv'.format(opt_name, dim))            
            with open(res_file_path, 'r') as fh:
                aps = float(fh.read())

            if np.isclose(aps, best_value_dict[dim]):
                print(" & \\cellcolor{{c1}}{:d}".format(int(aps)), end='')   
            elif  np.isclose(aps, second_best_value_dict[dim]):                
                print(" & \\cellcolor{{c2}}{:d}".format(int(aps)), end='') 
            else:
                print(" & {:d}".format(int(aps)), end='')                
                            
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
    per_metric = 'sp1'    

    ap_name = 'kt_ecj19'

    ap_nickname_dict =  {'mk_21':'$\mathcal{A}_{\mathrm{mk}}$', 'jped_gecco21':'$\mathcal{A}_{\mathrm{jped}}$', 'dlvat_foga19':'$\mathcal{A}_{\mathrm{dlvat}}$', 'fixed_dlvat_foga19':'$\mathcal{A}_{\mathrm{dlvat}}$', 'kt_ecj19':'$\mathcal{A}_{\mathrm{kt}}$', 'bmtp_gecco12':'$\mathcal{A}_{\mathrm{bmtp}}$', 'lsap_k2_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}2}$', 'lsap_k4_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}4}$', 'lsap_k6_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}6}$', 'lsap_k8_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}8}$', 'lsap_k10_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}10}$', 'lsap_k12_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}12}$', 'lsap_k14_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}14}$', 'lsap_k16_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}16}$', 'lsap_k18_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}18}$'}
        
    for ap_name in ['kt_ecj19', 'fixed_dlvat_foga19', 'jped_gecco21', 'bmtp_gecco12', 'mk_21', 'lsap_k2_sp1_target-2.0_dims2_3_5_10', 'lsap_k4_sp1_target-2.0_dims2_3_5_10', 'lsap_k6_sp1_target-2.0_dims2_3_5_10', 'lsap_k8_sp1_target-2.0_dims2_3_5_10', 'lsap_k10_sp1_target-2.0_dims2_3_5_10', 'lsap_k12_sp1_target-2.0_dims2_3_5_10', 'lsap_k14_sp1_target-2.0_dims2_3_5_10', 'lsap_k16_sp1_target-2.0_dims2_3_5_10', 'lsap_k18_sp1_target-2.0_dims2_3_5_10']:
        run(all_fun_ids, dims, sids, per_metric, ap_name)
        
    # ap_name = 'kt_ecj19'            
    # for cv_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:
    #     print_aps(ap_name, ap_nickname_dict[ap_name], cv_type)
    #     print("\\\\")
            
