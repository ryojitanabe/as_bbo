"""
Postprocessing results of algorithm selection.
"""
import os
import numpy as np
import csv
import statistics as stat
import shutil
import logging
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def calc_relmetric(bbob_suite, all_fun_ids, dims, per_metric, target, common_ap_dir_path, all_aps):
    all_algs = []
    for ap_name in all_aps:
        ap_config_file_path = os.path.join('./alg_portfolio', ap_name, 'ap_config.csv')
        ap_algs = np.loadtxt(ap_config_file_path, delimiter=",", comments="#", dtype=str)
        all_algs.extend(list(ap_algs))

    # Delete duplicated algorithms
    all_algs = list(set(all_algs))
    
    # Copy the files for a given metric to reach the target value
    for alg in all_algs:    
        copy_dir_path = os.path.join(common_ap_dir_path, 'config_algs', 'fevals_to_reach', alg)
        os.makedirs(copy_dir_path, exist_ok=True)
        
        for dim in dims:
            for fun_id in all_fun_ids:                
                # Only the first five instances are considered
                for instance_id in range(1, 5+1):                
                    original_file_path = './pp_{}_exdata/fevals_to_reach/{}/f{}_DIM{}_i{}.csv'.format(bbob_suite, alg, fun_id, dim, instance_id)                
                    with open(original_file_path, 'r') as fh:
                        for str_line in fh:
                            tmp_target, feval, is_success = str_line.split(',')
                            
                            if tmp_target == target:
                                feval = int(float(feval.replace('\n','')))
                                is_success = int(float(is_success.replace('\n','')))
                                feval_copy_file_path = os.path.join(copy_dir_path, '{}_f{}_DIM{}_i{}.csv'.format(alg, fun_id, dim, instance_id))
                                with open(feval_copy_file_path, 'w') as fh:
                                    fh.write("{},{}".format(feval, is_success))
                                break                   

    # Copy the files of the SP1
    for alg in all_algs:    
        copy_dir_path = os.path.join(common_ap_dir_path, 'config_algs', per_metric, alg)        
        os.makedirs(copy_dir_path, exist_ok=True)

        for dim in dims:
            for fun_id in all_fun_ids:                
                original_file_path = './pp_{}_exdata/{}/{}/f{}_DIM{}_liid0.csv'.format(bbob_suite, per_metric, alg, fun_id, dim)
                with open(original_file_path, 'r') as fh:
                    for str_line in fh:
                        tmp_target, metric_value = str_line.split(',')                        
                        if tmp_target == target:
                            metric_value = float(metric_value.replace('\n',''))
                            metric_copy_file_path = os.path.join(copy_dir_path, '{}_f{}_DIM{}.csv'.format(alg, fun_id, dim))
                            with open(metric_copy_file_path, 'w') as fh:
                                fh.write("{}".format(metric_value))
                            break
        
    # 3. Calculate the relative metric value and record the best metric value 
    out_dir_path = os.path.join(common_ap_dir_path, 'config_algs', rel_per_metric)
    os.makedirs(out_dir_path, exist_ok=True)
    
    for dim in dims:
        for fun_id in all_fun_ids:
            # Find the best metric value
            metric_values = {}
            for alg in all_algs:
                metric_file_path = os.path.join(common_ap_dir_path, 'config_algs', per_metric, alg, '{}_f{}_DIM{}.csv'.format(alg, fun_id, dim))                
                with open(metric_file_path, 'r') as fh:
                    metric_values[alg] = float(fh.read())                    

            if np.count_nonzero(~np.isnan(np.array(list(metric_values.values())))) == 0:
                logger.warning("AP=%s, f=%d, dim=%d: All METRIC values are NaN.", ap_name, fun_id, dim)
                best_metric_value = np.nan
            else:
                best_metric_value = np.nanmin(np.array(list(metric_values.values())))

            out_file_path = os.path.join(common_ap_dir_path, 'config_algs', per_metric, 'best_f{}_DIM{}.csv'.format(fun_id, dim))
            with open(out_file_path, 'w') as fh:
                fh.write("{}".format(best_metric_value))

            # Calculate the relative metric value
            for alg in all_algs:
                out_file_path = os.path.join(common_ap_dir_path, 'config_algs', rel_per_metric, '{}_f{}_DIM{}.csv'.format(alg, fun_id, dim))
                rel_metric_value = np.nan
                if np.isnan(best_metric_value) == False:
                    rel_metric_value = metric_values[alg] / best_metric_value
                with open(out_file_path, 'w') as fh:
                    fh.write("{}".format(rel_metric_value))

    # 4. Assign the PAR10 value to the missing relMETRIC value
    for dim in dims:
        # 4.1 Find the worst relMETRIC value for each dimension
        worst_rel_metric_value = -1
        for fun_id in all_fun_ids:
            for alg in all_algs:
                relmetric_file_path = os.path.join(common_ap_dir_path, 'config_algs', rel_per_metric, '{}_f{}_DIM{}.csv'.format(alg, fun_id, dim))
                with open(relmetric_file_path, 'r') as fh:
                    rel_metric_value = float(fh.read())
                    if np.isnan(rel_metric_value) == False and rel_metric_value > worst_rel_metric_value:
                        worst_rel_metric_value = rel_metric_value
        par10_value = 10 * worst_rel_metric_value

        # Record the PAR10 value to quickly calculate the relMETRIC value in the algorithm selection phase
        # I hope that there is no algorithm of the name "par10"
        out_file_path = os.path.join(common_ap_dir_path, 'config_algs', rel_per_metric, 'par10_DIM{}.csv'.format(dim))
        with open(out_file_path, 'w') as fh:
            fh.write("{}".format(par10_value))

        # 4.2 Replace NaN with the PAR10 value
        for fun_id in all_fun_ids:
            for alg in all_algs:
                relmetric_file_path = os.path.join(common_ap_dir_path, 'config_algs', rel_per_metric, '{}_f{}_DIM{}.csv'.format(alg, fun_id, dim))           
                with open(relmetric_file_path, 'r') as fh:
                    rel_metric_value = float(fh.read())
                if np.isnan(rel_metric_value) == True:
                    with open(relmetric_file_path, 'w') as fh:
                        fh.write("{}".format(par10_value))

def calc_relmetric_sbs(bbob_suite, all_fun_ids, dims, per_metric, target, common_ap_dir_path, ap):
    rel_per_metric = 'rel' + per_metric
    sbs_dir_path = os.path.join(common_ap_dir_path, 'sbs', 'rel_{}'.format(per_metric))
    os.makedirs(sbs_dir_path, exist_ok=True)                    
    
    for dim in dims:
        sbs_file_path = os.path.join('./alg_portfolio', ap, 'sbs_{}'.format(per_metric), 'sbs_DIM{}.csv'.format(dim))
        with open(sbs_file_path, 'r') as fh:
            sbs = fh.read()

        all_metric_values = []
        for fun_id in all_fun_ids:
            original_path = os.path.join(common_ap_dir_path, 'config_algs/rel_{}'.format(per_metric), '{}_f{}_DIM{}.csv'.format(sbs, fun_id, dim))
            copy_file_path = os.path.join(sbs_dir_path, '{}_f{}_DIM{}.csv'.format(ap, fun_id, dim))
            shutil.copyfile(original_path, copy_file_path)

            with open(original_path, 'r') as fh:
                all_metric_values.append(float(fh.read()))

        mean_metric_value = stat.mean(all_metric_values)
        out_file_path = os.path.join(sbs_dir_path, '{}_mean_DIM{}.csv'.format(ap, dim))        
        with open(out_file_path, 'w') as fh:
            fh.write(str(mean_metric_value))
            
def calc_relmetric_vbs(bbob_suite, all_fun_ids, dims, per_metric, target, common_ap_dir_path, ap):
    rel_per_metric = 'rel' + per_metric
    vbs_dir_path = os.path.join(common_ap_dir_path, 'vbs', 'rel_{}'.format(per_metric))
    os.makedirs(vbs_dir_path, exist_ok=True)                            
    
    for dim in dims:
        all_metric_values = []        
        for fun_id in all_fun_ids:        
            vbs_file_path = os.path.join('./alg_portfolio', ap, 'vbs_{}'.format(per_metric), 'f{}_DIM{}.csv'.format(fun_id, dim))
            with open(vbs_file_path, 'r') as fh:
                vbs = fh.read()

            original_path = os.path.join(common_ap_dir_path, 'config_algs/rel_{}'.format(per_metric), '{}_f{}_DIM{}.csv'.format(vbs, fun_id, dim))
            copy_file_path = os.path.join(vbs_dir_path, '{}_f{}_DIM{}.csv'.format(ap, fun_id, dim))
            shutil.copyfile(original_path, copy_file_path)

            with open(original_path, 'r') as fh:
                all_metric_values.append(float(fh.read()))

        mean_metric_value = stat.mean(all_metric_values)
        out_file_path = os.path.join(vbs_dir_path, '{}_mean_DIM{}.csv'.format(ap, dim))        
        with open(out_file_path, 'w') as fh:
            fh.write(str(mean_metric_value))
            
def print_table_aps(all_aps, ap_nickname_dict):
    for ap in all_aps:        
        ap_config_file_path = os.path.join('./alg_portfolio', ap, 'ap_config.csv')
        with open(ap_config_file_path, 'r') as fh:
            algs = fh.read()
        algs = algs.replace('DTS-CMA-ES_005-2pop_v26_1model', 'DTS-CMA-ES18')            
        algs = algs.replace('_', '\_')
        algs = algs.replace(',', ', ')
        print("{} & \\shortstack{{{}}}\\\\".format(ap_nickname_dict[ap], algs))
        print("\\midrule")                        

def print_sbs_vbs_res(all_aps, ap_nickname_dict, dims, common_ap_dir_path, per_metric):
    rel_per_metric = 'rel' + per_metric
    vbs_dir_path = os.path.join(common_ap_dir_path, 'vbs', 'rel_{}'.format(per_metric))
    os.makedirs(vbs_dir_path, exist_ok=True)                                

    for ap in all_aps:        
        print(ap_nickname_dict[ap], end=' ')

        for dim in dims:
            vbs_mean_relmetric_file_path = os.path.join(common_ap_dir_path, 'vbs', 'rel_{}'.format(per_metric), '{}_mean_DIM{}.csv'.format(ap, dim))
            with open(vbs_mean_relmetric_file_path, 'r') as fh:
                vbs_mean_relmetric = float(fh.read())
            
            print(" & {:.2f}".format(vbs_mean_relmetric), end='')

        for dim in dims:
            sbs_mean_relmetric_file_path = os.path.join(common_ap_dir_path, 'sbs', 'rel_{}'.format(per_metric), '{}_mean_DIM{}.csv'.format(ap, dim))
            with open(sbs_mean_relmetric_file_path, 'r') as fh:
                sbs_mean_relmetric = float(fh.read())
            
            print(" & {:.2f}".format(sbs_mean_relmetric), end='')

        print("\\\\")
        print("\\midrule")                

def print_as_res(all_aps, ap_nickname_dict, dims, common_ap_dir_path, per_metric):    
    print("%%%%%%%%%%%")
    print("\\subfloat[{}]{{".format(cv_type.upper().replace('_', '-')))
    print("\\begin{tabular}{lllllll}")
    print("\\toprule")

    print("AP & $n=2$ & $n=3$ & $n=5$ & $n=10$\\\\")
    print("\\midrule")                    

    for ap in all_aps:        
        as_system = '{}_{}_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none_slsqp_multiplier50'.format(ap, selector, cv_type)
        
        print(ap_nickname_dict[ap], end=' ')

        for dim in dims:
            as_median_relmetric_file_path = os.path.join(common_ap_dir_path, 'as_systems', as_system, 'rel_{}'.format(per_metric), 'median_DIM{}.csv'.format(dim))
            with open(as_median_relmetric_file_path, 'r') as fh:
                median_relmetric = float(fh.read())
            
            print(" & {:.2f}".format(median_relmetric), end='')

        print("\\\\")
        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
               
def calc_relmetric_as(bbob_suite, all_fun_ids, dims, per_metric, target, common_ap_dir_path, sids, as_system):
    rel_per_metric = 'rel' + per_metric
    as_dir_path = os.path.join(common_ap_dir_path, 'as_systems', as_system, 'rel_{}'.format(per_metric))
    os.makedirs(as_dir_path, exist_ok=True)                            
    
    for dim in dims:
        par10_file_path = os.path.join(common_ap_dir_path, 'config_algs', 'rel_{}'.format(per_metric), 'par10_DIM{}.csv'.format(dim))
        with open(par10_file_path, 'r') as fh:
            par10_value = float(fh.read())
        
        for fun_id in all_fun_ids:
            best_metric_file_path = os.path.join(common_ap_dir_path, 'config_algs', per_metric, 'best_f{}_DIM{}.csv'.format(fun_id, dim))
            with open(best_metric_file_path, 'r') as fh:
                best_metric_value = float(fh.read())            
                    
            for sid in sids:
                metric_file_path = os.path.join('./pp_as_results', as_system.replace('sid', 'sid{}'.format(sid)), per_metric, 'f{}_DIM{}_liid0.csv'.format(fun_id, dim))            
                with open(metric_file_path, 'r') as fh:
                    metric_value = float(fh.read())

                if np.isnan(metric_value) == True:
                    rel_metric_value = par10_value
                else:
                    rel_metric_value = metric_value / best_metric_value

                out_file_path = os.path.join(as_dir_path, 'f{}_DIM{}_sid{}.csv'.format(fun_id, dim, sid))
                with open(out_file_path, 'w') as fh:
                    fh.write("{}".format(rel_metric_value))                                    

    # Calculate the median of the 31 mean relative metric values
    for dim in dims:        
        all_stat_values = []
        for sid in sids:
            rel_metric_values = []
            for fun_id in all_fun_ids:
                rel_metric_file_path = os.path.join(as_dir_path, 'f{}_DIM{}_sid{}.csv'.format(fun_id, dim, sid))                
                with open(rel_metric_file_path, 'r') as fh:
                    rel_metric_values.append(float(fh.read()))
            mean_metric_value = stat.mean(rel_metric_values)
            all_stat_values.append(mean_metric_value)

        stat_value = stat.median(all_stat_values)
        out_file_path = os.path.join(as_dir_path, 'median_DIM{}.csv'.format(dim))
        with open(out_file_path, 'w') as fh:
            fh.write("{}".format(stat_value))        

def calc_aps(common_ap_dir_path, all_portfolios, selector, dim, all_fun_ids, sids, cv_type, per_metric): 
    res_dir_path = os.path.join(common_ap_dir_path, 'aps', selector+'_'+cv_type)
    os.makedirs(res_dir_path, exist_ok=True)    
    
    for ap1 in all_portfolios:
        sum_lost = 0
        
        for fun_id in all_fun_ids:
            ap1_metric_list = []
            for sid in sids:
                as_dir_path = os.path.join(common_ap_dir_path, 'as_systems', '{}_{}_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none_slsqp_multiplier50'.format(ap1, selector, cv_type), 'rel_'+per_metric)
                metric_file_path = os.path.join(as_dir_path, 'f{}_DIM{}_sid{}.csv'.format(fun_id, dim, sid))
                with open(metric_file_path, 'r') as fh:
                    ap1_metric_list.append(float(fh.read()))
            
            for ap2 in all_portfolios:                
                if ap1 == ap2:
                    # print("---")
                    # print(ap1, ap2)
                    continue
                
                ap2_metric_list = []
                for sid in sids:
                    as_dir_path = os.path.join(common_ap_dir_path, 'as_systems', '{}_{}_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none_slsqp_multiplier50'.format(ap2, selector, cv_type), 'rel_'+per_metric)
                    metric_file_path = os.path.join(as_dir_path, 'f{}_DIM{}_sid{}.csv'.format(fun_id, dim, sid))
                    with open(metric_file_path, 'r') as fh:
                        ap2_metric_list.append(float(fh.read()))
                                            
                if ap1_metric_list != ap2_metric_list:
                    stat_res = stats.mannwhitneyu(ap1_metric_list, ap2_metric_list, alternative='two-sided')
                    if stat_res.pvalue < 0.05 and stat.median(ap1_metric_list) > stat.median(ap2_metric_list):
                        sum_lost += 1

        aps = float(sum_lost) / len(all_fun_ids)
        print(ap1, aps, sum_lost)

        res_file_path = os.path.join(res_dir_path, '{}_DIM{}.csv'.format(ap1, dim))
        with open(res_file_path, 'w') as fh:
            fh.write(str(aps))

def calc_aps_mean_metric(common_ap_dir_path, all_portfolios, selector, dim, all_fun_ids, sids, cv_type, per_metric): 
    res_dir_path = os.path.join(common_ap_dir_path, 'aps_mean_sp1', selector+'_'+cv_type)
    os.makedirs(res_dir_path, exist_ok=True)    
    
    for ap1 in all_portfolios:
        sum_lost = 0

        ap1_metric_list = []        
        for sid in sids:
            fun_metric_list = []
            for fun_id in all_fun_ids:
                as_dir_path = os.path.join(common_ap_dir_path, 'as_systems', '{}_{}_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none_slsqp_multiplier50'.format(ap1, selector, cv_type), 'rel_'+per_metric)
                metric_file_path = os.path.join(as_dir_path, 'f{}_DIM{}_sid{}.csv'.format(fun_id, dim, sid))
                with open(metric_file_path, 'r') as fh:
                    fun_metric_list.append(float(fh.read()))

            mean_metric_value = stat.mean(fun_metric_list)
            ap1_metric_list.append(mean_metric_value)
            
        for ap2 in all_portfolios:                
            if ap1 == ap2:
                continue

            ap2_metric_list = []        
            for sid in sids:
                fun_metric_list = []
                for fun_id in all_fun_ids:            
                    as_dir_path = os.path.join(common_ap_dir_path, 'as_systems', '{}_{}_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none_slsqp_multiplier50'.format(ap2, selector, cv_type), 'rel_'+per_metric)
                    metric_file_path = os.path.join(as_dir_path, 'f{}_DIM{}_sid{}.csv'.format(fun_id, dim, sid))
                    with open(metric_file_path, 'r') as fh:
                        fun_metric_list.append(float(fh.read()))

                mean_metric_value = stat.mean(fun_metric_list)
                ap2_metric_list.append(mean_metric_value)
                        
            if ap1_metric_list != ap2_metric_list:
                stat_res = stats.mannwhitneyu(ap1_metric_list, ap2_metric_list, alternative='two-sided')
                if stat_res.pvalue < 0.05 and stat.median(ap1_metric_list) > stat.median(ap2_metric_list):
                    sum_lost += 1

        #aps = float(sum_lost) / len(all_fun_ids)
        aps = sum_lost
        print(selector, dim, ap1, aps, sum_lost)

        res_file_path = os.path.join(res_dir_path, '{}_DIM{}.csv'.format(ap1, dim))
        with open(res_file_path, 'w') as fh:
            fh.write(str(aps))
            
def print_aps(all_portfolios, ap_nickname_dict, common_ap_dir_path, selector, selector_nickname, cv_type):
    best_dict = {}
    second_best_dict = {}
    best_value_dict = {}
    second_best_value_dict = {}
    for dim in dims:
        aps_list = []
        for ap in all_portfolios:            
            res_dir_path = os.path.join(common_ap_dir_path, 'aps_mean_sp1', selector+'_'+cv_type)
            res_file_path = os.path.join(res_dir_path, '{}_DIM{}.csv'.format(ap, dim))
            
            with open(res_file_path, 'r') as fh:
                aps = float(fh.read())
            aps_list.append(aps)

        l = np.argsort(np.array(aps_list))
        best_dict[dim] = all_portfolios[l[0]]
        second_best_dict[dim] = all_portfolios[l[1]]        

        aps_list = list(set(aps_list))
        aps_list = np.sort(np.array(aps_list))
        best_value_dict[dim] = aps_list[0]
        second_best_value_dict[dim] = aps_list[1]
        
    print("%%%%%%%%%%%")
    print("\\subfloat[{}]{{".format(cv_type.upper().replace('_', '-')))

    print("\\begin{tabular}{lC{1em}C{1em}C{1em}C{1em}}")
    print("\\toprule")

    # print("System & $n=2$ & $n=3$ & $n=5$ & $n=10$\\\\")

    print(" & \\multicolumn{4}{c}{$n$}\\\\")
    print("\cmidrule{2-5}")
    print(" & $2$ & $3$ & $5$ & $10$\\\\")
    #\\raisebox{1em}{Portfolio}
    print("\\midrule")                    
    
    for ap in all_portfolios:
        print("{}".format(ap_nickname_dict[ap]), end='')

        for dim in dims:
            res_dir_path = os.path.join(common_ap_dir_path, 'aps_mean_sp1', selector+'_'+cv_type)
            #res_dir_path = os.path.join(common_ap_dir_path, 'aps', selector+'_'+cv_type)            
            res_file_path = os.path.join(res_dir_path, '{}_DIM{}.csv'.format(ap, dim))        
            with open(res_file_path, 'r') as fh:
                aps = float(fh.read())

            if np.isclose(aps, best_value_dict[dim]):
                #print(" & \\cellcolor{{c1}}{:.2f}".format(aps), end='')
                print(" & \\cellcolor{{c1}}{:d}".format(int(aps)), end='')                
            #elif ap == second_best_dict[dim] or np.isclose(aps, second_best_value_dict[dim]):
            elif  np.isclose(aps, second_best_value_dict[dim]):                
                #print(" & \\cellcolor{{c2}}{:.2f}".format(aps), end='')
                print(" & \\cellcolor{{c2}}{:d}".format(int(aps)), end='')                                
            else:
                #print(" & {:.2f}".format(aps), end='')
                print(" & {:d}".format(int(aps)), end='')                

        print("\\\\")
       
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
            
if __name__ == '__main__':
    target = '-2.0'    
    bbob_suite = 'bbob'
    all_fun_ids = range(1, 24+1)
    if bbob_suite == 'bbob-noisy':
        all_fun_ids = range(101, 130+1)

    sids = range(0, 31)        
    dims = [2, 3, 5, 10]
    per_metric = 'sp1'
    rel_per_metric = 'rel_' + per_metric

    common_ap_dir_path = './pp_14_ap'
    all_aps = ['kt_ecj19', 'fixed_dlvat_foga19', 'jped_gecco21', 'bmtp_gecco12', 'mk_21', 'lsap_k2_sp1_target-2.0_dims2_3_5_10', 'lsap_k4_sp1_target-2.0_dims2_3_5_10', 'lsap_k6_sp1_target-2.0_dims2_3_5_10', 'lsap_k8_sp1_target-2.0_dims2_3_5_10', 'lsap_k10_sp1_target-2.0_dims2_3_5_10', 'lsap_k12_sp1_target-2.0_dims2_3_5_10', 'lsap_k14_sp1_target-2.0_dims2_3_5_10', 'lsap_k16_sp1_target-2.0_dims2_3_5_10', 'lsap_k18_sp1_target-2.0_dims2_3_5_10']

    ap_nickname_dict = {'mk_21':'$\mathcal{A}_{\mathrm{mk}}$', 'jped_gecco21':'$\mathcal{A}_{\mathrm{jped}}$', 'dlvat_foga19':'$\mathcal{A}_{\mathrm{dlvat}}$', 'fixed_dlvat_foga19':'$\mathcal{A}_{\mathrm{dlvat}}$', 'kt_ecj19':'$\mathcal{A}_{\mathrm{kt}}$', 'bmtp_gecco12':'$\mathcal{A}_{\mathrm{bmtp}}$', 'lsap_k2_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}2}$', 'lsap_k4_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}4}$', 'lsap_k6_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}6}$', 'lsap_k8_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}8}$', 'lsap_k10_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}10}$', 'lsap_k12_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}12}$', 'lsap_k14_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}14}$', 'lsap_k16_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}16}$', 'lsap_k18_sp1_target-2.0_dims2_3_5_10':'$\mathcal{A}_{\mathrm{ls}18}$'}

    selector_nickname_dict = {'multiclass_classification':'Classification', 'hiearchical_regression':'Regression', 'pairwise_classification':'P-classification', 'pairwise_regression':'P-regression', 'clustering':'Clustering'}
    
    # # Calculate the bestSP1, the PAR10, and the relSP1 based on all algorithms in "all_aps".
    calc_relmetric(bbob_suite, all_fun_ids, dims, per_metric, target, common_ap_dir_path, all_aps)

    # # Calculate the relSP1 value of the SBS
    for ap in all_aps:
        calc_relmetric_sbs(bbob_suite, all_fun_ids, dims, per_metric, target, common_ap_dir_path, ap)
    
    # Calculate the relSP1 value of the VBS
    for ap in all_aps:
        calc_relmetric_vbs(bbob_suite, all_fun_ids, dims, per_metric, target, common_ap_dir_path, ap)
        
    # # Calculate the relSP1 value of algorithm selection systems
    sids = range(0, 31)

    for ap in all_aps:
        for cv_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:
            for selector in ['multiclass_classification', 'hiearchical_regression', 'pairwise_classification', 'pairwise_regression', 'clustering']:            
                as_system = '{}_{}_{}_ihs_multiplier50_sid_sp1_basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta_dims2_3_5_10_none_slsqp_multiplier50'.format(ap, selector, cv_type)
                calc_relmetric_as(bbob_suite, all_fun_ids, dims, per_metric, target, common_ap_dir_path, sids, as_system)
                
    for selector in ['multiclass_classification', 'hiearchical_regression', 'pairwise_classification', 'pairwise_regression', 'clustering']:
        for cv_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:
            for dim in [2, 3, 5, 10]:            
                calc_aps_mean_metric(common_ap_dir_path, all_aps, selector, dim, all_fun_ids, sids, cv_type, per_metric)

    
    # selector = 'multiclass_classification'
    # #selector = 'hiearchical_regression'    
    # #selector = 'pairwise_classification' 
    # #selector = 'pairwise_regression'
    # #selector = 'clustering'
                
    # for cv_type in ['loio_cv', 'lopo_cv', 'lopoad_cv']:
    #     print_aps(all_aps, ap_nickname_dict, common_ap_dir_path, selector, selector_nickname_dict[selector], cv_type)

