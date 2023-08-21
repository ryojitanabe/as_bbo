"""
Constructing algorithm portfolios.
"""
import numpy as np
import pandas as pd
import os
#import wget
import urllib.error
import urllib.request
import tarfile
import glob
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(filename='{}.log'.format(__file__), level=logging.DEBUG)

def construct_ap_kt(bbob_suite, dims, fun_ids, all_algs, target='-2.0', top_k=3):
    """
    Construct an algorithm portfolio presented by the following article:
    Pascal Kerschke, Heike Trautmann: Automated Algorithm Selection on Continuous Black-Box Problems by Combining Exploratory Landscape Analysis and Machine Learning. Evol. Comput. 27(1): 99-127 (2019)
    For each function and each dimension, the 213 optimizers are ranked based on their ERT values. Then, the top 3 optimizers are selected.
    """
    dims_str = '_'.join([str(d) for d in dims])
    out_file_path = './alg_portfolio/ap_by_ktmethod_k{}_target{}_dims{}'.format(top_k, target, dims_str)
    os.makedirs(out_file_path, exist_ok=True)
    out_file_path = os.path.join(out_file_path, 'ap_config.csv')

    best_optimizer_list = []
    for dim in dims:
        best_optimizer_set_each_dim = set()        
        for fun_id in fun_ids:
            res_file_path = './pp_{}_exdata/ranking_ert_target{}/f{}_DIM{}.csv'.format(bbob_suite, target, fun_id, dim)
            # ranking_df = pd.read_csv(res_file_path, names=('alg', 'ert'))        
            ranking_df = pd.read_csv(res_file_path, header=0)        
            top_k_optimizers = ranking_df.loc[0:top_k-1, 'alg'].values
            for x in top_k_optimizers:
                best_optimizer_set_each_dim.add(x)
        best_optimizer_list.append(best_optimizer_set_each_dim)

    ap = best_optimizer_list[0]
    for best_optimizer_set in best_optimizer_list:
        ap = ap & best_optimizer_set

    with open(out_file_path, 'w') as fh:
        fh.write("{}".format(','.join(ap)))

def portfolio_construction(portfolio_list):
    """
    Construct a set of portfoilios. Results are saved in ./alg_portfolio.

    Parameters
    ----------
    portfolio_list: string list
        A list of portfolios

    Returns
    ----------
    """    
    ap_dict = {
        'kt':['BrentSTEPrr', 'BrentSTEPqi', 'fmincon', 'fminunc', 'MLSL', 'HMLSL', 'MCS', 'IPOP400D', 'HCMA', 'CMA-CSA', 'SMAC-BBOB', 'OQNLP'],
        'dlvat':['BrentSTEPrr', 'fmincon', 'fminunc', 'HMLSL', 'MCS', 'IPOP400D', 'HCMA', 'CMA-CSA', 'MLSL', 'OQNLP'],
        'jped':['BrentSTEPrr', 'BrentSTEPqi', 'fmincon', 'fminunc', 'MLSL', 'HMLSL', 'MCS', 'IPOP400D', 'HCMA', 'CMA-CSA', 'BIPOP-CMA-ES', 'OQNLP'],
        'bmtp':['BFGS', 'BIPOP-CMA-ES', 'LSfminbnd', 'LSstep'],
        'mk':['1plus2mirser', 'BIPOP-CMA-ES', 'LSstep', 'NELDERDOERR'],
        'ls2':['HCMA', 'HMLSL'],
        'ls4':['HCMA', 'HMLSL', 'BrentSTEPqi', 'HE-ES'],
        'ls6':['HCMA', 'HMLSL', 'BrentSTEPqi', 'CMAES-APOP-Var1', 'DTS-CMA-ES_005-2pop_v26_1model', 'HE-ES'],
        'ls8':['BIPOP-saACM-k', 'HMLSL', 'BrentSTEPqi', 'DTS-CMA-ES', 'CMAES-APOP-Var1', 'DTS-CMA-ES_005-2pop_v26_1model', 'HE-ES', 'SLSQP-11-scipy'],
        'ls10':['MOS', 'BIPOP-saACM-k', 'HMLSL', 'SMAC-BBOB', 'BrentSTEPqi', 'DTS-CMA-ES', 'PSA-CMA-ES', 'DTS-CMA-ES_005-2pop_v26_1model', 'HE-ES', 'SLSQP-11-scipy'],
        'ls12':['LSstep', 'MOS', 'BIPOP-saACM-k', 'fmincon', 'HMLSL', 'lmm-CMA-ES', 'SMAC-BBOB', 'BrentSTEPqi', 'PSA-CMA-ES', 'DTS-CMA-ES_005-2pop_v26_1model', 'HE-ES', 'SLSQP-11-scipy'],
        'ls14':['LSstep', 'MOS', 'BIPOP-saACM-k', 'fmincon', 'HMLSL', 'OQNLP', 'SMAC-BBOB', 'BrentSTEPqi', 'DTS-CMA-ES', 'PSA-CMA-ES', 'DTS-CMA-ES_005-2pop_v26_1model', 'HE-ES', 'lq-CMA-ES', 'SLSQP-11-scipy'],
        'ls16':['LSstep', 'MCS', 'AVGNEWUOA', 'MOS', 'BIPOP-saACM-k', 'fmincon', 'MLSL', 'OQNLP', 'P-DCN', 'BrentSTEPqi', 'DTS-CMA-ES', 'PSA-CMA-ES'
 'DTS-CMA-ES_005-2pop_v26_1model', 'HE-ES', 'lq-CMA-ES', 'SLSQP-11-scipy'],
        'ls18':['LSstep', 'MCS', 'AVGNEWUOA', 'MOS', 'BIPOP-saACM-k', 'fmincon', 'lmm-CMA-ES', 'MLSL', 'OQNLP', 'P-DCN', 'BrentSTEPqi', 'BrentSTEPrr', 'DTS-CMA-ES', 'PSA-CMA-ES', 'DTS-CMA-ES_005-2pop_v26_1model', 'HE-ES', 'lq-CMA-ES', 'SLSQP-11-scipy']
    }

    for portfolio in portfolio_list:
        algs = ap_dict[portfolio]
        ap_dir_path = './alg_portfolio/{}'.format(portfolio)
        os.makedirs(ap_dir_path, exist_ok=True)
        ap_config_file_path = os.path.join(ap_dir_path, 'ap_config.csv')
        with open(ap_config_file_path, 'w') as fh:
            fh.write("{}".format(','.join(algs)))
    
def pp_alg_portfolio_data(bbob_suite, dims, fun_ids, ap_name, target='-2.0', per_metric='sp1'):
    """
    Postprocess benchmarking results of algorithms in a portfolio for the LOPO-CV and the LOPOAD-CV.

    Parameters
    ----------
    bbob_suite: string
        The name of a bbob suite (bbob, bbob-noisy, bbob-biobj, bbob-largescale, bbob-mixint, bbob-biobj-mixint)
    dims: integer list
        A list of dimensions [2, 3, 5, 10]
    fun_ids: integer list
        A list of function IDs
    ap_name: string
        The name of an algorithm portfolio
    target: string
        A target value (10^target)
    per_metric: string
        A performance metric (ert or sp1)

    Returns
    ----------
    """    
    ap_dir_path = os.path.join('./alg_portfolio', ap_name)
    ap_config_file_path = os.path.join(ap_dir_path, 'ap_config.csv')
    ap_conifg_list = np.loadtxt(ap_config_file_path, delimiter=",", comments="#", dtype=str)

    rel_per_metric = 'rel' + per_metric
    
    # 1. Copy all the files that contain the number of function evaluations to reach the target value
    copy_dir_path = os.path.join(ap_dir_path, 'fevals_to_reach')
    os.makedirs(copy_dir_path, exist_ok=True)    
    for alg in ap_conifg_list:
        for dim in dims:
            for fun_id in fun_ids:                
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

    # 2. Copy the files for a given metric to reach the target value                            
    copy_dir_path = os.path.join(ap_dir_path, per_metric)
    os.makedirs(copy_dir_path, exist_ok=True)
    for alg in ap_conifg_list:
        for dim in dims:
            for fun_id in fun_ids:                
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
    
    # 3. Calculate the relMETRIC value and record the best METRIC value 
    out_dir_path = os.path.join(ap_dir_path, rel_per_metric)
    os.makedirs(out_dir_path, exist_ok=True)    
    for dim in dims:
        for fun_id in fun_ids:
            #logger.debug("-------------- dim=%d, f=%d -----------",dim, fun_id)
            alg_metric_values = {}
            for alg in ap_conifg_list:
                metric_file_path = os.path.join(ap_dir_path, per_metric, '{}_f{}_DIM{}.csv'.format(alg, fun_id, dim))
                with open(metric_file_path, 'r') as fh:
                    metric_value = fh.read()
                alg_metric_values[alg] = float(metric_value)

            # Find the minimum METRIC value from the dictionary alg_metric_values, where the NaN value is excluded
            if np.count_nonzero(~np.isnan(np.array(list(alg_metric_values.values())))) == 0:
                logger.warning("AP=%s, f=%d, dim=%d: All METRIC values are NaN.", ap_name, fun_id, dim)
                best_metric_value = np.nan
            else:
                best_metric_value = np.nanmin(np.array(list(alg_metric_values.values())))
            
            # Record the best METRIC value to quickly calculate the relMETRIC value in the algorithm selection phase
            # I hope that there is no algorithm whose name is "best"
            out_file_path = os.path.join(ap_dir_path, per_metric, 'best_f{}_DIM{}.csv'.format(fun_id, dim))
            with open(out_file_path, 'w') as fh:
                fh.write("{}".format(best_metric_value))
            
            for alg in ap_conifg_list:
                out_file_path = os.path.join(ap_dir_path, rel_per_metric, '{}_f{}_DIM{}.csv'.format(alg, fun_id, dim))
                rel_metric_value = np.nan
                if np.isnan(best_metric_value) == False:
                    rel_metric_value = alg_metric_values[alg] / best_metric_value
                with open(out_file_path, 'w') as fh:
                    fh.write("{}".format(rel_metric_value))

    # 4. Assign the PAR10 value to the missing relMETRIC value
    out_dir_path = os.path.join(ap_dir_path, rel_per_metric)
    os.makedirs(out_dir_path, exist_ok=True)
    for dim in dims:
        # 4.1 Find the worst relMETRIC value for each dimension
        worst_rel_metric_value = -1
        for fun_id in fun_ids:
            for alg in ap_conifg_list:
                relmetric_file_path = os.path.join(ap_dir_path, rel_per_metric, '{}_f{}_DIM{}.csv'.format(alg, fun_id, dim))
                with open(relmetric_file_path, 'r') as fh:
                    rel_metric_value = float(fh.read())
                    if np.isnan(rel_metric_value) == False and rel_metric_value > worst_rel_metric_value:
                        worst_rel_metric_value = rel_metric_value
        par10_value = 10 * worst_rel_metric_value

        # Record the PAR10 value to quickly calculate the relMETRIC value in the algorithm selection phase
        # I hope that there is no algorithm of the name "par10"
        out_file_path = os.path.join(ap_dir_path, rel_per_metric, 'par10_DIM{}.csv'.format(dim))
        with open(out_file_path, 'w') as fh:
            fh.write("{}".format(par10_value))

        # 4.2 Replace NaN with the PAR10 value
        for fun_id in fun_ids:
            for alg in ap_conifg_list:
                relmetric_file_path = os.path.join(ap_dir_path, rel_per_metric, '{}_f{}_DIM{}.csv'.format(alg, fun_id, dim))
                rel_metric_value = -1                
                with open(relmetric_file_path, 'r') as fh:
                    rel_metric_value = float(fh.read())
                if np.isnan(rel_metric_value) == True:
                    with open(relmetric_file_path, 'w') as fh:
                        fh.write("{}".format(par10_value))
            
    # 5. Select the single-best solver (SBS) from a set of algorithms based on their relMETRIC with PAR10 values
    out_dir_path = os.path.join(ap_dir_path, 'sbs_'+per_metric)
    os.makedirs(out_dir_path, exist_ok=True)
    for dim in dims:        
        alg_mean_relmetric = {}
        for alg in ap_conifg_list:
            sum_relmetric = 0
            for fun_id in fun_ids:
                relmetric_file_path = os.path.join(ap_dir_path, rel_per_metric, '{}_f{}_DIM{}.csv'.format(alg, fun_id, dim))
                with open(relmetric_file_path, 'r') as fh:
                    rel_metric_value = float(fh.read())
                    sum_relmetric += rel_metric_value                    
            alg_mean_relmetric[alg] = sum_relmetric / len(fun_ids)
        
        sbs = min(alg_mean_relmetric, key=alg_mean_relmetric.get)
        sbs_file_path = os.path.join(ap_dir_path, 'sbs_'+per_metric, 'sbs_DIM{}.csv'.format(dim))
        with open(sbs_file_path, 'w') as fh:
            fh.write("{}".format(sbs))

def instance_ids_lo(n_iids=5, liid=0):
    iids = []
    for i in range(1, n_iids+1):
        if i != liid:
            iids.append(i)
    return iids
            
def pp_alg_portfolio_data_loio_cv(bbob_suite, dims, fun_ids, ap_name, target='-2.0', per_metric='ert'):
    """
    Postprocess benchmarking results of algorithms in a portfolio for the LOIO-CV.

    Parameters
    ----------
    bbob_suite: string
        The name of a bbob suite (bbob, bbob-noisy, bbob-biobj, bbob-largescale, bbob-mixint, bbob-biobj-mixint)
    dims: integer list
        A list of dimensions [2, 3, 5, 10]
    fun_ids: integer list
        A list of function IDs
    ap_name: string
        The name of an algorithm portfolio
    target: string
        A target value (10^target)
    per_metric: string
        A performance metric (ert or sp1)

    Returns
    ----------
    """    
    ap_dir_path = os.path.join('./alg_portfolio', ap_name)
    ap_config_file_path = os.path.join(ap_dir_path, 'ap_config.csv')
    ap_conifg_list = np.loadtxt(ap_config_file_path, delimiter=",", comments="#", dtype=str)

    rel_per_metric = 'rel' + per_metric
    
    # 1. Copy all the files that contain the number of function evaluations to reach the target value
    copy_dir_path = os.path.join(ap_dir_path, 'fevals_to_reach')
    os.makedirs(copy_dir_path, exist_ok=True)    
    for alg in ap_conifg_list:
        for dim in dims:
            for fun_id in fun_ids:                
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

    # 2. Copy the files for the calculation of a given metric (ERT or SP1) to reach the target value
    copy_dir_path = os.path.join(ap_dir_path, per_metric)
    os.makedirs(copy_dir_path, exist_ok=True)
    for alg in ap_conifg_list:
        for dim in dims:
            for fun_id in fun_ids:                
                for left_instance_id in range(0, 5+1):                
                    all_instance_ids = instance_ids_lo(5, left_instance_id)                
                    original_file_path = './pp_{}_exdata/{}/{}/f{}_DIM{}_liid{}.csv'.format(bbob_suite, per_metric, alg, fun_id, dim, left_instance_id)
                    with open(original_file_path, 'r') as fh:
                        for str_line in fh:
                            tmp_target, metric_value = str_line.split(',')                        
                            if tmp_target == target:
                                metric_value = float(metric_value.replace('\n',''))
                                metric_copy_file_path = os.path.join(copy_dir_path, '{}_f{}_DIM{}_liid{}.csv'.format(alg, fun_id, dim, left_instance_id))
                                with open(metric_copy_file_path, 'w') as fh:
                                    fh.write("{}".format(metric_value))
                                break
    
    # 3. Calculate the relative metric value and record the best metric value 
    out_dir_path = os.path.join(ap_dir_path, rel_per_metric)
    os.makedirs(out_dir_path, exist_ok=True)    
    for dim in dims:
        for fun_id in fun_ids:
            for left_instance_id in range(0, 5+1):                
                all_instance_ids = instance_ids_lo(5, left_instance_id)                            
                alg_metric_values = {}
                for alg in ap_conifg_list:
                    metric_file_path = os.path.join(ap_dir_path, per_metric, '{}_f{}_DIM{}_liid{}.csv'.format(alg, fun_id, dim, left_instance_id))
                    with open(metric_file_path, 'r') as fh:
                        metric_value = fh.read()
                    alg_metric_values[alg] = float(metric_value)

                # Find the minimum metric value from the dictionary alg_metric_values, where the NaN value is excluded
                metric_array = np.array(list(alg_metric_values.values()))
                if np.all(np.isnan(metric_array)) == False:                
                    best_metric_value = np.nanmin(metric_array)
                else:
                    best_metric_value = np.nan
                    
                # Record the best metric value to quickly calculate the relative metric value in the algorithm selection phase
                # I hope that there is no algorithm whose name is "best"
                out_file_path = os.path.join(ap_dir_path, per_metric, 'best_f{}_DIM{}_liid{}.csv'.format(fun_id, dim, left_instance_id))
                with open(out_file_path, 'w') as fh:
                    fh.write("{}".format(best_metric_value))

                for alg in ap_conifg_list:
                    out_file_path = os.path.join(ap_dir_path, rel_per_metric, '{}_f{}_DIM{}_liid{}.csv'.format(alg, fun_id, dim, left_instance_id))
                    rel_metric_value = np.nan
                    if np.isnan(best_metric_value) == False:
                        rel_metric_value = alg_metric_values[alg] / best_metric_value
                    with open(out_file_path, 'w') as fh:
                        fh.write("{}".format(rel_metric_value))
            
    # 4. Assign the PAR10 value to the missing relERT value
    # It is not obvious how to assign the PAR10 value to the missing relERT value for LOIO-CV and LOPO-AD-CV. In this work, I calculated the PAR10 value based on results on all 5 instances for each dimension. Then, I assigned the SAME PAR10 value to the missing relERT value for LOPO-CV, LOIO-CV, and LOPO-AD-CV.
    out_dir_path = os.path.join(ap_dir_path, rel_per_metric)
    os.makedirs(out_dir_path, exist_ok=True)
    
    for dim in dims:
        # for left_instance_id in range(0, 5+1):                
        #     all_instance_ids = instance_ids_lo(5, left_instance_id)
        
        # 4.1 Find the worst relERT value for each dimension
        worst_rel_metric_value = -1
        for fun_id in fun_ids:
            for alg in ap_conifg_list:
                rel_metric_file_path = os.path.join(ap_dir_path, rel_per_metric, '{}_f{}_DIM{}_liid0.csv'.format(alg, fun_id, dim))
                with open(rel_metric_file_path, 'r') as fh:
                    rel_metric_value = float(fh.read())
                    if np.isnan(rel_metric_value) == False and rel_metric_value > worst_rel_metric_value:
                        worst_rel_metric_value = rel_metric_value
        par10_value = 10 * worst_rel_metric_value

        # Record the PAR10 value to quickly calculate the relERT value in the algorithm selection phase
        # I hope that there is no algorithm of the name "par10"
        out_file_path = os.path.join(ap_dir_path, rel_per_metric, 'par10_DIM{}_liid0.csv'.format(dim))
        with open(out_file_path, 'w') as fh:
            fh.write("{}".format(par10_value))

        # 4.2 Replace NaN with the PAR10 value
        for fun_id in fun_ids:
            for left_instance_id in range(0, 5+1):                
                all_instance_ids = instance_ids_lo(5, left_instance_id)                
                for alg in ap_conifg_list:
                    rel_metric_file_path = os.path.join(ap_dir_path, rel_per_metric, '{}_f{}_DIM{}_liid{}.csv'.format(alg, fun_id, dim, left_instance_id))
                    rel_metric_value = -1                
                    with open(rel_metric_file_path, 'r') as fh:
                        rel_metric_value = float(fh.read())
                    if np.isnan(rel_metric_value) == True:
                        with open(rel_metric_file_path, 'w') as fh:
                            fh.write("{}".format(par10_value))
                        
    # 5. Select the single-best solver (SBS) from a set of algorithms based on their relERT with PAR10 values
    out_dir_path = os.path.join(ap_dir_path, 'sbs_' + per_metric)
    os.makedirs(out_dir_path, exist_ok=True)

    for dim in dims:        
        alg_mean_rel_metric = {}
        for alg in ap_conifg_list:
            sum_metric = 0
            for fun_id in fun_ids:
                rel_metric_file_path = os.path.join(ap_dir_path, rel_per_metric, '{}_f{}_DIM{}_liid0.csv'.format(alg, fun_id, dim))
                with open(rel_metric_file_path, 'r') as fh:
                    rel_metric_value = float(fh.read())
                    sum_metric += rel_metric_value                    
            alg_mean_rel_metric[alg] = sum_metric / len(fun_ids)
            
        sbs = min(alg_mean_rel_metric, key=alg_mean_rel_metric.get)
        sbs_file_path = os.path.join(ap_dir_path, 'sbs_'+per_metric, 'sbs_DIM{}.csv'.format(dim))
        with open(sbs_file_path, 'w') as fh:
            fh.write("{}".format(sbs))
            
def print_ap(ap_name):
    ap_dir_path = os.path.join('./alg_portfolio', ap_name)
    ap_config_file_path = os.path.join(ap_dir_path, 'ap_config.csv')
    ap_conifg_list = np.loadtxt(ap_config_file_path, delimiter=",", comments="#", dtype=str)
    
    for alg_name in ap_conifg_list:
        alg_name = alg_name.replace('_', '\\_')
        print('\\texttt{{{}}}'.format(alg_name), end=',')


def ecdf_vbs(bbob_suite='bbob', vbs_dir_path='./', all_fun_ids=range(1,24+1), test_instance_ids=[1, 2, 3, 4, 5], dim=2, target='-2', ecdf_file_path='./'):
    target_pows = np.arange(2, -2.1, -0.2)
    target_pows = np.round(target_pows, 2)
    
    solved_counts = np.zeros(dim * 10**6+1)
    for fun_id in all_fun_ids:
        vbs_file_path = os.path.join(vbs_dir_path, 'f{}_DIM{}.csv'.format(fun_id, dim))
        with open(vbs_file_path, 'r') as fh:
            vbs_alg = fh.read()
                
        for instance_id in test_instance_ids:                       
            fevals_file_path = os.path.join('pp_bbob_exdata/fevals_to_reach', vbs_alg, 'f{}_DIM{}_i{}.csv'.format(fun_id, dim, instance_id))
            res_arr = np.loadtxt(fevals_file_path, delimiter=",", comments="#", dtype=float)
            for target_, feval, success in res_arr[::-1]:
                if target_ < -2:
                    break
                if success == 1:
                    solved_counts[int(feval):] += 1                                    
    solved_counts /= len(all_fun_ids) * len(test_instance_ids) * len(target_pows)
    
    with open(ecdf_file_path, 'w') as fh:
        for feval, count in enumerate(solved_counts):
            fh.write("{},{}\n".format(feval, count))

def vbs_construction(ap_dir_path, dim, fun_id, vbs_file_path, per_metric):
    ap_config_file_path = os.path.join(ap_dir_path, 'ap_config.csv')
    ap_algs = np.loadtxt(ap_config_file_path, delimiter=",", comments="#", dtype=str)    
    best_alg = ''
    best_score = 1e+30

    for alg in ap_algs:
        rel_metric_file_path = os.path.join(ap_dir_path, 'rel' + per_metric, '{}_f{}_DIM{}_liid{}.csv'.format(alg, fun_id, dim, 0))
        with open(rel_metric_file_path, 'r') as fh:
            score = float(fh.read())
        if score < best_score:
            best_score = score
            best_alg = alg

    with open(vbs_file_path, 'w') as fh:
        fh.write(best_alg)    
            
def run():
    # 1. Extract the algorithm name and the url link to data
    dims = [2, 3, 5, 10]
    bbob_suite = 'bbob'
    all_fun_ids = range(1, 24+1)
    if bbob_suite == 'bbob-noisy':
        all_fun_ids = range(101, 130+1)    

    alg_url = np.loadtxt('./{}_exdata/alg_url.csv'.format(bbob_suite), delimiter=",", comments="#", dtype=str)
    
    # Construct an algorithm portfolio by the Kerschke and Trautmann's systematic method
    # construct_ap_kt(bbob_suite, dims, all_fun_ids, all_algs=alg_url[:, 0], target='-2.0', top_k=3)

    # Construct the 14 algorithm portfolio used in the paper
    portfolio_list = ['kt', 'dlvat', 'jped', 'bmtp', 'mk', 'ls2', 'ls4', 'ls6', 'ls8', 'ls10', 'ls12', 'ls14', 'ls16', 'ls18']
    portfolio_construction(portfolio_list)
    
    # Postprocess performance data of each optimizer in an algorithm portfolio, including the calculation of relERT and relSP1 with PAR10.
    for ap_name in portfolio_list:
        for per_metric in ['ert', 'sp1']:
            # # # for LOIO-CV
            pp_alg_portfolio_data_loio_cv(bbob_suite, dims, all_fun_ids, ap_name, target='-2.0', per_metric=per_metric)
            # # # for LOPO-CV and LOPOAD-CV
            pp_alg_portfolio_data(bbob_suite, dims, all_fun_ids, ap_name, target='-2.0', per_metric=per_metric)
            
            # # Construct the VBS
            ap_dir_path = os.path.join('alg_portfolio', ap_name)
            vbs_dir_path = os.path.join(ap_dir_path, 'vbs_'+per_metric)
            os.makedirs(vbs_dir_path, exist_ok=True)
            for dim in dims:
                for fun_id in all_fun_ids:
                    vbs_file_path = os.path.join(vbs_dir_path, 'f{}_DIM{}.csv'.format(fun_id, dim))
                    vbs_construction(ap_dir_path, dim, fun_id, vbs_file_path, per_metric)            

            ecdf_dir_path = os.path.join(ap_dir_path, 'vbs_'+per_metric+'_ecdf')
            os.makedirs(ecdf_dir_path, exist_ok=True)        
            for dim in dims:
                ecdf_file_path = os.path.join(ecdf_dir_path, 'DIM{}.csv'.format(dim))            
                ecdf_vbs(bbob_suite=bbob_suite, vbs_dir_path=vbs_dir_path, all_fun_ids=all_fun_ids, dim=dim, ecdf_file_path=ecdf_file_path)
                                
def init_dict_rankings(bbob_suite, dims, fun_ids, all_algs, target='-2.0', per_metric='ert'):    
    dict_rankings = {}
    for alg in all_algs:
        dict_rankings[alg] = {}
        for dim in dims:
            dict_rankings[alg][dim] = {}
            for fun_id in fun_ids:
                dict_rankings[alg][dim][fun_id] = {}                        
    
    for dim in dims:
        for fun_id in fun_ids:
            res_file_path = './pp_{}_exdata/ranking_{}_target{}/f{}_DIM{}.csv'.format(bbob_suite, per_metric,target, fun_id, dim)
            #print(res_file_path)
            ranking_df = pd.read_csv(res_file_path, header=0)
            sorted_all_algs = ranking_df['alg'].values
            ert_rankings = ranking_df.rank(method='min', na_option='keep')['metric'].values

            for alg, r in zip(sorted_all_algs, ert_rankings):
                dict_rankings[alg][dim][fun_id] = r

    # alg = 'HCMA'
    # for dim in dims:
    #     print(dim)
    #     for fun_id in fun_ids:
    #         print(dict_rankings[alg][dim][fun_id], end=" ")
    #     print("")
                                
    return dict_rankings

def eval_portfolio(all_algs, dict_rankings, dims, fun_ids, x, k):
    fx = 0
    coff = 1.0 / (len(all_algs) * len(dims) * len(fun_ids))    
    sel_ids = np.where(x == 1)[0]
    ranks = np.zeros(k)
    for dim in dims:
        for fun_id in fun_ids:
            ranks = np.zeros(k)
            for i, alg in enumerate(all_algs[sel_ids]):
                ranks[i] = dict_rankings[alg][dim][fun_id]

            # When all optimizers in the portfolio failed the search
            if np.isnan(ranks).all():
                fx += 1
                #fx += len(all_algs)* len(dims) * len(fun_ids)           
            # When at least one optimizer in the portfolio succeeded the search
            else:
                fx += coff * np.nanmin(ranks)
                #fx += np.nanmin(ranks)
                
    return fx

def construct_ap_ls(bbob_suite, dims, fun_ids, all_algs, target='-2.0', run_id=0, k=3,  per_metric='ert'):
    """
    Construct an algorithm portfolio presented by the following article:
    """
    np.random.seed(seed=run_id)
    
    dims_str = '_'.join([str(d) for d in dims])    
    out_dir_path = './results_ls_ap/k{}_{}_target{}_dims{}'.format(k, per_metric, target, dims_str)
    os.makedirs(out_dir_path, exist_ok=True)
    ls_file_path = os.path.join(out_dir_path, '{}th_run.csv'.format(run_id))

    # Initialize the performance rankings for each function and for each dimension.
    dict_rankings = init_dict_rankings(bbob_suite, dims, fun_ids, all_algs, target, per_metric)

    # Initialize a search point x, where x is a binary vector, and x[i] = 1 means that the i-th algorithm is selected for a portfolio
    x = np.zeros(len(all_algs), dtype=int)
    sel_ids = np.random.choice(len(all_algs), k, replace=False)
    np.put(x, sel_ids, 1)

    # Evaluate the initial x
    evals = 0
    fx = eval_portfolio(all_algs, dict_rankings, dims, fun_ids, x, k)
    evals += 1       
    # Log the best-so-far x
    with open(ls_file_path, 'w') as fh:
        tmp = np.where(x == 1)[0]    
        ap_str = ','.join([str(a) for a in all_algs[tmp]])            
        fh.write('{},{},{}\n'.format(evals, fx, ap_str))       
    
    rnd_ids = np.arange(len(all_algs))
    imp_x_exist = True
    
    while imp_x_exist:
        imp_x_exist = False
        rnd_ids = np.random.permutation(rnd_ids)
        for i, alg in zip(rnd_ids, all_algs):
            if x[i] == 0 and i not in sel_ids:
                for j, sel_id in enumerate(sel_ids):
                    x[i] = 1
                    x[sel_id] = 0
                    f_cand_x = eval_portfolio(all_algs, dict_rankings, dims, fun_ids, x, k)
                    evals += 1
                    if f_cand_x < fx:
                        fx = f_cand_x
                        sel_ids[j] = i
                        imp_x_exist = True

                        # Log the best-so-far x
                        with open(ls_file_path, 'a') as fh:
                            tmp = np.where(x == 1)[0]    
                            ap_str = ','.join([str(a) for a in all_algs[tmp]])            
                            fh.write('{},{},{}\n'.format(evals, fx, ap_str))               
                        
                        break
                    else:
                        x[i] = 0
                        x[sel_id] = 1

def run_ls():
    bbob_suite = 'bbob'
    target = '-2.0'
    dims = [2, 3, 5, 10]

    per_metric = 'sp1'
    
    # 1. Extract the algorithm name and the url link to data
    all_fun_ids = range(1, 24+1)
    if bbob_suite == 'bbob-noisy':
        all_fun_ids = range(101, 130+1)    

    alg_url = np.loadtxt('./{}_exdata/alg_url.csv'.format(bbob_suite), delimiter=",", comments="#", dtype=str)    

    # 2. Run the local search method
    for k in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
        for run_id in range(0, 31):
            construct_ap_ls(bbob_suite, dims, all_fun_ids, all_algs=alg_url[:, 0], target=target, run_id=run_id, k=k, per_metric=per_metric)

    # 3. Save the best portfolio configuration
    for k in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
        dims_str = '_'.join([str(d) for d in dims])    
        ls_dir_path = './results_ls_ap/k{}_{}_target{}_dims{}'.format(k, per_metric, target, dims_str)        

        # Find the best portfolio configuration over the 31 runs    
        bsf_fx = 1e+20
        bsf_x = None
        for run_id in range(0, 31):
            ls_file_path = os.path.join(ls_dir_path, '{}th_run.csv'.format(run_id))            
            df = pd.read_csv(ls_file_path, header=None)
            # return df.tail(n).values.tolist()
            fx = df.iloc[-1][1]
            x = df.iloc[-1][2:k+2].values
            if fx < bsf_fx:
                bsf_fx = fx
                bsf_x = x

        # Save the best portfolio configuration
        dims_str = '_'.join([str(d) for d in dims])
        ap_dir_path = './alg_portfolio/lsap_k{}_{}_target{}_dims{}'.format(k, per_metric, target, dims_str)
        os.makedirs(ap_dir_path, exist_ok=True)
        out_file_path = os.path.join(ap_dir_path, 'ap_config.csv')        
        with open(out_file_path, 'w') as fh:
            fh.write("{}".format(','.join(bsf_x)))
        out_file_path = os.path.join(ap_dir_path, 'ap_fx_evals.csv')
        with open(out_file_path, 'w') as fh:
            fh.write("{}".format(bsf_fx))

        # Postprocess performance data of each optimizer in an algorithm portfolio, including the calculation of relERT with PAR10.
        ap_name = 'lsap_k{}_{}_target{}_dims{}'.format(k, per_metric, target, dims_str)

        for per_metric in ['ert', 'sp1']:
            # for LOIO-CV
            pp_alg_portfolio_data_loio_cv(bbob_suite, dims, all_fun_ids, ap_name, target, per_metric=per_metric)
            # for LOPO-CV and LOPOAD-CV
            pp_alg_portfolio_data(bbob_suite, dims, all_fun_ids, ap_name, target, per_metric=per_metric)

            # Construct the VBS
            ap_dir_path = os.path.join('alg_portfolio', ap_name)
            vbs_dir_path = os.path.join(ap_dir_path, 'vbs_'+per_metric)
            os.makedirs(vbs_dir_path, exist_ok=True)
            for dim in dims:
                for fun_id in all_fun_ids:
                    vbs_file_path = os.path.join(vbs_dir_path, 'f{}_DIM{}.csv'.format(fun_id, dim))
                    vbs_construction(ap_dir_path, dim, fun_id, vbs_file_path, per_metric)

            # Calculate the ECDF of the VBS
            ecdf_dir_path = os.path.join(ap_dir_path, 'vbs_'+per_metric+'_ecdf')        
            os.makedirs(ecdf_dir_path, exist_ok=True)        
            for dim in dims:
                ecdf_file_path = os.path.join(ecdf_dir_path, 'DIM{}.csv'.format(dim))            
                ecdf_vbs(bbob_suite=bbob_suite, vbs_dir_path=vbs_dir_path, all_fun_ids=all_fun_ids, dim=dim, ecdf_file_path=ecdf_file_path)
                                
if __name__ == '__main__':
    run()
    #run_ls()
