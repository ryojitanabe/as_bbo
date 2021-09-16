"""
Post-processing benchmarking results downloaded from the COCO archive (https://numbbo.github.io/data-archive/).
"""
import numpy as np
import pandas as pd
import logging
import os
import urllib.error
import urllib.request
import tarfile
import glob

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def make_alg_url_file(bbob_suite, alg_url_file_path, bbob_md_file_path='./bbob.md', incomplete_data_sets=['JADEb', 'EvoSpace-PSO-GA', 'Ord-N-DTS-CMA-ES']):
    """
    Make a CSV file that save a list of algorithms and a list of URL links to their experimental data based on a markdown file, e.g., 'bbob.md' from https://raw.githubusercontent.com/numbbo/data-archive/master/bbob.md.
    Parameters
    ----------
    bbob_suite: string
        The name of a bbob suite (bbob, bbob-noisy, bbob-biobj, bbob-largescale, bbob-mixint, bbob-biobj-mixint)
    alg_url_file_path: path
        A path to save the CSV file.
    bbob_md_file_path: path
        A markdown file path.
    incomplete_data_sets: string list
        A list of algorithms whose data are incomplete.

    Returns
    ----------
    """        
    fh_csv = open(alg_url_file_path, 'w')

    with open(bbob_md_file_path, 'r') as fh:
        for str_line in fh:
            if 'Inofficial' in str_line:
                break
            if '[data]' in str_line:
                res = str_line.split('|')
                alg_name = res[2]
                alg_name = alg_name.replace(' ', '')
                alg_name = alg_name.replace(']]', '')
                url = res[5].split('}}')[1]
                url = url.replace(')', '')
                url = url.replace(' ', '')

                # The two data sets (the 2015 and 2013 BBOB settings) are available for these five optimizers
                if alg_name == 'GP1-CMAES' or alg_name == 'GP5-CMAES' or alg_name == 'IPOPCMAv3p61' or alg_name == 'RF1-CMAES' or alg_name == 'RF5-CMAES':
                    url = url.replace('-[2013instances]({{page.dataDir', '')                                                        
                url =  'http://coco.gforge.inria.fr/data-archive/{}/'.format(bbob_suite) + url

                if alg_name not in incomplete_data_sets: 
                    fh_csv.write(alg_name + "," + url + "\n")
    fh_csv.close()

def download_file(url, dst_path):
    """
    Download a file. This function was copied from https://note.nkmk.me/python-download-web-images/ .

    Parameters
    ----------
    url: string    
        A download URL to the benchmarking data of the i-th algorithm.
    dst_path: path
        A path to save the downloaded data

    Returns
    ----------
    """    
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        logger.error(e)
        
def download_data(alg_url, bbob_suite):
    """
    Download experimental data from the COCO data archive.

    Parameters
    ----------
    alg_url: 2d-ndarray
        alg_url[i][0] is the name of the i-th algorithm. alg_url[i][1] is the URL to the benchmarking data of the i-th algorithm.
    bbob_suite: string
        The name of a bbob suite (bbob, bbob-noisy, bbob-biobj, bbob-largescale, bbob-mixint, bbob-biobj-mixint)

    Returns
    ----------
    """    
    i = 0
    for alg, url in alg_url:
        dest_path = os.path.join('./{}_exdata'.format(bbob_suite), os.path.basename(url))
        if os.path.exists(dest_path):
            logger.warning("The directory %s already exists, so %s was not downloaded", dest_path, os.path.basename(url))
        else:
            download_file(url, dest_path)                
        i += 1
        logger.info("Data for %s have been downloaded (%d)", alg, i)        

def unpack_data(exdata_dir_path, alg_url):
    """
    Unpack tar files.

    Parameters
    ----------
    exdata_dir_path: path
        A experimental data directory file path.
    alg_url: 2d-ndarray
        alg_url[i][0] is the name of the i-th algorithm. alg_url[i][1] is the URL to the benchmarking data of the i-th algorithm.

    Returns
    ----------
    """                
    i = 0
    for alg, url in alg_url:                
        tar_file_path = url.split('/')[-1]        
        tar_file_path = os.path.join(exdata_dir_path, tar_file_path)
        renamed_dir_path = os.path.join(exdata_dir_path, alg)
        
        if os.path.exists(renamed_dir_path):
            logger.warning("The directory %s already exists, so the corresponding tar file was not unpacked", renamed_dir_path)            
        else:
            with tarfile.open(tar_file_path, 'r:*') as tar:
                # The name of an unpacked directory does not always represent its algorithm name. For example, "MOS_torre_noiseless.tar.gz" includes a directory "BBOB2010rawdata", not "MOS". For the sake of simplicity, the name of each directory is renamed to its algorithm name.
                unpacked_dir_path = os.path.join(exdata_dir_path, tar.getnames()[0].split('/')[0])
                tar.extractall(exdata_dir_path)
                os.rename(unpacked_dir_path, renamed_dir_path)
        i += 1
        logger.info("Data for {} have been unpacked ({})".format(alg, i))

def read_bbobexp_file(res_file_path):
    """
    Read and save experimental data of a given file.

    Parameters
    ----------
    res_file_path: path
        A file path that provides benchmarking results on possibly multiple instances

    Returns
    ----------
    all_run_data: 2d-list
        A 2d-list. all_run_data[i] provides experimental data for the i-th position (not instance) in a given file.
    """                
    all_run_data = []
    with open(res_file_path) as fh:
        run_data = []
        for str_line in fh:
            if '%' in str_line:
                run_data = np.array(run_data)
                all_run_data.append(run_data)
                run_data = []
            else:
                str_line = str_line.replace('\t', '') 
                str_line = str_line.split(' ')                        

                # This procedure is for the other BBOB function sets with no constraint.
                run_data.append([float(str_line[0]), float(str_line[2])])
                
                # The following procedure is only for the BBOB-constrained function set. This uses the sum of the number of objective function evaluations and the number of constraint function evaluations. But, is this manner correct?
                #run_data.append([float(str_line[0])+float(str_line[1]), float(str_line[2])])

        # add the last "run_data"
        run_data = np.array(run_data)
        all_run_data.append(run_data)

    # delete the first empty element
    del all_run_data[0]
    
    return all_run_data
        
def pp_bbob_data_file(res_file_path, data_pos, out_file_path, targets, target_pows, max_fevals):
    """
    Calculate the number of function evaluations (fevals) to reach target values　10^2, ..., 10^-8.

    Parameters
    ----------
    res_file_path: path
        A file path that provides benchmarking results on possibly multiple instances
    data_pos: int
        A position of data in res_file_path to be read    
    out_file_path: path
        A file path to save postprocessing results.
    targets: integer list 
        A list of targets [10^{2}, 10^{1.8}, ..., 10^{-7.8}, 10^{-8}]
    targets: string list 
        A list of superior numbers ['2', '1.8', ..., '-7.8', '-8']
    max_fevals: int
        The maximum number of function evaluations.

    Returns
    -------    

    """                
    all_run_data = read_bbobexp_file(res_file_path)

    if len(all_run_data) == 1:    
        run_data = all_run_data[0]
    else:
        run_data = all_run_data[data_pos]
        
    fevals = run_data[:, 0].astype(np.int)       
    # TODO: This may have to be changed for bbob-constrained
    error_vals = run_data[:, 1]

    target_id = 0
    #fevals_to_reach = np.full(len(targets), np.nan)
    fevals_to_reach = np.full(len(targets), max_fevals)
    successful_runs = np.zeros(len(targets))
    
    for feval, error in zip(fevals, error_vals):                
        while error < targets[target_id]:
            fevals_to_reach[target_id] = feval
            successful_runs[target_id] = 1
            target_id += 1
            if target_id >= len(targets):
                break
        # TODO: Is there a better way?
        if target_id >= len(targets):
            break

    with open(out_file_path, 'w') as fh:
        #fh.write("#target,the number of function evaluations to reach the target,0:the run was unsuccessful and 1:the run was successful\n")        
        for target, feval, is_success in zip(target_pows[::-1], fevals_to_reach[::-1], successful_runs[::-1]):
                fh.write("{},{},{}\n".format(target, feval, is_success))                                                    
        
def pp_fevals_to_reach(bbob_suite, dim, fun_id, alg, fevals_dir_path):
    """
    Calculate the number of function evaluations (fevals) to reach target values　10^2, ..., 10^-8.

    Parameters
    ----------
    bbob_suite: string
        The name of a bbob suite (bbob, bbob-noisy, bbob-biobj, bbob-largescale, bbob-mixint, bbob-biobj-mixint)
    dim: int
        Dimension.
    fun_id: int
        The target function ID.
    alg: string
        The name of a given algorithm.
    all_instance_ids: interger list
        A list of instance IDs [1, 2, 3, 4, 5]   
    fevals_dir_path: path
        A directory path to save the number of function evaluations.

    Returns
    -------    

    """            
    # targets = 10^{2}, 10^{1.8}, ..., 10^{-7.8}, 10^{-8}
    targets = [10**i for i in np.arange(2, -8.1, -0.2)]
    # target_pows = 2, 1.8, ..., -7.8, -8
    target_pows = np.arange(2, -8.1, -0.2)
    target_pows = np.round(target_pows, 2)
    target_pows = target_pows.astype(str)

    # Experimental data are distributed here and there. 
    for target_instance_id in range(1, 5+1):        
        # Note that the file "*_i{}.dat" does not always correspond to the result on the {}-th function instance. The glob function may return file names with the same instance ID.
        # E.g., ['bbob-largescale_data/sepCMA/fmin_on_bbob-largescale_batch002of16_budget50000xD/data_f1/bbobexp_f1_DIM40_i1.dat', 'bbob-largescale_data/sepCMA/fmin_on_bbob-largescale_batch001of16_budget50000xD/data_f1/bbobexp_f1_DIM40_i1.dat']
        info_file_path_list = glob.glob('{}_exdata/{}/**/bbobexp_f{}.info'.format(bbob_suite, alg, fun_id), recursive=True)
        info_file_path_list.extend(glob.glob('{}_exdata/{}/**/bbobexp_f{}_i*.info'.format(bbob_suite, alg, fun_id), recursive=True))

        for info_file_path in info_file_path_list:                  
            true_instance_id_list = []
            res_file_name = ''
            max_fevals = 0
            
            with open(info_file_path, 'r') as fh:
                for str_line in fh:
                    #if 'bbobexp_f{}_DIM{}.'.format(fun_id, dim) in str_line or 'bbobexp_f{}_DIM{}_'.format(fun_id, dim) in str_line:
                    if '_f{}_DIM{}.'.format(fun_id, dim) in str_line or '_f{}_DIM{}_'.format(fun_id, dim) in str_line:
                        for tmp in str_line.split(', '):
                            if ':' in tmp:
                                true_instance_id = int(tmp.split(':')[0])
                                #max_fevals = int(tmp.split(':')[1])
                                max_fevals = int(tmp.split(':')[1].split('|')[0])
                                
                                true_instance_id_list.append(true_instance_id)

                                if true_instance_id == target_instance_id:
                                    res_file_name = str_line.split(', ')[0]
                                    res_file_name = res_file_name.replace('\\', '/')
                                    break
                                
                    if target_instance_id in true_instance_id_list:
                        break
                            
            if target_instance_id in true_instance_id_list:
                # The index method of the list object returns the first index whose element matches a given element. For example, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5].index(2) returns 1, while the 1-st, 6-th, and 11-th elements are 2. Since our benchmarking framework forcuses only on the first five instances, the property of this method is not problematic. However, when forcusing on ALL THE 15 INSTANCES, the current implementation cause significant trouble.
                data_pos = true_instance_id_list.index(target_instance_id)
                res_file_path = os.path.join(os.path.dirname(info_file_path), res_file_name)

                out_file_path = os.path.join(fevals_dir_path, 'f{}_DIM{}_i{}.csv'.format(fun_id, dim, target_instance_id))
                pp_bbob_data_file(res_file_path, data_pos, out_file_path, targets, target_pows, max_fevals)

def calc_ert_sp1(bbob_suite, dim, fun_id, alg, all_instance_ids, ert_file_path, sp1_file_path):        
    """
    Calculate the ERT and SP1 values.

    Parameters
    ----------
    bbob_suite: string
        The name of a bbob suite (bbob, bbob-noisy, bbob-biobj, bbob-largescale, bbob-mixint, bbob-biobj-mixint)
    dim: int
        Dimension.
    fun_id: int
        The target function ID.
    alg: string
        The name of a given algorithm.
    all_instance_ids: interger list
        A list of instance IDs [1, 2, 3, 4, 5]   
    ert_file_path: path
        A file path to save the ERT value.
    sp1_file_path: path
        A file path to save the SP1 value.

    Returns
    -------    

    """        

    # targets = 10^{2}, 10^{1.8}, ..., 10^{-7.8}, 10^{-8}
    targets = [10**i for i in np.arange(2, -8.1, -0.2)]
    # target_pows = 2, 1.8, ..., -7.8, -8
    target_pows = np.arange(2, -8.1, -0.2)
    target_pows = np.round(target_pows, 2)
    target_pows = target_pows.astype(str)

    sum_fevals = {}
    sum_succ_fevals = {}
    sum_successes = {}
    
    for target in target_pows:
        sum_fevals[target] = 0
        sum_succ_fevals[target] = 0
        sum_successes[target] = 0            
        
    for instance_id in all_instance_ids:
        fevals_file_path = './pp_{}_exdata/fevals_to_reach/{}/f{}_DIM{}_i{}.csv'.format(bbob_suite, alg, fun_id, dim, instance_id)
        with open(fevals_file_path, 'r') as fh:
            for str_line in fh:
                target, feval, is_success = str_line.split(',')
                is_success = float(is_success.replace('\n',''))

                sum_fevals[target] += float(feval)
                sum_successes[target] += float(is_success)
                if int(is_success) == 1:
                    sum_succ_fevals[target] += float(feval)
                
    with open(ert_file_path, 'w') as fh:
        for target in target_pows[::-1]:
            if sum_successes[target] > 0:
                ert_value = sum_fevals[target] / sum_successes[target]
                fh.write("{},{}\n".format(target, ert_value))
            else:
                fh.write("{},{}\n".format(target, np.nan))

    with open(sp1_file_path, 'w') as fh:
        for target in target_pows[::-1]:
            if sum_successes[target] > 0:
                succ_prob = sum_successes[target] / len(all_instance_ids)
                mean_succ_feval = sum_succ_fevals[target] / sum_successes[target]
                sp1_value = mean_succ_feval / succ_prob
                fh.write("{},{}\n".format(target, sp1_value))
            else:
                fh.write("{},{}\n".format(target, np.nan))
                
# def sanity_check(dim, fun_id, alg):
#     for instance_id in range(1, 5+1):
#         fevals_file_path = 'pp_bbob_exdata/fevals_to_reach/{}/f{}_DIM{}_i{}.csv'.format(alg, fun_id, dim, instance_id)
#         if os.path.exists(fevals_file_path) == False:
#             logger.error("{} does not exist".format(fevals_file_path))
            
def calc_ranking(bbob_suite, dim, fun_id, ranking_dir_path, all_algs, target='-2.0', per_metric='ert'):
    """
    Calculate the rankings of optimizers based on a given performance metric.

    Parameters
    ----------
    bbob_suite: string
        The name of a bbob suite (bbob, bbob-noisy, bbob-biobj, bbob-largescale, bbob-mixint, bbob-biobj-mixint)
    dim: int
        Dimension.
    fun_id: int
        The target function ID.
    ranking_dir_path: path
        A file path.
    target: string
        The error threshold value.
    target: per_metric
        A performance metric (ERT or SP1)

    Returns
    -------    

    """        

    out_file_path = os.path.join(ranking_dir_path, 'f{}_DIM{}.csv'.format(fun_id, dim))
    performance_df = pd.DataFrame(columns=['alg', 'metric'], index=range(len(all_algs)))
    
    for i, alg in enumerate(all_algs):
        # "liid0" means results on all five instances (IIDs: 1, 2, 3, 4, 5).
        metric_file_path = './pp_{}_exdata/{}/{}/f{}_DIM{}_liid0.csv'.format(bbob_suite, per_metric, alg, fun_id, dim) 
        alg_df = pd.read_csv(metric_file_path, names=('target', 'metric'))
        j = alg_df.query('target =={}'.format(target)).index[0]
        performance_df.loc[i, 'alg'] = alg
        performance_df.loc[i, 'metric'] = alg_df.loc[j, 'metric']
        
    performance_df = performance_df.sort_values('metric')
    performance_df.to_csv(out_file_path, index=False)

def instance_ids_lo(n_iids=5, liid=0):
    """
    Compute a set of n_iids - 1 instance IDs.

    Parameters
    ----------
    n_iids: int
        The number of instances
    liid: int
        The instance ID to be left.

    Returns
    -------    
    iids: interger list
        A set of instance IDs without liid.
    """        
    iids = []
    for i in range(1, n_iids+1):
        if i != liid:
            iids.append(i)
    return iids

def ecdf_single(bbob_suite, alg, all_fun_ids, test_instance_ids, dim, target, ecdf_file_path):
    """
    Compute the proportion of function instances reached by an optimizer.

    Parameters
    ----------
    bbob_suite: string
        The name of a bbob suite (bbob, bbob-noisy, bbob-biobj, bbob-largescale, bbob-mixint, bbob-biobj-mixint)
    alg: string
        The name of an optimizer.
    all_fun_ids: interger list
        A set of function IDs.        
    test_instance_ids: interger list
        A set of instance IDs.    
    dim: int
        Dimension.
    target: float
        The error threshold value.
    ecdf_file_path: path
        A file path.

    Returns
    -------    

    """        
    target_pows = np.arange(2, -2.1, -0.2)
    target_pows = np.round(target_pows, 2)
    
    solved_counts = np.zeros(dim * 10**6+1)
    for fun_id in all_fun_ids:
        for instance_id in test_instance_ids:
            fevals_file_path = os.path.join('pp_bbob_exdata/fevals_to_reach', alg, 'f{}_DIM{}_i{}.csv'.format(fun_id, dim, instance_id))
            #print(fevals_file_path)
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

def run():
    """
    Post-processing experimental data obtained on the noiseless BBOB function set
    """        
    # Extract the algorithm name and the url link to data
    bbob_suite = 'bbob'
    exdata_dir_path = './{}_exdata'.format(bbob_suite)
    os.makedirs(exdata_dir_path, exist_ok=True)

    # For the noiseless BBOB function set
    all_fun_ids = range(1, 24+1)
    if bbob_suite == 'bbob-noisy':
        all_fun_ids = range(101, 130+1)
    
    # Download a markdown file 'bbob.md' from https://raw.githubusercontent.com/numbbo/data-archive/master/bbob.md (or, e.g., bbob-noisy.md). 'bbob.md' provide names and URL links to performance data of all algorithms benchmarked on the noiseless BBOB suite.
    logger.info('==================== Download a markdown file %s.md ====================', bbob_suite)    
    url = 'https://raw.githubusercontent.com/numbbo/data-archive/master/{}.md'.format(bbob_suite)
    download_file(url, os.path.join(exdata_dir_path, os.path.basename(url)))

    # Extract the name and the URL link of each algorithm from bbob.md
    # The incomplete_data_sets list include incomplete performance data to be removed. I manually found 'JADEb', 'EvoSpace-PSO-GA', 'Ord-N-DTS-CMA-ES' are incompleted. 'BFGS-P-09' is incompleted for 20 dimensions
    # For bbob-noisy,  'CMAEGS' and 'xNESas' are incomplete.
    # TODO: Implement an automatic sanity check function.
    make_alg_url_file(bbob_suite, alg_url_file_path='{}/alg_url.csv'.format(exdata_dir_path), bbob_md_file_path='{}/{}.md'.format(exdata_dir_path, bbob_suite), incomplete_data_sets=['JADEb', 'EvoSpace-PSO-GA', 'Ord-N-DTS-CMA-ES', 'BFGS-P-09', 'BFGS-P-Instances', 'BFGS-P-range', 'BFGS-P-StPt'])        
    alg_url = np.loadtxt('{}/alg_url.csv'.format(exdata_dir_path), delimiter=",", comments="#", dtype=np.str)

    # Download performance data for all algorithms listed in alg_url
    logger.info('==================== Download exdata ====================')
    download_data(alg_url, bbob_suite)
    
    # Unpack the tar files
    logger.info('==================== Unpack the tar files ====================')
    unpack_data(exdata_dir_path, alg_url)
    
    # Calculate the number of function evaluations to reach a target in {10^{2}, 10^{1.8}, ..., 10^{-7.8}, 10^{-8}}. Results are saved in ./pp_bbob_exdata/fevals_to_reach.
    # NOTE: Only the first five instances are considered.
    logger.info('==================== Calculate the number of function evaluations to reach targets ====================')
    i = 0
    for alg, url in alg_url:
        fevals_dir_path = os.path.join('pp_{}_exdata'.format(bbob_suite), 'fevals_to_reach', alg)
        if os.path.exists(fevals_dir_path):
            logger.warning("The directory %s already exists, so FEvals was not calculated", fevals_dir_path)
            continue        
        os.makedirs(fevals_dir_path, exist_ok=True)

        for dim in [2, 3, 5, 10]:
            for fun_id in all_fun_ids:
                pp_fevals_to_reach(bbob_suite, dim, fun_id, alg, fevals_dir_path)
                
        i += 1
        logger.info("%s: FEvals have been postprocessed (%d)", alg, i)
        
    # Calculate the ERT values of each algorithm for targets {10^{2}, 10^{1.8}, ..., 10^{-7.8}, 10^{-8}}
    logger.info('==================== Calculate the ERT and SP1 values for each target ====================')
    i = 0
    for alg, url in alg_url:
        ert_dir_path = os.path.join('pp_{}_exdata'.format(bbob_suite), 'ert', alg)
        if os.path.exists(ert_dir_path):
            logger.warning("The directory %s already exists, so ERT was not calculated", ert_dir_path)
            continue        
        os.makedirs(ert_dir_path, exist_ok=True)

        sp1_dir_path = os.path.join('pp_{}_exdata'.format(bbob_suite), 'sp1', alg)
        if os.path.exists(sp1_dir_path):
            logger.warning("The directory %s already exists, so SP1 was not calculated", sp1_dir_path)
            continue        
        os.makedirs(sp1_dir_path, exist_ok=True)
                
        for dim in [2, 3, 5, 10]:
            for fun_id in all_fun_ids:
                for left_instance_id in range(0, 5+1):
                    ert_file_path = os.path.join(ert_dir_path, 'f{}_DIM{}_liid{}.csv'.format(fun_id, dim, left_instance_id))
                    sp1_file_path = os.path.join(sp1_dir_path, 'f{}_DIM{}_liid{}.csv'.format(fun_id, dim, left_instance_id))                    
                    all_instance_ids = instance_ids_lo(5, left_instance_id)
                    calc_ert_sp1(bbob_suite, dim, fun_id, alg, all_instance_ids, ert_file_path, sp1_file_path)
                        
        i += 1
        logger.info("%s: ERT and SP1 have been postprocessed (%d)", alg, i)
        
    # Rank all the optimizers for each function and each dimension.
    logger.info('==================== Rank all the optimizers ====================')
    target = '-2.0'
    for per_metric in ['ert', 'sp1']:
        ranking_dir_path = './pp_{}_exdata/ranking_{}_target{}'.format(bbob_suite, per_metric, target)
        if os.path.exists(ranking_dir_path):
            logger.warning("The directory %s already exists, so the ranking was not calculated", ranking_dir_path)
        else:
            os.makedirs(ranking_dir_path, exist_ok=True)
            for dim in [2, 3, 5, 10]:
                for fun_id in all_fun_ids:
                    calc_ranking(bbob_suite, dim, fun_id, ranking_dir_path, all_algs=alg_url[:, 0], target=target, per_metric=per_metric)
                    logger.info("Rankings with targets 10^%s based on %s have been postprocessed (f=%d, dim=%d)", target, per_metric, fun_id, dim)

    logger.info('==================== Calculate the propotion of function instances reached for the ECDF figure ====================')
    i = 0
    for alg, url in alg_url:
        ecdf_dir_path = os.path.join('pp_{}_exdata'.format(bbob_suite), 'ecdf', alg)
        if os.path.exists(ecdf_dir_path):
            logger.warning("The directory %s already exists, so the propotion of function instances reached was not calculated", ecdf_dir_path)
            continue
        
        os.makedirs(ecdf_dir_path, exist_ok=True)

        test_instance_ids = [1, 2, 3, 4, 5]
        for dim in [2, 3, 5, 10]:
            ecdf_file_path = os.path.join(ecdf_dir_path, 'DIM{}.csv'.format(dim))
            ecdf_single(bbob_suite, alg, all_fun_ids, test_instance_ids, dim, float(target), ecdf_file_path)
                        
        i += 1
        logger.info("%s: The propotion of function instances reached has been postprocessed (%d)", alg, i)
                
if __name__ == '__main__':
    run()
