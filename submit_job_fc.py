#!/usr/bin/env python
import subprocess

if __name__ == '__main__':
    sampling_method = 'ihs'
    sample_multiplier = 50
    
    for sid in range(0, 31):
        dir_sampling_method = '{}_multiplier{}_sid{}'.format(sampling_method, sample_multiplier, sid)                
        for ela_feature_class in ['basic', 'ela_distr', 'pca', 'limo', 'ic', 'disp', 'nbc', 'ela_level', 'ela_meta']:
            for dim in [2, 3, 5, 10]:
                for fun_id in range(1, 24+1):
                    s_arg1 = 'arg1=--dir_sampling_method {}'.format(dir_sampling_method)
                    s_arg2 = 'arg2=--ela_feature_class {}'.format(ela_feature_class)
                    s_arg3 = 'arg3=--dim {}'.format(dim)
                    s_arg4 = 'arg4=--fun_id {}'.format(fun_id)
                    s_args = ','.join([s_arg1, s_arg2, s_arg3, s_arg4])
                    subprocess.run(['qsub', '-l', 'walltime=72:00:00', '-v', s_args, 'job_fc.sh'])                
