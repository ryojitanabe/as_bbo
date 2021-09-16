#!/usr/bin/env python
import subprocess

if __name__ == '__main__':
    sample_multiplier = 50
    sampling_method = 'ihs'

    for sid in range(0, 31):
        s_arg1 = 'arg1=--sample_multiplier {}'.format(sample_multiplier)
        s_arg2 = 'arg2=--sampling_method {}'.format(sampling_method)
        s_arg3 = 'arg3=--sample_id {}'.format(sid)
        s_args = ','.join([s_arg1, s_arg2, s_arg3])
        subprocess.run(['qsub', '-l', 'walltime=72:00:00', '-v', s_args, 'job_sample.sh'])
