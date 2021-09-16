#!/usr/bin/env python
import subprocess

if __name__ == '__main__':
    presolver = 'slsqp'
    budget_multiplier = 50
    
    for sid in range(0, 31):
        s_arg1 = 'arg1=--budget_multiplier {}'.format(budget_multiplier)
        s_arg2 = 'arg2=--presolver {}'.format(presolver)
        s_arg3 = 'arg3=--sample_id {}'.format(sid)
        s_args = ','.join([s_arg1, s_arg2, s_arg3])
        subprocess.run(['qsub', '-l', 'walltime=72:00:00', '-v', s_args, 'job_ps.sh'])
