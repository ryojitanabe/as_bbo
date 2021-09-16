#!/bin/sh

cd $PBS_O_WORKDIR

python feature_computation.py $arg1 $arg2 $arg3 $arg4
