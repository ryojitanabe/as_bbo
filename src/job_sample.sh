#!/bin/sh

cd $PBS_O_WORKDIR

python sampler.py $arg1 $arg2 $arg3
