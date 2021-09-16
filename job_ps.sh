#!/bin/sh

cd $PBS_O_WORKDIR

python presolver.py $arg1 $arg2 $arg3
