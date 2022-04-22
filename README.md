# Benchmarking Feature-based Algorithm Selection Systems for Black-box Numerical Optimization

This repository provides the source code to reproduce results shown in the following paper.

> Ryoji Tanabe, **Benchmarking Feature-based Algorithm Selection Systems for Black-box Numerical Optimization**,  IEEE Transactions on Evolutionary Computation, [pdf](https://arxiv.org/abs/2109.08377)

The source code highly depends on the COCO framework and the flacco package:

> Nikolaus Hansen, Anne Auger, Raymond Ros, Olaf Mersmann, Tea Tusar, and Dimo Brockhoff, **COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting**, Optimization Methods and Software, 36(1): 114-144 (2021), [link](https://arxiv.org/abs/1603.08785)

> Pascal Kerschke and Heike Trautmann, **Comprehensive Feature-Based Landscape Analysis of Continuous and Constrained Optimization Problems Using the R-package flacco**. In Applications in Statistical Computing - From Music Data Analysis to Industrial Quality Improvement, 93-123 (2019), [link](https://arxiv.org/abs/1708.05258)
 
# Requirements

This code require R (=>3.6), Python (=>3.8), numpy, sklearn, pyDOE, and sobol_seq. In addition, this code require the [flacco](https://github.com/kerschke/flacco) package (R), the [pflacco](https://github.com/Reiyan/pflacco) interface (Python), the [cocoex](https://github.com/numbbo/coco) module (Python), and [SMAC3](https://automl.github.io/SMAC3/master/). Optionally, this code require [the Torque manager](https://github.com/adaptivecomputing/torque) for batch style processing.

As discussed [here](https://stackoverflow.com/questions/68176331/from-scipy-optimize-shgo-lib-sobol-seq-import-sobol-does-not-work), SMAC3 is incompatible with scipy 1.7. pflacco is also incompatible with the latest version of rpy2 (=> 3.4.5). You may need to downgrade the versions of both scipy and rpy2 by e.g., ``pip install -U scipy==1.6`` and ``pip install -U rpy2==3.3.6``.

# Usage

Because each step is extremely time-consuming, I divided the overall process into multiple steps.
 
## 1. Postprocess performance data on the 24 noiseless BBOB functions

This program does not require an actual algorithm run. Instead, this program uses the benchmarking results of optimizers who participated in [the BBOB workshop](https://numbbo.github.io/workshops/index.html). For this purpose, we use [COCO data archives](https://numbbo.github.io/data-archive/). [The COCO data archive for the noiseless BBOB suite](https://numbbo.github.io/data-archive/bbob/) provides names and URL links to benchmarking results of optimizers on the noiseless BBOB suite. The following command postprocesses the results on the 24 noiseless BBOB functions.

```
$ python pp_exdata.py
```

Incomplete performance data are excluded from the postprocessing process. Currently, complete data of 209 algorithms are available. The postprocessed data are saved in the "./pp\_bbob\_exdata" directory.

* ``./pp_bbob_exdata/fevals_to_reach``: For each algorithm, each dimension, each function, and each instance, the number of function evaluations to reach a given target value is saved in this directory.
* ``./pp_bbob_exdata/ert``: For each algorithm, each dimension, and each function, the ERT value for a target value is saved in this directory.
* ``./pp_bbob_exdata/sp1``: For each algorithm, each dimension, and each function, the SP1 value for a target value is saved in this directory.
* ``./pp_bbob_exdata/ranking_ert_target-2.0`` For each dimension and each function, the ranking of all algorithms based on their ERT values is saved in this directory. ``-2.0`` represents that a given target value was 10^-2.0.
* ``./pp_bbob_exdata/ranking_sp1_target-2.0`` For each dimension and each function, the ranking of all algorithms based on their SP1 values is saved in this directory. ``-2.0`` represents that a given target value was 10^-2.0.

## 2. Construct an algorithm portfolio

Algorithm selection requires a portfolio of multiple algorithms. The following command constructs the 14 algorithm portfolios ($A _ {kt}$, ..., $A _ {ls18}$) used in the above paper. The results are saved in the "./alg\_portfolio" directory. 

```
$ python portfolio_construction.py
```

If you want to run the local search method described in the paper, please uncomment ``run_ls()``.

## 3. Generate the sample of solutions

The initial sample (i.e., a set of solutions X) is needed for each function instance. The following command generates samples of the size 50 * DIM on the BBOB functions with 2, 3, 5, 10 dimensions by the improved Latin hypercube sampling method. The results are saved in the "./sample_data" directory. Here, ``sample_id`` is the ID of each sample. 31 independent runs can be performed by setting ``sample_id`` from 0 to 30.

```
$ python sampler.py --sampling_method ihs --sample_multiplier 50 --sample_id 0
```

Optionally, 31 runs of the sampler can be performed in a batch manner by using Torque as follows:

```
$ python submit_job_sample.py
```


## 4. Run a pre-solver

This program supports the use of a pre-solver, which is usually performed independently from the sampling method. The following command runs SLSQP with 50*DIM function evaluations. The results of SLSQP are saved in the "./presolver_data" directory.

```
$ python presolver.py --presolver slsqp --budget_multiplier 50 --sample_id 0
```

In addition to SLSQP, SMAC can be used as a pre-solver:

```
$ python presolver.py --presolver smac --budget_multiplier 50 --sample_id 0
```

Optionally, 31 runs of the pre-solver can be performed in a batch manner by using [the Torque manager](https://github.com/adaptivecomputing/torque) as follows:

```
$ python submit_job_presolver.py
```

## 5. Compute features

Features are computed based on the sample for each function instance. The following command computes the 'basic' features based on the sample in "ihs\_multiplier50\_sid0" on the 2-dimensional $f \_ 1$ in the BBOB function set.  The results are saved in the "./ela\_feature\_dataset" directory.

```
$ python feature_computation.py --dir_sampling_method ihs_multiplier50_sid0 --ela_feature_class basic --dim 2 --fun_id 1
```

Optionally, features can be computed in a batch manner by using Torque. The following command computes nine feature classes ('basic', 'ela\_distr', 'pca', 'limo', 'ic', 'disp', 'nbc', 'ela\_level', and 'ela\_meta') on the 24 BBOB functions with 2, 3, 5, and 10 dimensions. 31 runs are performed.

```
$ python submit_job_fc.py
```

## 6. Make table data

After the features have been computed, they should be summarized as table data. The results of algorithms in an algorithm portfolio should also be summarized in the same table. The following command makes a table that consists of the 'basic', 'ela\_distr', 'pca', 'limo', 'ic', 'disp', 'nbc', 'ela\_level', and 'ela\_meta' classes on the 24 BBOB functions with 2, 3, 5, and 10 dimensions for each of the 14 portfolios ($A _ {kt}$, ..., $A _ {ls18}$). The results are saved in the "./alg\_portfolio/PORTFOLIO\_NAME/feature\_table\_data" directory.

```
$ python table_data.py
```

## 7. Perform algorithm selection

The following command performs algorithm selection with the classification-based selection method on the portfolio $A _ {kt}$. The nine feature classes computed based on the sample "ihs\_multiplier50\_sid0" are used. Algorithm selection is performed for the LOPO-CV for 2 dimension, and $f1$ is used in the testing phase. The results of algorithm selection are saved in the "./as\_results" directory.

```
$ python alg_selection.py --ap_name kt --dir_sampling_method ihs_multiplier50_sid0 --as_run_id 0 --ela_feature_classes basic_ela_distr_pca_limo_ic_disp_nbc_ela_level_ela_meta --dims dims2_3_5_10 --selector multiclass_classification --cross_valid_type lopo_cv --feature_selector none --dim 2 --left_fun_id 1
```

Optionally, algorithm selection can be conducted in a batch manner by using Torque. The following command performs algorithm selection with the five selection methods for all the 14 portfolios on the 24 BBOB functions with 2, 3, 5, and 10 dimensions for the LOIO-CV, the LOPO-CV, and the LOPOAD-CV. 31 runs are performed.

```
$ python submit_job_as.py
```

## 9. Postprocess results of algorithm selection

The following command postprocesses results of algorithm selection with and without a pre-solver. The SP1, relSP1, ERT, and relERT values are calculated based on results of the pre-solver and algorithm selection. The postprocessing results  are saved in the "./pp\_as\_results" directory. 

```
$ python pp_as_results.py
```

Since the pre-solving phase is totally independent from the algorithm selection process, our postprocessing considers the results of the pre-solver this time. Please note that this postprocessing method is not practical.

The results can be further postprocessed by using plot\_box.py, plot\_comp2.py  plot\_n\_sbs.py, print\_per\_metric.py, pp\_per\_score.py , and pp\_per\_metric\_union\_ap.py. For details, please refer to the file.

