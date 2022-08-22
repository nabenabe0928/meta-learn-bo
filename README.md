# Transfer learning for multi-objective Bayesian optimization
[![Build Status](https://github.com/nabenabe0928/meta-learn-bo/workflows/Functionality%20test/badge.svg?branch=main)](https://github.com/nabenabe0928/meta-learn-bo)
[![codecov](https://codecov.io/gh/nabenabe0928/meta-learn-bo/branch/main/graph/badge.svg?token=T6MX4JQHOV)](https://codecov.io/gh/nabenabe0928/meta-learn-bo)

The codes are written based on [this repository](https://github.com/automl/transfer-hpo-framework).

Since the goal of this repository to reproduce the performance of the original methods, we focus only on the default settings provided by the authors.
More specifically, RGPE supports only the transfer acquisition function (TAF) version in this repository.

Furthermore, this repository only supports multi-objective optimizaiton settings
although single-objective optimization settings can be easily obtained with small changes.

In this codebase, we are trying to reproduce the results of TST-R and RGPE using the TAF acquisition function in [`Practical transfer learning for Bayesian optimization`](https://arxiv.org/pdf/1802.02219v3.pdf).

You can find the information for RGPE in the paper and find the information that for TST-R in the paper [`Two-stage transfer surrogate model for automatic hyperparameter optimization`](https://www.ismll.uni-hildesheim.de/pub/pdfs/wistuba_et_al_ECML_2016.pdf).

## Initial setup

Optionally, you can create a new conda environment and run the following:

```shell
$ pip install -r requirements.txt
```

You can test by:

```shell
$ python -m examples.example_toy_func
```
