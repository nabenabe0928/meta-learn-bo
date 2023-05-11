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

## Code example

The Bayesian optimization using this package is performed as follows.
For more details, please check [examples](examples/).

```python
import warnings

from meta_learn_bo import (
    HyperParameterType,
    MetaLearnGPSampler,
    RankingWeightedGaussianProcessEnsemble,
    get_random_samples,
)


warnings.filterwarnings("ignore")


def func(eval_config, shift=0):
    assert eval_config["i"] in [-2, -1, 0, 1, 2]
    assert eval_config["c"] in ["A", "B", "C"]

    x, y = eval_config["x"], eval_config["y"]
    f1 = (x + shift) ** 2 + (y + shift) ** 2
    f2 = (x - 2 + shift) ** 2 + (y - 2 + shift) ** 2
    return {"f1": f1, "f2": f2}


def run():
    bounds = {"x": (-5, 5), "y": (-5, 5), "i": (-2, 2), "c": (0, 2)}
    hp_info = {
        "x": HyperParameterType.Continuous,
        "y": HyperParameterType.Continuous,
        "i": HyperParameterType.Integer,
        "c": HyperParameterType.Categorical,
    }
    minimize = {"f1": True, "f2": True}
    categories = {"c": ["A", "B", "C"]}
    kwargs = dict(minimize=minimize, bounds=bounds, hp_info=hp_info, categories=categories)

    metadata = {
        "src": get_random_samples(n_samples=30, obj_func=lambda eval_config: func(eval_config, shift=2), **kwargs)
    }
    init_data = get_random_samples(n_samples=10, obj_func=func, **kwargs)

    rgpe = RankingWeightedGaussianProcessEnsemble(init_data=init_data, metadata=metadata, **kwargs)
    sampler = MetaLearnGPSampler(max_evals=20, obj_func=func, model=rgpe, **kwargs)
    sampler.optimize()
    print(sampler.observations)


if __name__ == "__main__":
    run()

```

# Citations

For the citation, please use the following format:
```
@article{watanabe2023ctpe,
  title={Speeding up Multi-objective Hyperparameter Optimization by Task Similarity-Based Meta-Learning for the Tree-structured {P}arzen Estimator},
  author={S. Watanabe and N. Awad and M. Onishi and F. Hutter},
  journal={International Joint Conference on Artificial Intelligence},
  year={2023}
}
```
