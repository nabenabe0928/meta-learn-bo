import unittest

import pytest

from meta_learn_bo.rgpe import RankingWeightedGaussianProcessEnsemble
from meta_learn_bo.taf import TransferAcquisitionFunction
from meta_learn_bo.utils import get_train_data

import torch

from utils import (
    get_kwargs_and_observations,
    get_kwargs_and_observations_for_categorical,
    obj_func,
    obj_func_for_categorical,
)


def test_validate_input_and_properties() -> None:
    for acq_fn_type in ["parego", "ehvi"]:
        kwargs, observations = get_kwargs_and_observations(size=5)
        kwargs.update(n_bootstraps=50, acq_fn_type=acq_fn_type)
        metadata = {}
        _, metadata["src20"] = get_kwargs_and_observations(size=10)
        _, metadata["src30"] = get_kwargs_and_observations(size=15)
        rgpe = RankingWeightedGaussianProcessEnsemble(init_data=observations, metadata=metadata, **kwargs)

        original = rgpe._task_names[0]
        rgpe._task_names[0] = rgpe._target_task_name
        with pytest.raises(ValueError):
            rgpe._validate_input()
        rgpe._task_names[0] = original

        original = rgpe._hp_names.copy()
        rgpe._hp_names.pop(-1)
        with pytest.raises(ValueError):
            rgpe._validate_input()
        rgpe._hp_names = original

        original = rgpe._hp_names.copy()
        rgpe._hp_names.append("dummy")
        with pytest.raises(ValueError):
            rgpe._validate_input()
        rgpe._hp_names = original

        original = rgpe._metadata.copy()
        rgpe._metadata[rgpe._task_names[0]].pop(rgpe._hp_names[0])
        with pytest.raises(ValueError):
            rgpe._validate_input()
        rgpe._metadata = original

        assert isinstance(rgpe._task_weights_repr(), str)
        assert id(rgpe._observations) != id(rgpe.observations)
        assert isinstance(rgpe.acq_fn, TransferAcquisitionFunction)


def test_validate_input_categorical() -> None:
    categories = {"c0": ["a", "b"], "c1": ["A", "B", "C", "D", "E"]}
    for acq_fn_type in ["parego", "ehvi"]:
        kwargs, observations = get_kwargs_and_observations_for_categorical(size=5)
        kwargs.update(n_bootstraps=50, acq_fn_type=acq_fn_type, categories=categories)
        metadata = {}
        _, metadata["src20"] = get_kwargs_and_observations_for_categorical(size=10)
        _, metadata["src30"] = get_kwargs_and_observations_for_categorical(size=15)
        rgpe = RankingWeightedGaussianProcessEnsemble(init_data=observations, metadata=metadata, **kwargs)
        assert len(rgpe._cat_dims) == 2

        original = rgpe._categories.copy()
        rgpe._categories = None
        with pytest.raises(ValueError):
            rgpe._validate_input()
        rgpe._categories = original.copy()

        original = rgpe._categories.copy()
        rgpe._categories.pop("c0")
        with pytest.raises(ValueError):
            rgpe._validate_input()
        rgpe._categories = original.copy()

        original = rgpe._categories.copy()
        rgpe._categories["c0"] = [0, 1]
        with pytest.raises(ValueError):
            rgpe._validate_input()
        rgpe._categories = original.copy()

        original = rgpe._categories.copy()
        rgpe._categories["c0"] = ["a", "b", "c"]
        with pytest.raises(ValueError):
            rgpe._validate_input()
        rgpe._categories = original.copy()


def test_update():
    n_init = 5
    for acq_fn_type in ["parego", "ehvi"]:
        kwargs, observations = get_kwargs_and_observations(size=n_init)
        kwargs_for_proc = kwargs.copy()
        kwargs_for_proc["hp_names"] = list(kwargs_for_proc["hp_info"].keys())
        kwargs_for_proc.pop("hp_info")
        X_train, _ = get_train_data(observations, **kwargs_for_proc)
        kwargs.update(n_bootstraps=50, acq_fn_type=acq_fn_type)
        metadata = {}
        _, metadata["src20"] = get_kwargs_and_observations(size=10)
        _, metadata["src30"] = get_kwargs_and_observations(size=15)
        rgpe = RankingWeightedGaussianProcessEnsemble(init_data=observations, metadata=metadata, **kwargs)

        with torch.no_grad():
            model = rgpe._base_models[rgpe._task_names[0]]
            pred_old = model.posterior(X_train).mean

        eval_config = rgpe.optimize_acq_fn()
        results = obj_func(eval_config)
        rgpe.update(eval_config=eval_config, results=results)
        assert rgpe.observations[rgpe._hp_names[0]].size == n_init + 1

        with torch.no_grad():
            model = rgpe._base_models[rgpe._task_names[0]]
            pred_new = model.posterior(X_train).mean

        # for parego, base_model must be updated after each update
        # for ehvi, base_model outputs the same value
        n_close = torch.sum(torch.isclose(pred_old, pred_new))
        assert n_close == (0 if acq_fn_type == "parego" else 10)


def test_update_for_categorical():
    n_init = 3
    categories = {"c0": ["a", "b"], "c1": ["A", "B", "C", "D", "E"]}
    for acq_fn_type in ["parego", "ehvi"]:
        kwargs, observations = get_kwargs_and_observations_for_categorical(size=n_init)
        kwargs_for_proc = kwargs.copy()
        kwargs_for_proc["hp_names"] = list(kwargs_for_proc["hp_info"].keys())
        kwargs_for_proc.pop("hp_info")
        X_train, _ = get_train_data(observations, **kwargs_for_proc)
        kwargs.update(n_bootstraps=50, acq_fn_type=acq_fn_type)
        metadata = {}
        _, metadata["src20"] = get_kwargs_and_observations_for_categorical(size=4)
        _, metadata["src30"] = get_kwargs_and_observations_for_categorical(size=5)
        rgpe = RankingWeightedGaussianProcessEnsemble(
            init_data=observations, metadata=metadata, categories=categories, **kwargs
        )

        with torch.no_grad():
            model = rgpe._base_models[rgpe._task_names[0]]
            pred_old = model.posterior(X_train).mean

        for i in range(5):
            eval_config = rgpe.optimize_acq_fn()
            results = obj_func_for_categorical(eval_config)
            rgpe.update(eval_config=eval_config, results=results)
            assert rgpe.observations[rgpe._hp_names[0]].size == n_init + i + 1

        with torch.no_grad():
            model = rgpe._base_models[rgpe._task_names[0]]
            pred_new = model.posterior(X_train).mean

        # for parego, base_model must be updated after each update
        # for ehvi, base_model outputs the same value
        n_close = torch.sum(torch.isclose(pred_old, pred_new))
        assert n_close == (0 if acq_fn_type == "parego" else pred_new.shape[0] * 3)


def test_validate_config_and_results():
    n_init = 3
    categories = {"c0": ["a", "b"], "c1": ["A", "B", "C", "D", "E"]}
    acq_fn_type = "parego"
    kwargs, observations = get_kwargs_and_observations_for_categorical(size=n_init)
    kwargs_for_proc = kwargs.copy()
    kwargs_for_proc["hp_names"] = list(kwargs_for_proc["hp_info"].keys())
    kwargs_for_proc.pop("hp_info")
    X_train, _ = get_train_data(observations, **kwargs_for_proc)
    kwargs.update(n_bootstraps=50, acq_fn_type=acq_fn_type)
    metadata = {}
    _, metadata["src20"] = get_kwargs_and_observations_for_categorical(size=4)
    _, metadata["src30"] = get_kwargs_and_observations_for_categorical(size=5)
    rgpe = RankingWeightedGaussianProcessEnsemble(
        init_data=observations, metadata=metadata, categories=categories, **kwargs
    )
    eval_config = {"x0": 3.0, "x1": 7.6, "i0": -1, "i1": 3, "c0": "a", "c1": "D"}
    results = {"f1": 0.0, "f2": 1.0, "f3": 2.0}
    rgpe._validate_config_and_results(eval_config, results)

    eval_config = {"x1": 7.6, "i0": -1, "i1": 3, "c0": "a", "c1": "D"}
    with pytest.raises(KeyError):
        rgpe._validate_config_and_results(eval_config, results)

    eval_config = {"x0": 3.0, "x1": 7.6, "i0": -1, "i1": 3, "c0": "a", "c1": "D"}
    results = {"f2": 1.0, "f3": 2.0}
    with pytest.raises(KeyError):
        rgpe._validate_config_and_results(eval_config, results)

    eval_config = {"x0": 3.0, "x1": 7.6, "i0": -1, "i1": 3, "c0": 1, "c1": "D"}
    results = {"f1": 0.0, "f2": 1.0, "f3": 2.0}
    with pytest.raises(TypeError):
        rgpe._validate_config_and_results(eval_config, results)

    eval_config = {"x0": 3.0, "x1": 7.6, "i0": -1.0, "i1": 3, "c0": "a", "c1": "D"}
    with pytest.raises(TypeError):
        rgpe._validate_config_and_results(eval_config, results)

    eval_config = {"x0": 3.0, "x1": 7.6, "i0": -1, "i1": 3, "c0": "dummy", "c1": "D"}
    with pytest.raises(ValueError):
        rgpe._validate_config_and_results(eval_config, results)

    eval_config = {"x0": 3.0, "x1": 30.0, "i0": -1, "i1": 3, "c0": "a", "c1": "D"}
    with pytest.raises(ValueError):
        rgpe._validate_config_and_results(eval_config, results)


if __name__ == "__main__":
    unittest.main()
