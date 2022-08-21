import unittest
from typing import Dict, List, Tuple

import pytest

from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP

from meta_learn_bo.utils import (
    HyperParameterType,
    NumericType,
    convert_categories_into_index,
    denormalize,
    fit_model,
    get_acq_fn,
    get_fixed_features_list,
    get_model_and_train_data,
    get_train_data,
    normalize,
    optimize_acq_fn,
    sample,
    scalarize,
    validate_weights,
)

import numpy as np

import torch


N_SAMPLES = 10


def test_hyperparameter_type():
    assert HyperParameterType.Categorical == str
    assert HyperParameterType.Categorical != float
    assert HyperParameterType.Categorical != int

    assert HyperParameterType.Continuous != str
    assert HyperParameterType.Continuous == float
    assert HyperParameterType.Continuous != int

    assert HyperParameterType.Integer != str
    assert HyperParameterType.Integer != float
    assert HyperParameterType.Integer == int


def get_random_observations(
    size: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[NumericType, NumericType]], List[str], Dict[str, bool]]:
    hp_names = ["x0", "x1"]
    bounds = {
        hp_names[0]: (-5.0, 5.0),
        hp_names[1]: (5.0, 25.0),
    }
    observations = {
        hp_name: np.random.random(size) * (bounds[hp_name][1] - bounds[hp_name][0]) + bounds[hp_name][0]
        for hp_name in hp_names
    }
    obj_names = ["f1", "f2", "f3"]
    observations.update({obj_name: np.random.random(size) * 2 - 1 for obj_name in obj_names})
    minimize = {obj_name: bool((i + 1) % 2) for i, obj_name in enumerate(obj_names)}
    return observations, bounds, hp_names, minimize


def get_categorical_random_observations(
    size: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[NumericType, NumericType]], List[str], Dict[str, bool]]:
    hp_names = ["x0", "x1", "c0", "c1"]
    bounds = {
        hp_names[0]: (-5.0, 5.0),
        hp_names[1]: (5.0, 25.0),
        hp_names[2]: (0, 1),
        hp_names[3]: (0, 4),
    }
    observations = {
        hp_name: np.random.random(size) * (bounds[hp_name][1] - bounds[hp_name][0]) + bounds[hp_name][0]
        for hp_name in hp_names[:2]
    }
    observations.update({hp_name: np.random.randint(bounds[hp_name][-1] + 1, size=size) for hp_name in hp_names[2:]})
    obj_names = ["f1", "f2", "f3"]
    observations.update({obj_name: np.random.random(size) * 2 - 1 for obj_name in obj_names})
    minimize = {obj_name: bool((i + 1) % 2) for i, obj_name in enumerate(obj_names)}
    return observations, bounds, hp_names, minimize


def test_validate_weights() -> None:
    w = torch.arange(5) / 3
    with pytest.raises(ValueError):
        validate_weights(weights=w)

    w = torch.rand(5)
    w /= w.sum()
    validate_weights(weights=w)


def test_normalize() -> None:
    n_samples = N_SAMPLES
    observations, bounds, hp_names, _ = get_random_observations(size=n_samples)
    X = normalize(observations=observations, bounds=bounds, hp_names=hp_names)
    assert X.shape == (2, n_samples)
    assert torch.all(torch.logical_and(0 <= X, X <= 1))


def test_denormalize() -> None:
    observations, bounds, hp_names, _ = get_random_observations(1)
    X = normalize(observations=observations, bounds=bounds, hp_names=hp_names).squeeze()
    X_de = denormalize(X=X, bounds=bounds, hp_names=hp_names)
    for hp_name in hp_names:
        assert np.isclose(observations[hp_name][0], X_de[hp_name])

    with pytest.raises(ValueError):
        X = normalize(observations=observations, bounds=bounds, hp_names=hp_names)
        X_de = denormalize(X=X, bounds=bounds, hp_names=hp_names)


def test_get_train_data():
    n_samples = N_SAMPLES
    observations, bounds, hp_names, minimize = get_random_observations(size=n_samples)
    X_train, Y_train = get_train_data(observations, bounds, hp_names, minimize)
    X_ans = normalize(observations, bounds, hp_names).T
    assert X_train.shape == (n_samples, 2)
    assert Y_train.shape == (3, n_samples)
    assert torch.allclose(X_train, X_ans)
    assert torch.allclose(Y_train.mean(dim=-1), torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(Y_train.std(dim=-1), torch.ones(3, dtype=torch.float64))
    Y_ans = torch.tensor(np.asarray([observations[obj_name] for obj_name in minimize.keys()]))
    Y_ans = (Y_ans - Y_ans.mean(dim=-1)[:, None]) / Y_ans.std(dim=-1)[:, None]
    for idx, do_min in enumerate(minimize.values()):
        assert torch.allclose(Y_ans[idx], (1 - 2 * do_min) * Y_train[idx])

    X_train, Y_train = get_train_data(observations, bounds, hp_names, minimize, weights=torch.tensor([0.5, 0.3, 0.2]))
    assert X_train.shape == (n_samples, 2)
    assert Y_train.shape == (n_samples,)
    assert torch.allclose(X_train, X_ans)
    assert torch.allclose(Y_train.mean(dim=-1), torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(Y_train.std(dim=-1), torch.ones(3, dtype=torch.float64))


def test_scalarize() -> None:
    n_samples = N_SAMPLES
    observations, bounds, hp_names, minimize = get_random_observations(size=n_samples)
    _, Y_train = get_train_data(observations, bounds, hp_names, minimize)
    Y = scalarize(Y_train, weights=torch.tensor([0.5, 0.3, 0.2]))
    assert Y.shape == (n_samples,)


def test_sample() -> None:
    n_samples = N_SAMPLES
    observations, bounds, hp_names, minimize = get_random_observations(size=n_samples)
    X_train, Y_train = get_train_data(observations, bounds, hp_names, minimize)
    model = fit_model(X_train, Y_train, cat_dims=[])
    Y = sample(model, X_train)
    assert Y.shape == (1, n_samples, 3)

    X_train, Y_train = get_train_data(observations, bounds, hp_names, minimize, weights=torch.tensor([0.5, 0.3, 0.2]))
    model = fit_model(X_train, Y_train, cat_dims=[], scalarize=True)
    Y = sample(model, X_train)
    assert Y.shape == (1, n_samples, 1)


def test_fit_model() -> None:
    n_samples = N_SAMPLES
    observations, bounds, hp_names, minimize = get_random_observations(size=n_samples)
    X_train, Y_train = get_train_data(observations, bounds, hp_names, minimize)
    model = fit_model(X_train, Y_train, cat_dims=[])
    assert isinstance(model, ModelListGP)
    fit_model(X_train, Y_train, cat_dims=[], state_dict=model.state_dict())

    X_train, Y_train = get_train_data(observations, bounds, hp_names, minimize, weights=torch.tensor([0.5, 0.3, 0.2]))
    model = fit_model(X_train, Y_train, cat_dims=[], scalarize=True)
    assert isinstance(model, SingleTaskGP)
    fit_model(X_train, Y_train, cat_dims=[], scalarize=True, state_dict=model.state_dict())

    with pytest.raises(IndexError):
        fit_model(X_train, Y_train, cat_dims=[], scalarize=False)

    observations, bounds, hp_names, minimize = get_categorical_random_observations(size=n_samples)
    X_train, Y_train = get_train_data(observations, bounds, hp_names, minimize, weights=torch.tensor([0.5, 0.3, 0.2]))
    model = fit_model(X_train, Y_train, cat_dims=[2, 3], scalarize=True)
    assert isinstance(model, MixedSingleTaskGP)
    fit_model(X_train, Y_train, cat_dims=[2, 3], scalarize=True, state_dict=model.state_dict())


def test_get_model_and_train_data() -> None:
    n_samples = N_SAMPLES
    observations, bounds, hp_names, minimize = get_random_observations(size=n_samples)
    model, X_train, Y_train = get_model_and_train_data(observations, bounds, hp_names, minimize, cat_dims=[])
    assert isinstance(model, ModelListGP)

    model, X_train, Y_train = get_model_and_train_data(
        observations, bounds, hp_names, minimize, cat_dims=[], weights=torch.tensor([0.5, 0.3, 0.2])
    )
    assert isinstance(model, SingleTaskGP)


def test_get_acq_fn() -> None:
    n_samples = N_SAMPLES
    observations, bounds, hp_names, minimize = get_random_observations(size=n_samples)
    model, X_train, Y_train = get_model_and_train_data(observations, bounds, hp_names, minimize, cat_dims=[])
    acq_fn = get_acq_fn(model, X_train, Y_train, acq_fn_type="ehvi")
    assert isinstance(acq_fn, ExpectedHypervolumeImprovement)

    with pytest.raises(UnsupportedError):
        get_acq_fn(model, X_train, Y_train, acq_fn_type="parego")

    model, X_train, Y_train = get_model_and_train_data(
        observations, bounds, hp_names, minimize, cat_dims=[], weights=torch.tensor([0.5, 0.3, 0.2])
    )
    acq_fn = get_acq_fn(model, X_train, Y_train, acq_fn_type="parego")
    assert isinstance(acq_fn, ExpectedImprovement)

    with pytest.raises(RuntimeError):
        get_acq_fn(model, X_train, Y_train, acq_fn_type="ehvi")

    with pytest.raises(ValueError):
        acq_fn = get_acq_fn(model, X_train, Y_train, acq_fn_type="dummy")


def test_optimize_acq_fn() -> None:  # TODO
    n_samples = N_SAMPLES
    observations, bounds, hp_names, minimize = get_random_observations(size=n_samples)
    model, X_train, Y_train = get_model_and_train_data(observations, bounds, hp_names, minimize, cat_dims=[])
    acq_fn = get_acq_fn(model, X_train, Y_train, acq_fn_type="ehvi")
    eval_config = optimize_acq_fn(acq_fn, bounds, hp_names, fixed_features_list=None)
    assert len(eval_config.keys()) == len(hp_names)
    for key, value in eval_config.items():
        assert isinstance(value, (int, float))
        assert key in hp_names

    model, X_train, Y_train = get_model_and_train_data(
        observations, bounds, hp_names, minimize, cat_dims=[], weights=torch.tensor([0.5, 0.3, 0.2])
    )
    acq_fn = get_acq_fn(model, X_train, Y_train, acq_fn_type="parego")
    eval_config = optimize_acq_fn(acq_fn, bounds, hp_names, fixed_features_list=None)
    assert len(eval_config.keys()) == len(hp_names)
    for key, value in eval_config.items():
        assert isinstance(value, (int, float))
        assert key in hp_names

    observations, bounds, hp_names, minimize = get_categorical_random_observations(size=n_samples)
    model, X_train, Y_train = get_model_and_train_data(
        observations, bounds, hp_names, minimize, cat_dims=[2, 3], weights=torch.tensor([0.5, 0.3, 0.2])
    )
    categories = {"c0": ["a", "b"], "c1": ["A", "B", "C", "D", "E"]}
    assert isinstance(model, MixedSingleTaskGP)
    acq_fn = get_acq_fn(model, X_train, Y_train, acq_fn_type="parego")
    fixed_features_list = get_fixed_features_list(hp_names, cat_dims=[2, 3], categories=categories)
    eval_config = optimize_acq_fn(acq_fn, bounds, hp_names, fixed_features_list=fixed_features_list)
    assert len(eval_config.keys()) == len(hp_names)
    for key, value in eval_config.items():
        assert isinstance(value, (int, float))
        assert key in hp_names


def test_convert_categories_into_index() -> None:
    data_old = {
        "x": np.random.random(N_SAMPLES),
        "c0": np.random.randint(2, size=N_SAMPLES),
        "c1": np.random.randint(5, size=N_SAMPLES),
    }
    categories = {"c0": ["a", "b"], "c1": ["A", "B", "C", "D", "E"]}

    data_new = convert_categories_into_index({k: v.copy() for k, v in data_old.items()}, categories)
    assert np.allclose(data_old["x"], data_new["x"])
    assert np.allclose(data_old["c0"], data_new["c0"])
    assert np.allclose(data_old["c1"], data_new["c1"])

    data_old_str = {}
    data_old_str["x"] = data_old["x"].copy()
    data_old_str["c0"] = np.array([categories["c0"][v] for v in data_old["c0"]])
    data_old_str["c1"] = np.array([categories["c1"][v] for v in data_old["c1"]])
    data_new_str = convert_categories_into_index({k: v.copy() for k, v in data_old_str.items()}, categories)
    assert data_old_str["c0"].dtype == np.dtype("<U1")
    assert data_old_str["c1"].dtype == np.dtype("<U1")
    assert data_new_str["c0"].dtype == np.int64
    assert data_new_str["c1"].dtype == np.int64

    assert np.all(data_old["x"] == data_new_str["x"])
    assert np.all(data_old["c0"] == data_new_str["c0"])
    assert np.all(data_old["c1"] == data_new_str["c1"])

    data_old.pop("c0")
    data_old.pop("c1")
    data_new = convert_categories_into_index({k: v.copy() for k, v in data_old.items()}, categories=None)
    assert np.all(data_old["x"] == data_new["x"])

    data_old = {
        "x": np.random.random(N_SAMPLES),
        "c0": np.random.randint(2, size=N_SAMPLES),
        "c1": np.random.randint(10, size=N_SAMPLES),
    }
    categories = {"c0": ["a", "b"], "c1": ["A", "B", "C", "D", "E"]}
    with pytest.raises(ValueError):
        convert_categories_into_index(data_old, categories)


def test_get_fixed_features_list() -> None:
    hp_names = ["x0", "x1", "c0", "c1"]
    categories = {"c0": ["a", "b"], "c1": ["A", "B", "C", "D", "E"]}
    cat_dims = [2, 3]
    ffl = get_fixed_features_list(hp_names=hp_names, categories=categories, cat_dims=cat_dims)
    ans = [
        {2: 0.0, 3: 0.0},
        {2: 0.0, 3: 0.25},
        {2: 0.0, 3: 0.5},
        {2: 0.0, 3: 0.75},
        {2: 0.0, 3: 1.0},
        {2: 1.0, 3: 0.0},
        {2: 1.0, 3: 0.25},
        {2: 1.0, 3: 0.5},
        {2: 1.0, 3: 0.75},
        {2: 1.0, 3: 1.0},
    ]
    for e1, e2 in zip(ffl, ans):
        assert e1 == e2

    hp_names = ["x0", "x1"]
    categories = {}
    cat_dims = []
    ffl = get_fixed_features_list(hp_names=hp_names, categories=categories, cat_dims=cat_dims)
    assert ffl is None


if __name__ == "__main__":
    unittest.main()
