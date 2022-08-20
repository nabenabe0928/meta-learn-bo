import unittest
from typing import Dict, List, Tuple

import pytest

from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP

from meta_learn_bo.utils import (
    NumericType,
    denormalize,
    fit_model,
    get_acq_fn,
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


def test_optimize_acq_fn() -> None:
    n_samples = N_SAMPLES
    observations, bounds, hp_names, minimize = get_random_observations(size=n_samples)
    model, X_train, Y_train = get_model_and_train_data(observations, bounds, hp_names, minimize, cat_dims=[])
    acq_fn = get_acq_fn(model, X_train, Y_train, acq_fn_type="ehvi")
    eval_config = optimize_acq_fn(acq_fn, bounds, hp_names)
    assert len(eval_config.keys()) == len(hp_names)
    for key, value in eval_config.items():
        assert isinstance(value, (int, float))
        assert key in hp_names

    model, X_train, Y_train = get_model_and_train_data(
        observations, bounds, hp_names, minimize, cat_dims=[], weights=torch.tensor([0.5, 0.3, 0.2])
    )
    acq_fn = get_acq_fn(model, X_train, Y_train, acq_fn_type="parego")
    eval_config = optimize_acq_fn(acq_fn, bounds, hp_names)
    assert len(eval_config.keys()) == len(hp_names)
    for key, value in eval_config.items():
        assert isinstance(value, (int, float))
        assert key in hp_names


if __name__ == "__main__":
    unittest.main()
