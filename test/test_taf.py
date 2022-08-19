import unittest
from typing import Dict, List, Tuple

import pytest

from meta_learn_bo.taf import TransferAcquisitionFunction
from meta_learn_bo.utils import NumericType, get_acq_fn, get_model_and_train_data

import numpy as np

import torch


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


def test_validate_input() -> None:
    weights = torch.tensor([0.5, 0.3, 0.2])
    parego_weights = torch.tensor([0.5, 0.3, 0.2])
    acq_fn_list = []
    acq_fn_type = "parego"
    for i in range(1, 4):
        observations, bounds, hp_names, minimize = get_random_observations(size=10 * i)
        model, X_train, Y_train = get_model_and_train_data(observations, bounds, hp_names, minimize, parego_weights)
        acq_fn = get_acq_fn(model, X_train, Y_train, acq_fn_type=acq_fn_type)
        acq_fn_list.append(acq_fn)
        parego_weights, acq_fn_type = None, "ehvi"

    with pytest.raises(TypeError):
        TransferAcquisitionFunction(acq_fn_list=acq_fn_list, weights=weights)


def test_forward_in_taf() -> None:
    weights = torch.tensor([0.5, 0.3, 0.2])
    X = torch.rand((10, 1, 2))
    for acq_fn_type in ["ehvi", "parego"]:
        acq_fn_list = []
        parego_weights = torch.tensor([0.5, 0.3, 0.2]) if acq_fn_type == "parego" else None
        for i in range(1, 4):
            observations, bounds, hp_names, minimize = get_random_observations(size=10 * i)
            model, X_train, Y_train = get_model_and_train_data(
                observations, bounds, hp_names, minimize, weights=parego_weights
            )
            acq_fn = get_acq_fn(model, X_train, Y_train, acq_fn_type=acq_fn_type)
            acq_fn_list.append(acq_fn)
        else:
            taf = TransferAcquisitionFunction(acq_fn_list=acq_fn_list, weights=weights)
            results = taf(X)
            for w, acq_fn in zip(weights, acq_fn_list):
                results -= w * acq_fn(X)

            assert torch.allclose(results, torch.zeros_like(results))


def test_skip_small_weights_in_taf() -> None:
    weights = torch.tensor([0.5, 0.5 - 1e-8, 1e-8])
    X = torch.rand((10, 1, 2))
    acq_fn_list = []
    for i in range(1, 4):
        observations, bounds, hp_names, minimize = get_random_observations(size=10 * i)
        model, X_train, Y_train = get_model_and_train_data(observations, bounds, hp_names, minimize)
        acq_fn = get_acq_fn(model, X_train, Y_train, acq_fn_type="ehvi")
        acq_fn_list.append(acq_fn)
    else:
        taf = TransferAcquisitionFunction(acq_fn_list=acq_fn_list, weights=weights)
        results = taf(X)
        for w, acq_fn in zip(weights, acq_fn_list):
            results -= w * acq_fn(X)

        # not skiped, but still almost equal!
        assert torch.allclose(results, torch.zeros_like(results))

        results = taf(X)
        for w, acq_fn in zip(weights[:-1], acq_fn_list[:-1]):
            results -= w * acq_fn(X)

        # skiped and equal!
        assert torch.allclose(results, torch.zeros_like(results))


if __name__ == "__main__":
    unittest.main()
