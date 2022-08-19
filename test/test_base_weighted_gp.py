import unittest

import pytest

from meta_learn_bo.rgpe import RankingWeigtedGaussianProcessEnsemble
from meta_learn_bo.taf import TransferAcquisitionFunction
from meta_learn_bo.utils import get_train_data, optimize_acq_fn

import torch

from utils import get_kwargs_and_observations, obj_func


def test_validate_input_and_properties() -> None:
    for acq_fn_type in ["parego", "ehvi"]:
        kwargs, observations = get_kwargs_and_observations(size=5)
        kwargs.update(n_bootstraps=50, acq_fn_type=acq_fn_type)
        metadata = {}
        _, metadata["src20"] = get_kwargs_and_observations(size=10)
        _, metadata["src30"] = get_kwargs_and_observations(size=15)
        rgpe = RankingWeigtedGaussianProcessEnsemble(init_data=observations, metadata=metadata, **kwargs)

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


def test_update():
    n_init = 5
    for acq_fn_type in ["parego", "ehvi"]:
        kwargs, observations = get_kwargs_and_observations(size=n_init)
        kwargs_for_proc = kwargs.copy()
        X_train, _ = get_train_data(observations, **kwargs_for_proc)
        kwargs.update(n_bootstraps=50, acq_fn_type=acq_fn_type)
        metadata = {}
        _, metadata["src20"] = get_kwargs_and_observations(size=10)
        _, metadata["src30"] = get_kwargs_and_observations(size=15)
        rgpe = RankingWeigtedGaussianProcessEnsemble(init_data=observations, metadata=metadata, **kwargs)

        with torch.no_grad():
            model = rgpe._base_models[rgpe._task_names[0]]
            pred_old = model.posterior(X_train).mean

        eval_config = optimize_acq_fn(acq_fn=rgpe.acq_fn, bounds=kwargs["bounds"], hp_names=kwargs["hp_names"])
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


if __name__ == "__main__":
    unittest.main()
