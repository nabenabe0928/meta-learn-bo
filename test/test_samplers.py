from typing import Any, Dict, List, Tuple

from meta_learn_bo.models.rgpe import RankingWeightedGaussianProcessEnsemble
from meta_learn_bo.samplers.bo_sampler import MetaLearnGPSampler
from meta_learn_bo.samplers.random_sampler import RandomSampler, convert_val
from meta_learn_bo.utils import HyperParameterType, NumericType

import numpy as np


def func(eval_config: Dict[str, float], shift: int = 0) -> Dict[str, float]:
    assert "c" in eval_config
    x, y = eval_config["x"], eval_config["y"]
    f1 = (x + shift) ** 2 + (y + shift) ** 2
    f2 = (x - 2 + shift) ** 2 + (y - 2 + shift) ** 2
    return {"f1": f1, "f2": f2}


def get_sampler_kwargs() -> Dict[str, Any]:
    bounds = {"x": (-5, 5), "y": (-5, 5), "c": (0, 1)}
    hp_info = {
        "x": HyperParameterType.Continuous,
        "y": HyperParameterType.Continuous,
        "c": HyperParameterType.Categorical,
    }
    minimize = {"f1": True, "f2": True}
    categories = {"c": ["a", "b"]}
    kwargs = dict(minimize=minimize, bounds=bounds, hp_info=hp_info, categories=categories)
    return kwargs


def get_random_data(
    bounds: Dict[str, Tuple[NumericType, NumericType]],
    hp_info: Dict[str, HyperParameterType],
    minimize: Dict[str, bool],
    categories: Dict[str, List[str]],
    n_samples: int,
    shift: int,
) -> Dict[str, np.ndarray]:
    sampler = RandomSampler(
        obj_func=lambda eval_config: func(eval_config, shift=shift),
        max_evals=n_samples,
        bounds=bounds,
        hp_info=hp_info,
        minimize=minimize,
        categories=categories,
    )
    sampler.optimize()
    return sampler.observations


def test_convert_val() -> None:
    choices = None
    hp_type = HyperParameterType.Continuous
    val = convert_val(val=0.5, hp_type=hp_type, choices=choices)
    assert val == 0.5

    hp_type = HyperParameterType.Integer
    val = convert_val(val=0.4, hp_type=hp_type, choices=choices)
    assert val == 0

    val = convert_val(val=0.6, hp_type=hp_type, choices=choices)
    assert val == 1

    val = convert_val(val=-0.4, hp_type=hp_type, choices=choices)
    assert val == 0

    val = convert_val(val=-0.6, hp_type=hp_type, choices=choices)
    assert val == -1

    hp_type = HyperParameterType.Categorical
    choices = ["a", "b", "c"]
    for add in [0, 0.4, 0.6]:
        for idx in range(3):
            val = convert_val(val=idx + add, hp_type=hp_type, choices=choices)
            assert val == choices[idx]


def test_optimize_sampler() -> None:
    kwargs = get_sampler_kwargs()
    metadata = {"src": get_random_data(n_samples=30, shift=2, **kwargs)}
    init_data = get_random_data(n_samples=10, shift=0, **kwargs)

    rgpe = RankingWeightedGaussianProcessEnsemble(init_data=init_data, metadata=metadata, **kwargs)
    sampler = MetaLearnGPSampler(max_evals=5, obj_func=func, model=rgpe, **kwargs)
    sampler.optimize()

    data = sampler.observations
    for hp_name in sampler._hp_names:
        assert data[hp_name].size == 15

    for obj_name in sampler._obj_names:
        assert data[obj_name].size == 15
