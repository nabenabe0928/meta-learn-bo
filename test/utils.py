from typing import Any, Dict, List, Tuple

from meta_learn_bo.utils import HyperParameterType, NumericType

import numpy as np


def update_observations(
    observations: Dict[str, np.ndarray],
    eval_config: Dict[str, float],
    results: Dict[str, float],
) -> None:
    for hp_name, val in eval_config.items():
        observations[hp_name] = np.append(observations[hp_name], val)

    for obj_name, val in results.items():
        observations[obj_name] = np.append(observations[obj_name], val)


def obj_func(eval_config: Dict[str, float]) -> Dict[str, float]:
    x, y = eval_config["x0"], eval_config["x1"]
    f1 = 4 * (x**2 + y**2)
    f2 = (x - 5) ** 2 + (y - 5) ** 2
    return {"f1": f1, "f2": f2}


def obj_func_for_categorical(eval_config: Dict[str, float]) -> Dict[str, float]:
    assert eval_config["c0"] in ["a", "b"]
    assert eval_config["c1"] in ["A", "B", "C", "D", "E"]
    assert eval_config["i0"] in [-2, -1, 0, 1, 2]
    assert eval_config["i1"] in [0, 1, 2, 3]

    x, y = eval_config["x0"], eval_config["x1"]
    f1 = 4 * (x**2 + y**2)
    f2 = (x - 5) ** 2 + (y - 5) ** 2
    f3 = x + y
    return {"f1": f1, "f2": f2, "f3": f3}


def initial_sample(
    size: int,
    hp_names: List[str],
    bounds: Dict[str, Tuple[NumericType, NumericType]],
    minimize: Dict[str, bool],
) -> Dict[str, np.ndarray]:
    observations = {
        hp_name: np.random.random(size) * (bounds[hp_name][1] - bounds[hp_name][0]) + bounds[hp_name][0]
        for hp_name in hp_names
    }
    observations.update({obj_name: np.array([]) for obj_name in ["f1", "f2"]})
    for i in range(size):
        eval_config = {hp_name: observations[hp_name][i] for hp_name in hp_names}
        results = obj_func(eval_config)
        for k, v in results.items():
            observations[k] = np.append(observations[k], v)

    return observations


def get_kwargs_and_observations(size: int) -> Dict[str, Any]:
    hp_info: Dict[str, HyperParameterType] = {"x0": HyperParameterType.Continuous, "x1": HyperParameterType.Continuous}
    bounds: Dict[str, Tuple[float, float]] = {"x0": (-5, 5), "x1": (-5, 5)}
    minimize: Dict[str, bool] = {"f1": True, "f2": True}
    kwargs = dict(hp_info=hp_info, minimize=minimize, bounds=bounds)

    observations = initial_sample(size=size, minimize=minimize, bounds=bounds, hp_names=list(hp_info.keys()))
    return kwargs, observations


def get_kwargs_and_observations_for_categorical(size: int) -> Dict[str, Any]:
    hp_info: Dict[str, HyperParameterType] = {
        "x0": HyperParameterType.Continuous,
        "x1": HyperParameterType.Continuous,
        "i0": HyperParameterType.Integer,
        "i1": HyperParameterType.Integer,
        "c0": HyperParameterType.Categorical,
        "c1": HyperParameterType.Categorical,
    }
    hp_names = ["x0", "x1", "i0", "i1", "c0", "c1"]
    bounds = {
        hp_names[0]: (-5.0, 5.0),
        hp_names[1]: (5.0, 25.0),
        hp_names[2]: (-2, 2),
        hp_names[3]: (0, 3),
        hp_names[4]: (0, 1),
        hp_names[5]: (0, 4),
    }
    obj_names = ["f1", "f2", "f3"]
    minimize = {obj_name: bool((i + 1) % 2) for i, obj_name in enumerate(obj_names)}
    kwargs = dict(hp_info=hp_info, minimize=minimize, bounds=bounds)

    observations = {
        hp_name: np.random.random(size) * (bounds[hp_name][1] - bounds[hp_name][0]) + bounds[hp_name][0]
        for hp_name in hp_names[:2]
    }
    observations.update(
        {
            hp_name: np.random.randint(bounds[hp_name][-1] - bounds[hp_name][0] + 1, size=size) + bounds[hp_name][0]
            for hp_name in hp_names[2:4]
        }
    )
    observations.update({hp_name: np.random.randint(bounds[hp_name][-1] + 1, size=size) for hp_name in hp_names[4:]})
    observations.update({obj_name: np.random.random(size) * 2 - 1 for obj_name in obj_names})

    return kwargs, observations
