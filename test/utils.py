from typing import Any, Dict, List, Tuple

from meta_learn_bo.utils import NumericType

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
    hp_names: List[str] = ["x0", "x1"]
    bounds: Dict[str, Tuple[float, float]] = {"x0": (-5, 5), "x1": (-5, 5)}
    minimize: Dict[str, bool] = {"f1": True, "f2": True}
    kwargs = dict(hp_names=hp_names, minimize=minimize, bounds=bounds)

    observations = initial_sample(size=size, **kwargs)
    return kwargs, observations
