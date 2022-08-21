from typing import Any, Dict, List, Tuple

from meta_learn_bo.utils import HyperParameterType

import numpy as np


def get_toy_func_info() -> Dict[str, Any]:
    hp_info: Dict[str, HyperParameterType] = {
        "x0": HyperParameterType.Continuous,
        "x1": HyperParameterType.Continuous,
    }
    bounds: Dict[str, Tuple[float, float]] = {"x0": (-5, 5), "x1": (-5, 5)}
    minimize: Dict[str, bool] = {"f1": True, "f2": True}
    kwargs = dict(hp_info=hp_info, minimize=minimize, bounds=bounds)
    return kwargs


def get_categorical_toy_func_info() -> Dict[str, Any]:
    hp_info: Dict[str, HyperParameterType] = {
        "x0": HyperParameterType.Continuous,
        "x1": HyperParameterType.Continuous,
        "c": HyperParameterType.Categorical,
    }
    bounds: Dict[str, Tuple[float, float]] = {"x0": (-5, 5), "x1": (-5, 5), "c": (0, 1)}
    minimize: Dict[str, bool] = {"f1": True, "f2": True}
    categories: Dict[str, List[str]] = {"c": ["sin", "cos"]}
    kwargs = dict(hp_info=hp_info, minimize=minimize, bounds=bounds, categories=categories)
    return kwargs


def toy_func(eval_config: Dict[str, float]) -> Dict[str, float]:
    x, y = eval_config["x0"], eval_config["x1"]
    f1 = 4 * (x**2 + y**2)
    f2 = (x - 5) ** 2 + (y - 5) ** 2
    return {"f1": f1, "f2": f2}


def categorical_toy_func(eval_config: Dict[str, float]) -> Dict[str, float]:
    x, y, c = eval_config["x0"], eval_config["x1"], eval_config["c"]
    f1 = 4 * (x**2 + y**2)
    f2 = np.sin(x**2 + y**2) if c == "sin" else np.cos(x**2 + y**2)
    return {"f1": f1, "f2": f2}


def get_initial_samples(n_init: int) -> Dict[str, np.ndarray]:
    observations = {name: np.array([]) for name in ["x0", "x1", "f1", "f2"]}
    for _ in range(n_init):
        eval_config = {"x0": np.random.random() * 10 - 5, "x1": np.random.random() * 10 - 5}
        eval_config.update(toy_func(eval_config))
        for key, val in eval_config.items():
            observations[key] = np.append(observations[key], val)

    return observations


def get_initial_samples_for_categorical(n_init: int) -> Dict[str, np.ndarray]:
    observations = {name: np.array([]) for name in ["x0", "x1", "c", "f1", "f2"]}
    cats = ["sin", "cos"]
    for _ in range(n_init):
        eval_config = {
            "x0": np.random.random() * 10 - 5,
            "x1": np.random.random() * 10 - 5,
            "c": cats[np.random.randint(2)],
        }
        eval_config.update(categorical_toy_func(eval_config))
        for key, val in eval_config.items():
            observations[key] = np.append(observations[key], val)

    return observations
