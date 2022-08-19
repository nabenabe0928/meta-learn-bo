from typing import Any, Dict, List, Tuple

import numpy as np


def get_toy_func_info() -> Dict[str, Any]:
    hp_names: List[str] = ["x0", "x1"]
    bounds: Dict[str, Tuple[float, float]] = {"x0": (-5, 5), "x1": (-5, 5)}
    minimize: Dict[str, bool] = {"f1": True, "f2": True}
    kwargs = dict(hp_names=hp_names, minimize=minimize, bounds=bounds)
    return kwargs


def toy_func(eval_config: Dict[str, float]) -> Dict[str, float]:
    x, y = eval_config["x0"], eval_config["x1"]
    f1 = 4 * (x**2 + y**2)
    f2 = (x - 5) ** 2 + (y - 5) ** 2
    return {"f1": f1, "f2": f2}


def get_initial_samples(n_init: int) -> Dict[str, np.ndarray]:
    observations = {name: np.array([]) for name in ["x0", "x1", "f1", "f2"]}
    for _ in range(n_init):
        eval_config = {
            "x0": np.random.random() * 10 - 5,
            "x1": np.random.random() * 10 - 5
        }
        eval_config.update(toy_func(eval_config))
        for key, val in eval_config.items():
            observations[key] = np.append(observations[key], val)

    return observations
