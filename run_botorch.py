from typing import Dict, List, Tuple
import warnings

import numpy as np

import torch

from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.analytic import MultiObjectiveAnalyticAcquisitionFunction

from botorch_utils import (
    get_acq_fn,
    get_model_and_train_data,
    optimize_acq_fn,
)


warnings.filterwarnings("ignore")


class TransferMultiObjectiveAcquisitionFunction(MultiObjectiveAnalyticAcquisitionFunction):
    def __init__(self, acq_fn_list: List[ExpectedHypervolumeImprovement], weights: torch.Tensor):
        assert torch.isclose(weights.sum(), torch.tensor(1.0))
        self._acq_fn_list = acq_fn_list
        self._weights = weights

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = torch.tensor([weight * acq_fn(X) for acq_fn, weight in zip(self._acq_fn_list, self._weights)])
        return out.sum(axis=0)


def update_observations(
    observations: Dict[str, np.ndarray],
    eval_config: Dict[str, float],
    results: Dict[str, float],
) -> None:
    for hp_name, val in eval_config.items():
        observations[hp_name] = np.append(observations[hp_name], val)

    for obj_name, val in results.items():
        observations[obj_name] = np.append(observations[obj_name], val)


def initial_sample(
    n_init: int,
    bounds: Dict[str, Tuple[float, float]],
    hp_names: List[str],
    minimize: Dict[str, bool],
) -> Dict[str, np.ndarray]:

    obj_names = list(minimize.keys())
    observations: Dict[str, np.ndarray] = {name: np.array([]) for name in hp_names + obj_names}
    for i in range(n_init):
        eval_config = {}
        for hp_name in hp_names:
            lb, ub = bounds[hp_name]
            eval_config[hp_name] = np.random.random() * (ub - lb) + lb

        results = obj_func(eval_config)
        update_observations(observations=observations, eval_config=eval_config, results=results)

    return observations


def obj_func(eval_config: Dict[str, float]) -> Dict[str, float]:
    x, y = eval_config["x0"], eval_config["x1"]
    f1 = 4 * (x**2 + y**2)
    f2 = (x - 5) ** 2 + (y - 5) ** 2
    return {"f1": f1, "f2": f2}


def optimize(method: str = "parego"):
    hp_names: List[str] = ["x0", "x1"]
    bounds: Dict[str, Tuple[float, float]] = {"x0": (-5, 5), "x1": (-5, 5)}
    minimize: Dict[str, bool] = {"f1": True, "f2": True}
    kwargs = dict(hp_names=hp_names, minimize=minimize, bounds=bounds)
    observations = initial_sample(n_init=10, **kwargs)

    for t in range(100):
        weights = None
        if method == "parego":
            weights = torch.rand(2)
            weights /= torch.sum(weights)

        model, X_train, Y_train = get_model_and_train_data(observations=observations, weights=weights, **kwargs)
        acq_fn = get_acq_fn(model=model, X_train=X_train, Y_train=Y_train, method=method)

        eval_config = optimize_acq_fn(acq_fn=acq_fn, bounds=bounds, hp_names=hp_names)
        results = obj_func(eval_config)
        print(f"Iteration {t + 1}: ", eval_config, results)
        update_observations(observations=observations, eval_config=eval_config, results=results)

    print(observations)


if __name__ == "__main__":
    optimize(method="ehvi")
