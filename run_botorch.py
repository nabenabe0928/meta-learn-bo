from typing import Dict, List, Tuple
import warnings

import numpy as np

from meta_learn_bo.rgpe import RankingWeigtedGaussianProcessEnsemble
from meta_learn_bo.tstr import TwoStageTransferWithRanking
from meta_learn_bo.utils import optimize_acq_fn


warnings.filterwarnings("ignore")


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

    n_init, max_evals = 10, 90
    observations = initial_sample(n_init=n_init, **kwargs)
    bo_method = [
        RankingWeigtedGaussianProcessEnsemble,
        TwoStageTransferWithRanking,
    ][1]
    rgpe = bo_method(
        init_data=observations,
        metadata={"src": initial_sample(n_init=50, **kwargs)},
        # n_samples=100,
        max_evals=max_evals,
        hp_names=hp_names,
        method=method,
        minimize=minimize,
        bounds=bounds,
        target_task_name="target",
    )

    for t in range(max_evals - n_init):
        eval_config = optimize_acq_fn(acq_fn=rgpe.acq_fn, bounds=bounds, hp_names=hp_names)
        results = obj_func(eval_config)
        rgpe.update(eval_config=eval_config, results=results)
        print(f"Iteration {t + 1}: ", eval_config, results)

    print(rgpe.observations)


if __name__ == "__main__":
    optimize(method="ehvi")
    # optimize(method="parego")
