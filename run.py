from typing import Any, Dict, List, Optional, Tuple

import ConfigSpace as CS

import numpy as np

from smac.facade.smac_bb_facade import SMAC4BB
from smac.multi_objective.parego import ParEGO
from smac.optimizer.acquisition.maximizer import FixedSet
from smac.scenario.scenario import Scenario

# from meta_learn_bo.rgpe import RankingWeigtedGaussianProcessEnsemble
from meta_learn_bo.tstr import TwoStageTransferWithRanking
from meta_learn_bo.taf import TransferAcquisitionFunc


def acq_func_optimizer(
    option: Optional[str] = None,
) -> Tuple[Optional[FixedSet], Optional[Dict[str, List[Any]]]]:
    if option is None:
        return None, {}
    elif option == "fixed-set":
        return FixedSet, {"configuration": None}
    else:
        raise ValueError(f"option must be either None or fixed-set, but got {option}")


def sphere(config: CS.Configuration, shift: int = 0) -> float:
    X = config.get_array()
    return np.sum((X - shift) ** 2)


def multi_sphere(config: CS.Configuration) -> Tuple[float, float]:
    X = config.get_array()
    return {"f1": np.sum((X - 1) ** 2), "f2": np.sum((X + 1) ** 2)}


def get_metadata(shifts: List[int]) -> Dict[str, Dict[str, np.ndarray]]:
    metadata: Dict[str, Dict[str, np.ndarray]] = {}
    for shift in shifts:
        key = f"shift={shift}"
        metadata[key] = {}
        X = np.random.random((50, 2))
        Y = np.sum((X - shift) ** 2, axis=-1)
        metadata[key]["x0"] = X[:, 0]
        metadata[key]["x1"] = X[:, 1]
        metadata[key]["loss"] = Y

    return metadata


def main(max_evals: int, config_space: CS.ConfigurationSpace, seed: int):
    scenario = Scenario(
        dict(
            run_obj="quality",
            runcount_limit=max_evals,
            cs=config_space,
            output_dir=None,
            multi_objectives=["f1", "f2"],
        )
    )
    model_kwargs = dict(
        # metadata=get_metadata(shifts=[1, 2]),
        metadata={},
        # n_samples=1000,
        max_evals=max_evals,
        metric_name="loss",
    )
    optimizer = SMAC4BB(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        # model=RankingWeigtedGaussianProcessEnsemble,
        model=TwoStageTransferWithRanking,
        model_kwargs=model_kwargs,
        # tae_runner=lambda config: sphere(config, shift=0),
        tae_runner=multi_sphere,
        initial_design=None,
        initial_design_kwargs={},
        initial_configurations=None,
        acquisition_function=TransferAcquisitionFunc,
        acquisition_function_optimizer=None,
        acquisition_function_optimizer_kwargs={},
        multi_objective_algorithm=ParEGO,
    )

    # Disable initial random sampling
    optimizer.solver.epm_chooser.random_configuration_chooser = None
    optimizer.optimize()


if __name__ == "__main__":
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x0", -5, 5))
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x1", -5, 5))
    main(max_evals=10, config_space=config_space, seed=0)
