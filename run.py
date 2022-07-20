from typing import Any, Dict, List, Optional, Tuple

import ConfigSpace as CS

import numpy as np

from smac.facade.smac_bb_facade import SMAC4BB
from smac.optimizer.acquisition.maximizer import FixedSet
from smac.scenario.scenario import Scenario

from src.rgpe import RGPE
from src.taf import TAF


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


def main(
    max_evals: int, config_space: CS.ConfigurationSpace, seed: int
):
    scenario = Scenario(dict(
        run_obj="quality",
        runcount_limit=max_evals,
        cs=config_space,
        output_dir=None,
    ))
    model_kwargs = dict(
        metadata={},
        n_samples=1000,
        max_evals=max_evals,
        metric_name="loss",
    )
    optimizer = SMAC4BB(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        model=RGPE,
        model_kwargs=model_kwargs,
        tae_runner=lambda config: sphere(config, shift=0),
        initial_design=None,
        initial_design_kwargs={},
        initial_configurations=None,
        acquisition_function=TAF,
        acquisition_function_optimizer=None,
        acquisition_function_optimizer_kwargs={},
    )

    # Disable initial random sampling
    optimizer.solver.epm_chooser.random_configuration_chooser = None
    optimizer.optimize()


if __name__ == "__main__":
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x0", -5, 5))
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x1", -5, 5))
    main(max_evals=10, config_space=config_space, seed=0)
