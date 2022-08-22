from typing import Callable, Dict, List, Optional, Tuple, Union

from meta_learn_bo.samplers.base_sampler import BaseSampler
from meta_learn_bo.utils import HyperParameterType, NumericType, update_observations, validate_config_and_results

import numpy as np


def convert_val(
    val: NumericType,
    hp_type: HyperParameterType,
    choices: Optional[List[str]],
) -> Union[str, NumericType]:
    if hp_type == int:
        return int(np.round(val))
    elif hp_type == float:
        return val
    else:
        assert choices is not None  # mypy redefinition
        idx = int(val)
        return choices[idx]


class RandomSampler(BaseSampler):
    def __init__(
        self,
        bounds: Dict[str, Tuple[NumericType, NumericType]],
        hp_info: Dict[str, HyperParameterType],
        minimize: Dict[str, bool],
        max_evals: int,
        obj_func: Callable,
        verbose: bool = True,
        categories: Optional[Dict[str, List[str]]] = None,
        seed: Optional[int] = None,
    ):
        """Random sampler.

        Args:
            bounds (Dict[str, Tuple[NumericType, NumericType]]):
                The lower and upper bounds for each hyperparameter.
                If the parameter is categorical, it must be [0, the number of categories - 1].
                Dict[hp_name, Tuple[lower bound, upper bound]].
            hp_info (Dict[str, HyperParameterType]):
                The type information of each hyperparameter.
                Dict[hp_name, HyperParameterType].
            minimize (Dict[str, bool]):
                The direction of the optimization for each objective.
                Dict[obj_name, whether to minimize or not].
            obj_func (Callable):
                The objective function that takes `eval_config` and returns `results`.
            max_evals (int):
                How many hyperparameter configurations to evaluate during the optimization.
            categories (Optional[Dict[str, List[str]]]):
                Categories for each categorical parameter.
                Dict[categorical hp name, List[each category name]].
            verbose (bool):
                Whether to print the results at each iteration.
            seed (Optional[int]):
                The random seed.
        """
        super().__init__(
            bounds=bounds,
            hp_info=hp_info,
            minimize=minimize,
            max_evals=max_evals,
            obj_func=obj_func,
            verbose=verbose,
            categories=categories,
            seed=seed,
        )

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        return {
            hp_name: val.copy()
            if hp_name not in self._categories
            else np.asarray([self._categories[hp_name][idx] for idx in val])
            for hp_name, val in self._observations.items()
        }

    def sample(self) -> Dict[str, Union[str, NumericType]]:
        """
        Sample the next configuration according to the random sampler.

        Returns:
            eval_config (Dict[str, Union[str, NumericType]]):
                The hyperparameter configuration that were evaluated.
        """
        eval_config: Dict[str, Union[str, NumericType]] = {}
        for hp_name in self._hp_names:
            lb, ub = self._bounds[hp_name]
            val = self._rng.random() * (ub - lb) + lb
            choices = self._categories.get(hp_name, None)
            eval_config[hp_name] = convert_val(val=val, hp_type=self._hp_info[hp_name], choices=choices)

        return eval_config

    def update(self, eval_config: Dict[str, Union[str, NumericType]], results: Dict[str, float]) -> None:
        """
        Update the target observations.

        Args:
            eval_config (Dict[str, Union[str, NumericType]]):
                The hyperparameter configuration that were evaluated.
            results (Dict[str, float]):
                The results obtained from the evaluation of eval_config.
        """
        validate_config_and_results(
            eval_config=eval_config,
            results=results,
            hp_names=self._hp_names,
            obj_names=self._obj_names,
            hp_info=self._hp_info,
            bounds=self._bounds,
            categories=self._categories,
        )
        update_observations(
            observations=self._observations,
            eval_config=eval_config,
            results=results,
            hp_info=self._hp_info,
            categories=self._categories,
        )
