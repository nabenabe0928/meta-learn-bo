import itertools
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP

from meta_learn_bo.taf import TransferAcquisitionFunction
from meta_learn_bo.utils import (
    AcqFuncType,
    HyperParameterType,
    NumericType,
    PAREGO,
    get_acq_fn,
    get_model_and_train_data,
    get_train_data,
    optimize_acq_fn,
)

import numpy as np

import torch


def get_fixed_features_list(
    hp_names: List[str], cat_dims: List[int], categories: Dict[str, List[str]]
) -> Optional[List[Dict[int, float]]]:
    """
    Returns:
        fixed_features_list (Optional[List[Dict[int, float]]]):
            A list of maps `{feature_index: value}`.
            The i-th item represents the fixed_feature for the i-th optimization.
            Basically, we would like to perform len(fixed_features_list) times of
            optimizations and we use each `fixed_features` in each optimization.

    NOTE:
        Due to the normalization, we need to define each parameter to be in [0, 1].
        For this reason, when we have K categories, the possible choices will be
        [0, 1/(K-1), 2/(K-1), ..., (K-1)/(K-1)].
    """
    if len(cat_dims) == 0:
        return None

    fixed_features_list: List[Dict[int, float]] = []
    for feats in itertools.product(
        *(np.linspace(0, len(categories[hp_names[d]]) - 1, len(categories[hp_names[d]])) for d in cat_dims)
    ):
        fixed_features_list.append({d: val for val, d in zip(feats, cat_dims)})

    return fixed_features_list


class BaseWeightedGP(metaclass=ABCMeta):
    def __init__(
        self,
        init_data: Dict[str, np.ndarray],
        metadata: Dict[str, Dict[str, np.ndarray]],
        bounds: Dict[str, Tuple[NumericType, NumericType]],
        hp_info: Dict[str, HyperParameterType],
        minimize: Dict[str, bool],
        acq_fn_type: AcqFuncType,
        target_task_name: str,
        max_evals: int,
        categories: Optional[Dict[str, List[str]]],
        seed: Optional[int],
    ):
        """The base class for the weighted combination of
        Gaussian process based acquisition functions.

        Args:
            init_data (Dict[str, np.ndarray]):
                The observations of the target task
                sampled from the random sampling.
                Dict[hp_name/obj_name, the array of the corresponding param].
            metadata (Dict[str, Dict[str, np.ndarray]]):
                The observations of the tasks to transfer.
                Dict[task_name, Dict[hp_name/obj_name, the array of the corresponding param]].
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
            acq_fn_type (Literal[PAREGO, EHVI]):
                The acquisition function type.
            target_task_name (str):
                The name of the target task.
            max_evals (int):
                How many hyperparameter configurations to evaluate during the optimization.
            categories (Optional[Dict[str, List[str]]]):
                Categories for each categorical parameter.
                Dict[categorical hp name, List[each category name]].
            seed (Optional[int]):
                The random seed.

        NOTE:
            This implementation is exclusively for multi-objective optimization settings.
        """
        self._base_models: Dict[str, Union[SingleTaskGP, ModelListGP]] = {}
        self._acq_fns: Dict[str, Union[ExpectedImprovement, ExpectedHypervolumeImprovement]] = {}
        self._acq_fn: TransferAcquisitionFunction
        self._metadata: Dict[str, Dict[str, np.ndarray]] = metadata
        self._task_names = list(metadata.keys()) + [target_task_name]
        self._target_task_name = target_task_name
        self._n_tasks = len(self._task_names)
        self._bounds = bounds
        self._max_evals = max_evals
        self._hp_info = hp_info
        self._hp_names = list(hp_info.keys())
        # cat_dim specifies which dimensions are categorical parameters
        self._cat_dims = [idx for idx, hp_name in enumerate(self._hp_names) if hp_info[hp_name] == str]
        self._minimize = minimize
        self._obj_names = list(minimize.keys())
        self._observations: Dict[str, np.ndarray] = init_data
        self._acq_fn_type = acq_fn_type
        self._rng = np.random.RandomState(seed)
        self._task_weights: torch.Tensor
        self._categories: Dict[str, List[str]] = categories if categories is not None else {}
        self._fixed_features_list = get_fixed_features_list(
            hp_names=self._hp_names, cat_dims=self._cat_dims, categories=self._categories
        )

        self._validate_input()
        self._train()

    @property
    def acq_fn(self) -> TransferAcquisitionFunction:
        return self._acq_fn

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        return {
            hp_name: val.copy()
            if hp_name not in self._categories
            else np.asarray([self._categories[hp_name][idx] for idx in val])
            for hp_name, val in self._observations.items()
        }

    @property
    def fixed_features_list(self) -> Optional[List[Dict[int, float]]]:
        """
        Returns:
            fixed_features_list (Optional[List[Dict[int, float]]]):
                A list of maps `{feature_index: value}`.
                The i-th item represents the fixed_feature for the i-th optimization.
                Basically, we would like to perform len(fixed_features_list) times of
                optimizations and we use each `fixed_features` in each optimization.

        NOTE:
            Due to the normalization, we need to define each parameter to be in [0, 1].
            For this reason, when we have K categories, the possible choices will be
            [0, 1/(K-1), 2/(K-1), ..., (K-1)/(K-1)].
        """
        return self._fixed_features_list

    def _validate_input(self) -> None:
        if len(set(self._task_names)) != self._n_tasks:
            raise ValueError(f"task_names must be different from each other, but got {self._task_names}")

        if not all(hp_name in self._hp_names for hp_name in self._bounds.keys()):
            raise ValueError(
                "bounds must have the bounds for all hyperparameters. "
                f"Expected {self._hp_names}, but got {list(self._bounds.keys())}"
            )

        if not all(name in self._observations for name in self._hp_names + self._obj_names):
            raise ValueError(
                "observations must have the data for all hyperparameters and objectives. "
                f"Expected {self._hp_names + self._obj_names}, but got {list(self._observations.keys())}"
            )

        if len(self._cat_dims) > 0:
            cat_hp_names = [self._hp_names[d] for d in self._cat_dims]
            if self._categories is None or not all(self._hp_names[d] in self._categories for d in self._cat_dims):
                raise ValueError(
                    f"categories must include the categories for {cat_hp_names}, but got {self._categories}"
                )
            for hp_name in cat_hp_names:
                n_cats = len(self._categories[hp_name])
                if self._bounds[hp_name] != (0, n_cats - 1):
                    raise ValueError(
                        f"The categorical parameter `{hp_name}` has {n_cats} categories and expects "
                        f"the bound to be (0, n_cats - 1)=(0, {n_cats - 1}), but got {self._bounds[hp_name]}"
                    )

        for task_name in self._task_names[:-1]:
            observations = self._metadata[task_name]
            if not all(name in observations for name in self._hp_names + self._obj_names):
                raise ValueError(
                    f"metadata for {task_name} must have the data for all hyperparameters and objectives. "
                    f"Expected {self._hp_names + self._obj_names}, but got {list(observations.keys())}"
                )

    def _task_weights_repr(self) -> str:
        ws = ", ".join([f"{name}: {float(w):.3f}" for name, w in zip(self._task_names, self._task_weights)])
        return f"task weights = ({ws})"

    def optimize_acq_fn(self) -> Dict[str, Union[str, NumericType]]:
        raw_config = optimize_acq_fn(
            acq_fn=self.acq_fn,
            bounds=self._bounds,
            hp_names=self._hp_names,
            fixed_features_list=self._fixed_features_list,
        )
        eval_config: Dict[str, Union[str, NumericType]] = {}
        for hp_name, val in raw_config.items():
            type_ = self._hp_info[hp_name].value
            if type_ == float:
                eval_config[hp_name] = val
            elif type_ == int:
                eval_config[hp_name] = int(val + 0.5)
            else:
                cat_idx = int(val + 0.5)
                eval_config[hp_name] = self._categories[hp_name][cat_idx]

        return eval_config

    def update(self, eval_config: Dict[str, Union[str, NumericType]], results: Dict[str, float]) -> None:
        """
        Update the target observations, (a) Gaussian process model(s),
        and its/their acquisition function(s).
        If the acq_fn_type is ParEGO, we need to re-train each Gaussian process models
        and the corresponding acquisition functions.

        Args:
            eval_config (Dict[str, Union[str, NumericType]]):
                The hyperparameter configuration that were evaluated.
            results (Dict[str, float]):
                The results obtained from the evaluation of eval_config.
        """
        for hp_name, val in eval_config.items():
            if self._hp_info[hp_name] == str:
                assert isinstance(val, str)
                val = self._categories[hp_name].index(val)

            self._observations[hp_name] = np.append(self._observations[hp_name], val)

        for obj_name, val in results.items():
            self._observations[obj_name] = np.append(self._observations[obj_name], val)

        retrain = self._acq_fn_type == PAREGO
        self._train(train_meta_model=retrain)

    def _sample_scalarization_weight(self) -> Optional[torch.Tensor]:
        """
        Sample the weights for scalarization used in ParEGO.
        If acq_fn_type is not ParEGO, then it returns None.

        Returns:
            weights (Optional[torch.Tensor]):
                The weights used in the scalarization of ParEGO.
        """
        weights = None
        if self._acq_fn_type == PAREGO:
            n_obj = len(self._obj_names)
            weights = torch.as_tensor(self._rng.random(n_obj), dtype=torch.float32)
            weights /= weights.sum()

        return weights

    def _fetch_kwargs_for_model(self) -> Dict[str, Any]:
        """
        Fetch the information needed to preprocess observations
        to train a Gaussian process model.
        The weights are used for ParEGO's linear combination,

        Returns:
            kwargs_for_model (Dict[str, Any]):
                The keyword arguments for the data preprocessing.
        """
        return dict(
            bounds=self._bounds,
            hp_names=self._hp_names,
            minimize=self._minimize,
            cat_dims=self._cat_dims,
            weights=self._sample_scalarization_weight(),
        )

    def _update_models(self, kwargs_for_model: Dict[str, Any], train_meta_model: bool) -> None:
        """
        Update each Gaussian process model and their acquisition functions.

        Args:
            kwargs_for_model (Dict[str, Any]):
                The keyword arguments for the data preprocessing.
            train_meta_model (bool):
                Whether to train meta models again.
                We need to re-train each model when we use ParEGo
                as it generates new weights for the linear combination at each iteration.
        """
        for task_name in self._task_names[:-1]:
            if not train_meta_model:
                break

            observations = self._metadata[task_name]
            model, X_train, Y_train = get_model_and_train_data(observations=observations, **kwargs_for_model)
            self._base_models[task_name] = model
            self._acq_fns[task_name] = get_acq_fn(
                model=model, X_train=X_train, Y_train=Y_train, acq_fn_type=self._acq_fn_type
            )

        model, X_train, Y_train = get_model_and_train_data(observations=self._observations, **kwargs_for_model)
        self._base_models[self._target_task_name] = model
        self._acq_fns[self._target_task_name] = get_acq_fn(
            model=model, X_train=X_train, Y_train=Y_train, acq_fn_type=self._acq_fn_type
        )

    def _update_task_weights(self, kwargs_for_model: Dict[str, Any]) -> None:
        """
        Update task weights and instantiate the new acquisition function based on the weights.

        Args:
            kwargs_for_model (Dict[str, Any]):
                The keyword arguments for the data preprocessing.
        """
        kwargs_for_model.pop("cat_dims")
        X_train, Y_train = get_train_data(self._observations, **kwargs_for_model)
        Y_train = Y_train[None, :] if len(Y_train.shape) == 1 else Y_train
        # flip the sign because larger is better in Y_train
        self._task_weights = self._compute_rank_weights(X_train=X_train, Y_train=-Y_train)
        self._acq_fn = TransferAcquisitionFunction(
            acq_fn_list=[self._acq_fns[task_name] for task_name in self._task_names],
            weights=self._task_weights,
        )
        print(self._task_weights_repr())

    def _train(self, train_meta_model: bool = True) -> None:
        """
        Update Gaussian process models, their acquisition functions, and task weights.

        Args:
            train_meta_model (bool):
                Whether to train meta models again.
                We need to re-train each model when we use ParEGo
                as it generates new weights for the linear combination at each iteration.
        """
        kwargs_for_model = self._fetch_kwargs_for_model()  # Sample new weights for ParEGO here
        self._update_models(kwargs_for_model, train_meta_model=train_meta_model)
        self._update_task_weights(kwargs_for_model)

    @abstractmethod
    def _compute_rank_weights(self, X_train: torch.Tensor, Y_train: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for the method that computes the task weights.

        Args:
            X_train (torch.Tensor):
                The training data used for the task weights.
                In principle, this is the observations in the target task.
                X_train.shape = (n_evals, dim).
            Y_train (torch.Tensor):
                The training data used for the task weights.
                In principle, this is the observations in the target task.
                Y_train.shape = (n_obj, n_evals).

        Returns:
            torch.Tensor:
                The task weights.
                The sum of the weights must be 1.
                The shape is (n_tasks, ).
        """
        raise NotImplementedError
