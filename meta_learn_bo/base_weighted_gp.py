from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP

from meta_learn_bo.taf import TransferAcquisitionFunction
from meta_learn_bo.utils import AcqFuncType, NumericType, PAREGO, get_acq_fn, get_model_and_train_data, get_train_data

import numpy as np

import torch


class BaseWeightedGP:
    def __init__(
        self,
        init_data: Dict[str, np.ndarray],
        metadata: Dict[str, Dict[str, np.ndarray]],
        bounds: Dict[str, Tuple[NumericType, NumericType]],
        hp_names: List[str],
        minimize: Dict[str, bool],
        acq_fn_type: AcqFuncType,
        target_task_name: str,
        max_evals: int,
        seed: Optional[int] = None,
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
                Dict[hp_name, Tuple[lower bound, upper bound]].
            hp_names (List[str]):
                The list of hyperparameter names.
                List[hp_name].
            minimize (Dict[str, bool]):
                The direction of the optimization for each objective.
                Dict[obj_name, whether to minimize or not].
            acq_fn_type (Literal[PAREGO, EHVI]):
                The acquisition function type.
            target_task_name (str):
                The name of the target task.
            max_evals (int):
                How many hyperparameter configurations to evaluate during the optimization.
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
        self._hp_names = hp_names
        self._minimize = minimize
        self._obj_names = list(minimize.keys())
        self._observations: Dict[str, np.ndarray] = init_data
        self._acq_fn_type = acq_fn_type
        self._rng = np.random.RandomState(seed)
        self._task_weights: torch.Tensor
        self._validate_input()
        self._train()

    @property
    def acq_fn(self) -> TransferAcquisitionFunction:
        return self._acq_fn

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        return {key: val.copy() for key, val in self._observations.items()}

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

    def update(self, eval_config: Dict[str, NumericType], results: Dict[str, float]) -> None:
        """
        Update the target observations, (a) Gaussian process model(s),
        and its/their acquisition function(s).
        If the acq_fn_type is ParEGO, we need to re-train each Gaussian process models
        and the corresponding acquisition functions.

        Args:
            eval_config (Dict[str, NumericType]):
                The hyperparameter configuration that were evaluated.
            results (Dict[str, float]):
                The results obtained from the evaluation of eval_config.
        """
        for hp_name, val in eval_config.items():
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
            weights = torch.as_tensor(self._rng.random(self._n_tasks), dtype=torch.float32)
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