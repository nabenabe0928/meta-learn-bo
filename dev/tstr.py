from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP

import numpy as np

import torch

from fast_pareto import nondominated_rank

from dev.botorch_utils import get_acq_fn, get_model_and_train_data, sample
from dev.taf import TransferAcquisitionFunction


PAREGO, EHVI = "parego", "ehvi"
NumericType = Union[int, float]


class TwoStageTransferWithRanking:
    def __init__(
        self,
        init_data: Dict[str, np.ndarray],
        metadata: Dict[str, Dict[str, np.ndarray]],
        bounds: Dict[str, Tuple[float, float]],
        hp_names: List[str],
        minimize: Dict[str, bool],
        method: Literal[PAREGO, EHVI],
        target_task_name: str,
        max_evals: int,
        bandwidth: float = 0.1,
        seed: Optional[int] = None,
    ):
        self._base_models: Dict[str, Union[SingleTaskGP, ModelListGP]] = {}
        self._acq_fns: Dict[str, Union[ExpectedImprovement, ExpectedHypervolumeImprovement]] = {}
        self._acq_fn: TransferAcquisitionFunction
        self._metadata: Dict[str, Dict[str, np.ndarray]] = metadata
        self._bandwidth = bandwidth
        self._task_names = list(metadata.keys()) + [target_task_name]
        self._target_task_name = target_task_name
        self._n_tasks = len(self._task_names)
        self._bounds = bounds
        self._max_evals = max_evals
        self._hp_names = hp_names
        self._minimize = minimize
        self._obj_names = list(minimize.keys())
        self._observations: Dict[str, np.ndarray] = init_data
        self._method = method
        self._rng = np.random.RandomState(seed)
        self._task_weights: torch.Tensor
        self._train()

    @property
    def acq_fn(self) -> TransferAcquisitionFunction:
        return self._acq_fn

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        return {key: val.copy() for key, val in self._observations.items()}

    def _task_weights_repr(self) -> str:
        ws = ", ".join([f"{name}: {float(w):.3f}" for name, w in zip(self._task_names, self._task_weights)])
        return f"task weights = ({ws})"

    def update(self, eval_config: Dict[str, NumericType], results: Dict[str, float]) -> None:
        for hp_name, val in eval_config.items():
            self._observations[hp_name] = np.append(self._observations[hp_name], val)

        for obj_name, val in results.items():
            self._observations[obj_name] = np.append(self._observations[obj_name], val)

        retrain = self._method == PAREGO
        self._train(train_meta_model=retrain)

    def _sample_scalarization_weight(self) -> Optional[torch.Tensor]:
        weights = None
        if self._method == PAREGO:
            weights = torch.as_tensor(self._rng.random(self._n_tasks), dtype=torch.float32)
            weights /= weights.sum()

        return weights

    def _fetch_kwargs_for_model(self) -> Dict[str, Any]:
        return dict(
            bounds=self._bounds,
            hp_names=self._hp_names,
            minimize=self._minimize,
            weights=self._sample_scalarization_weight(),
        )

    def _train(self, train_meta_model: bool = True) -> None:
        kwargs = self._fetch_kwargs_for_model()
        for task_name in self._task_names[:-1]:
            if not train_meta_model:
                break

            observations = self._metadata[task_name]
            model, X_train, Y_train = get_model_and_train_data(observations=observations, **kwargs)
            self._base_models[task_name] = model
            self._acq_fns[task_name] = get_acq_fn(model=model, X_train=X_train, Y_train=Y_train, method=self._method)

        model, X_train, Y_train = get_model_and_train_data(observations=self._observations, **kwargs)
        self._base_models[self._target_task_name] = model
        self._acq_fns[self._target_task_name] = get_acq_fn(
            model=model, X_train=X_train, Y_train=Y_train, method=self._method
        )

        Y_train = Y_train[None, :] if len(Y_train.shape) == 1 else Y_train
        # flip the sign because larger is better in Y_train
        self._task_weights = self._compute_rank_weights(X_train=X_train, Y_train=-Y_train)
        self._acq_fn = TransferAcquisitionFunction(
            acq_fn_list=[self._acq_fns[task_name] for task_name in self._task_names],
            weights=self._task_weights,
        )
        print(self._task_weights_repr())

    def _compute_rank_weights(self, X_train: torch.Tensor, Y_train: torch.Tensor) -> torch.Tensor:
        n_evals = X_train.shape[0]
        if self._n_tasks == 1 or n_evals < 2:  # Not sufficient data points
            return torch.ones(self._n_tasks) / self._n_tasks

        n_pairs = n_evals * (n_evals - 1)
        # ranks.shape = (n_tasks - 1, n_evals)
        ranks = np.asarray(
            [
                # flip the sign because larger is better in base models
                nondominated_rank(-sample(self._base_models[task_name], X_train)[0].numpy(), tie_break=True)
                for task_name in self._task_names[:-1]
            ]
        )
        target = nondominated_rank(Y_train.T.numpy(), tie_break=True)
        discordant_info = (ranks[:, :, np.newaxis] < ranks[:, np.newaxis, :]) ^ (target[:, np.newaxis] < target)
        ts = torch.as_tensor(np.sum(discordant_info, axis=(1, 2)) / (n_pairs * self._bandwidth))
        ts = torch.minimum(ts, torch.tensor(1.0))

        weights = torch.ones(self._n_tasks) * 0.75
        weights[:-1] *= 1 - ts**2
        return weights / torch.sum(weights)  # normalize and return
