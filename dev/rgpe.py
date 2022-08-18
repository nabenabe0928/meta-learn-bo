from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP

import numpy as np

import torch

from fast_pareto import nondominated_rank

from dev.botorch_utils import fit_model, get_acq_fn, get_model_and_train_data, sample
from dev.taf import TransferAcquisitionFunction


PAREGO, EHVI = "parego", "ehvi"
NumericType = Union[int, float]


def drop_ranking_loss(
    ranking_loss: torch.Tensor,
    n_evals: int,
    max_evals: int,
    rng: np.random.RandomState,
) -> torch.Tensor:
    # ranking_loss.shape --> (n_tasks, n_samples)
    (n_tasks, n_samples) = ranking_loss.shape
    better_than_target = torch.sum(ranking_loss[:-1] < ranking_loss[-1], axis=-1)
    p_keep = (better_than_target / n_samples) * (1 - n_evals / max_evals)
    p_keep = torch.hstack([p_keep, torch.tensor(1.0)])  # the target task will not be dropped.

    rnd = torch.as_tensor(rng.random(n_tasks))
    # if rand > p_keep --> drop
    ranking_loss[rnd > p_keep] = torch.max(ranking_loss) * 2 + 1
    return ranking_loss


def leave_one_out_ranks(X: torch.Tensor, Y: torch.Tensor, scalarize: bool, state_dict: OrderedDict) -> torch.Tensor:
    n_samples = len(X)
    masks = torch.eye(n_samples, dtype=torch.bool)
    (n_obj, n_samples) = Y.shape
    loo_preds = np.zeros((n_samples, n_obj))
    for idx, mask in enumerate(masks):
        X_train, Y_train, x_test = X[~mask], Y[:, ~mask], X[mask]
        loo_model = fit_model(X_train=X_train, Y_train=Y_train, scalarize=scalarize, state_dict=state_dict)
        # predict returns the array with the shape of (batch, n_samples, n_objectives)
        loo_preds[idx] = sample(loo_model, x_test)[0][0].numpy()

    return torch.tensor(nondominated_rank(costs=loo_preds, tie_break=True))


def compute_rank_weights(ranking_loss: torch.Tensor) -> torch.Tensor:
    (n_tasks, n_samples) = ranking_loss.shape
    sample_wise_min = torch.amin(ranking_loss, axis=0)  # shape = (n_samples, )
    best_counts = torch.zeros(n_tasks)
    best_task_masks = (ranking_loss == sample_wise_min).T  # shape = (n_samples, n_tasks)
    counts_of_best_in_sample = torch.sum(best_task_masks, axis=-1)  # shape = (n_samples, )
    for best_task_mask, count in zip(best_task_masks, counts_of_best_in_sample):
        best_counts[best_task_mask] += 1.0 / count

    return best_counts / n_samples


class RankingWeigtedGaussianProcessEnsemble:
    def __init__(
        self,
        init_data: Dict[str, np.ndarray],
        metadata: Dict[str, Dict[str, np.ndarray]],
        n_samples: int,
        bounds: Dict[str, Tuple[float, float]],
        hp_names: List[str],
        minimize: Dict[str, bool],
        method: Literal[PAREGO, EHVI],
        target_task_name: str,
        max_evals: int,
        seed: Optional[int] = None,
    ):
        self._base_models: Dict[str, Union[SingleTaskGP, ModelListGP]] = {}
        self._acq_fns: Dict[str, Union[ExpectedImprovement, ExpectedHypervolumeImprovement]] = {}
        self._acq_fn: TransferAcquisitionFunction
        self._metadata: Dict[str, Dict[str, np.ndarray]] = metadata
        self._task_names = list(metadata.keys()) + [target_task_name]
        self._target_task_name = target_task_name
        self._n_tasks = len(self._task_names)
        self._bounds = bounds
        self._n_samples = n_samples
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

    def _bootstrap(self, ranks: torch.Tensor, X_train: torch.Tensor, Y_train: torch.Tensor) -> torch.Tensor:
        # n_samples --> number of bootstrap, n_evals --> number of target observations
        target_state_dict = self._base_models[self._target_task_name].state_dict()
        loo_ranks = leave_one_out_ranks(
            X=X_train, Y=Y_train, scalarize=self._method == PAREGO, state_dict=target_state_dict
        )
        ranks = torch.vstack([ranks, loo_ranks])
        (n_tasks, n_evals) = ranks.shape

        bs_indices = self._rng.choice(n_evals, size=(self._n_samples, n_evals), replace=True)
        bs_preds = torch.stack([r[bs_indices] for r in ranks])  # (n_tasks, n_samples, n_evals)
        rank_target = nondominated_rank(Y_train.T.numpy(), tie_break=True)
        bs_targets = torch.as_tensor(rank_target[bs_indices]).reshape((self._n_samples, n_evals))

        ranking_loss = torch.zeros((n_tasks, self._n_samples))
        ranking_loss[:-1] += torch.sum(
            (bs_preds[:-1, :, :, None] < bs_preds[:-1, :, None, :]) ^ (bs_targets[:, :, None] < bs_targets[:, None, :]),
            axis=(2, 3),
        )
        ranking_loss[-1] += torch.sum(
            (bs_preds[-1, :, :, None] < bs_targets[:, None, :]) ^ (bs_targets[:, :, None] < bs_targets[:, None, :]),
            axis=(1, 2),
        )
        return ranking_loss

    def _compute_rank_weights(self, X_train: torch.Tensor, Y_train: torch.Tensor) -> torch.Tensor:
        if self._n_tasks == 1 or X_train.shape[0] < 3:  # Not sufficient data points
            return torch.ones(self._n_tasks) / self._n_tasks

        (n_obj, n_evals) = Y_train.shape
        ranks = torch.zeros((self._n_tasks - 1, n_evals), dtype=torch.int32)
        for idx, task_name in enumerate(self._task_names[:-1]):
            model = self._base_models[task_name]
            # flip the sign because larger is better in base models
            rank = nondominated_rank(-sample(model, X_train)[0].numpy(), tie_break=True)
            ranks[idx] = torch.as_tensor(rank)

        ranking_loss = self._bootstrap(ranks=ranks, X_train=X_train, Y_train=Y_train)
        ranking_loss = drop_ranking_loss(
            ranking_loss=ranking_loss,
            n_evals=n_evals,
            max_evals=self._max_evals,
            rng=self._rng,
        )
        rank_weights = compute_rank_weights(ranking_loss=ranking_loss)
        return rank_weights
