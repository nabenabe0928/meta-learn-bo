from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

import torch

from fast_pareto import nondominated_rank

from dev.botorch_utils import EHVI, PAREGO, fit_model, sample
from dev.base_weighted_gp import BaseWeightedGP


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


class RankingWeigtedGaussianProcessEnsemble(BaseWeightedGP):
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

        self._n_samples = n_samples
        super().__init__(
            init_data=init_data,
            metadata=metadata,
            bounds=bounds,
            hp_names=hp_names,
            minimize=minimize,
            method=method,
            target_task_name=target_task_name,
            max_evals=max_evals,
            seed=seed,
        )

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
