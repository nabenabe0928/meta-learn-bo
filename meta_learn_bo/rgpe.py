from typing import Any, Dict, List

import numpy as np

from smac.epm.gaussian_process import GaussianProcess

from src.base_weighted_gp import BaseWeightedGP
from src.utils import get_gaussian_process


def drop_ranking_loss(
    ranking_loss: np.ndarray,
    n_evals: int,
    max_evals: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    # ranking_loss.shape --> (n_tasks, n_samples)
    (n_tasks, n_samples) = ranking_loss.shape
    better_than_target = np.sum(ranking_loss[:-1] < ranking_loss[-1], axis=-1)
    worse_than_target = n_samples - better_than_target
    p_keep = better_than_target / (better_than_target + worse_than_target)
    p_keep *= 1 - n_evals / max_evals
    p_keep = np.append(p_keep, 1)  # the target task will not be dropped.
    # if rand > p_keep --> drop
    ranking_loss[rng.random(n_tasks) > p_keep] = np.max(ranking_loss) * 2 + 1
    return ranking_loss


def leave_one_out_cross_validation(X: np.ndarray, Y: np.ndarray, target_model: GaussianProcess) -> np.ndarray:
    n_samples = len(X)
    masks = np.eye(n_samples, dtype=np.bool)
    keys = ["configspace", "bounds", "types", "rng", "kernel"]
    loo_preds: List[float] = []
    for mask in masks:
        _x_train, _y_train, _x_test = X[~mask], Y[~mask], X[mask]
        loo_model = get_gaussian_process(**{key: getattr(target_model, key) for key in keys})
        loo_model._train(X=_x_train, y=_y_train, do_optimize=False)
        # predict returns the array with the shape of ([mean/var], n_samples, n_objectives)
        pred_val: float = loo_model.predict(_x_test)[0][0][0]
        loo_preds.append(pred_val)

    return np.asarray(loo_preds)


def compute_rank_weights(ranking_loss: np.ndarray) -> np.ndarray:
    (n_tasks, n_samples) = ranking_loss.shape
    sample_wise_min = np.min(ranking_loss, axis=0)
    best_counts = np.zeros(n_tasks)
    best_task_masks = (ranking_loss == sample_wise_min).T  # shape = (n_samples, n_tasks)
    counts_of_best_in_sample = np.sum(best_task_masks, axis=-1)
    for best_task_mask, count in zip(best_task_masks, counts_of_best_in_sample):
        best_counts[best_task_mask] += 1.0 / count

    return best_counts / n_samples


class RankingWeigtedGaussianProcessEnsemble(BaseWeightedGP):
    def __init__(
        self,
        metadata: Dict[str, Dict[str, np.ndarray]],
        n_samples: int,
        max_evals: int,
        metric_name: str,
        **kwargs: Dict[str, Any],
    ):
        """Ranking-Weighted Gaussian Process Ensemble.

        Args:
            metadata (Dict[str, Dict[str, np.ndarray]]):
                Dict[task name, Dict[hp_name/metric_name, the array of the corresponding var]]
            n_samples (int):
                The number of samples to draw for the approximation of the posterior of a model.
            max_evals (int):
                How many function call we do during the whole optimization.
            metric_name (str):
                The name of the objective func.
        """
        super().__init__(metadata=metadata, max_evals=max_evals, metric_name=metric_name, **kwargs)
        self._n_samples = n_samples

    def _bootstrap(self, preds: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        n_tasks = len(preds) + 1
        loo_preds = leave_one_out_cross_validation(X=X, Y=Y, target_model=self.target_model)
        preds = loo_preds[np.newaxis, :] if preds.size == 0 else np.vstack([preds, loo_preds])
        (_, n_points) = preds.shape  # (n_tasks, n_points:=len(X))
        bs_indices = self._rng.choice(n_points, size=(self._n_samples, n_points), replace=True)

        bs_preds = np.asarray([pred[bs_indices] for pred in preds])  # (n_tasks, n_samples, n_points)
        bs_targets = Y[bs_indices].reshape((self._n_samples, len(Y)))

        ranking_loss = np.zeros((n_tasks, self._n_samples))
        ranking_loss[:-1] += np.sum(
            (bs_preds[:-1, :, :, np.newaxis] < bs_preds[:-1, :, np.newaxis, :])
            ^ (bs_targets[:, :, np.newaxis] < bs_targets[:, np.newaxis, :]),
            axis=(2, 3),
        )
        ranking_loss[-1] += np.sum(
            (bs_preds[-1, :, :, np.newaxis] < bs_targets[:, np.newaxis, :])
            ^ (bs_targets[:, :, np.newaxis] < bs_targets[:, np.newaxis, :]),
            axis=(1, 2),
        )
        return ranking_loss

    def _compute_rank_weights(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if X.shape[0] < 3:  # Not sufficient data points
            return np.ones(self._n_tasks) / self._n_tasks

        preds = [model.predict(X)[0].flatten() for model in self.base_models]
        ranking_loss = self._bootstrap(preds=np.asarray(preds), X=X, Y=Y)
        ranking_loss = drop_ranking_loss(
            ranking_loss=ranking_loss,
            n_evals=len(X),
            max_evals=self._max_evals,
            rng=self._rng,
        )
        return compute_rank_weights(ranking_loss=ranking_loss)
