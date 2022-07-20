from typing import Dict, List, Tuple

import numpy as np

from smac.epm.base_epm import BaseEPM
from smac.epm.gaussian_process import GaussianProcess

from src.utils import get_gaussian_process


def roll_col(X: np.ndarray, shift: int) -> np.ndarray:
    """Rotate columns to right by shift."""
    return np.concatenate((X[:, -shift:], X[:, :-shift]), axis=1)


def drop_ranking_loss(
    ranking_loss: np.ndarray,
    n_evals: int,
    max_evals: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    # ranking_loss.shape --> (n_tasks, n_samples)
    (n_tasks, n_samples) = ranking_loss.shape[-1]
    better_than_target = np.sum(ranking_loss[:-1] < ranking_loss[-1], axis=-1)
    worse_than_target = n_samples - better_than_target
    p_keep = better_than_target / (better_than_target + worse_than_target)
    p_keep *= (1 - n_evals / max_evals)
    # if rand > p_keep --> drop
    ranking_loss[rng.random(n_tasks) > p_keep] = np.max(ranking_loss) * 2 + 1
    return ranking_loss


def leave_one_out_cross_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    target_model: GaussianProcess,
) -> List[float]:
    n_samples = len(x_train)
    masks = np.eye(n_samples, dtype=np.bool)
    keys = ["configspace", "bounds", "types", "rng", "kernel"]
    loo_preds: List[float] = []
    for mask in masks:
        _x_train, _y_train, _x_test = x_train[~mask], y_train[~mask], x_train[mask]
        loo_model = get_gaussian_process(**{getattr(target_model, key) for key in keys})
        loo_model._train(X=_x_train, y=_y_train, do_optimize=False)
        # predict returns the array with the shape of (n_samples, n_objectives)
        pred_val: float = loo_model.predict(_x_test, cov_return_type=None)[0][0]
        loo_preds.append(pred_val)

    return loo_preds


def bootstrap(
    preds: List[np.ndarray],
    target_model: GaussianProcess,
    n_samples: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    n_tasks = len(preds) + 1
    loo_preds = leave_one_out_cross_validation(
        x_train=x_train, y_train=y_train, target_model=target_model
    )
    preds.append(loo_preds)
    preds = np.asarray(preds)

    bootstrap_indices = rng.choice(preds.shape[1],
                                   size=(n_samples, preds.shape[1]),
                                   replace=True)

    bootstrap_predictions = []
    bootstrap_targets = y_train[bootstrap_indices].reshape((n_samples, len(y_train)))
    for m in range(n_tasks):
        bootstrap_predictions.append(preds[m, bootstrap_indices])

    ranking_loss = np.zeros((n_tasks, n_samples))
    for i in range(len(n_tasks - 1)):

        for j in range(len(y_train)):
            ranking_loss[i] += np.sum(
                (
                    roll_col(bootstrap_predictions[i], j) < bootstrap_predictions[i])
                    ^ (roll_col(bootstrap_targets, j) < bootstrap_targets
                ), axis=1
            )
    for j in range(len(y_train)):
        ranking_loss[-1] += np.sum(
            (
                (roll_col(bootstrap_predictions[-1], j) < bootstrap_targets)
                ^ (roll_col(bootstrap_targets, j) < bootstrap_targets)
            ), axis=1
        )

    return ranking_loss


def _compute_rank_weights(ranking_loss: np.ndarray) -> np.ndarray:
    (n_tasks, n_samples) = ranking_loss.shape
    sample_wise_min = np.min(ranking_loss, axis=0)
    best_counts = np.zeros(n_tasks)
    best_task_masks = (ranking_loss == sample_wise_min)

    for best_task_mask in best_task_masks:
        best_counts[best_task_mask] += 1. / np.sum(best_task_mask)

    return best_counts / n_samples


def compute_rank_weights(
    x_train: np.ndarray,
    y_train: np.ndarray,
    base_models: List[GaussianProcess],
    target_model: GaussianProcess,
    n_samples: int,
    max_evals: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Compute ranking weights for each base model and the target model
    (using LOOCV for the target model).

    Returns
    -------
    weights : np.ndarray
    """
    preds = [model.predict(x_train)[0].flatten() for model in base_models]
    ranking_loss = bootstrap(
        preds=preds,
        target_model=target_model,
        n_samples=n_samples,
        x_train=x_train,
        y_train=y_train,
        rng=rng,
    )
    ranking_loss = drop_ranking_loss(
        ranking_loss=ranking_loss,
        n_evals=len(x_train),
        max_evals=max_evals,
        rng=rng,
    )
    return _compute_rank_weights(ranking_loss=ranking_loss)


class RGPE(BaseEPM):
    def __init__(
        self,
        metadata: Dict[str, Dict[str, np.ndarray]],
        n_samples: int,
        max_evals: int,
        metric_name: str,
        **kwargs
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
        super().__init__(**kwargs)

        self._max_evals = max_evals
        self._n_samples = n_samples
        self._rng = np.random.RandomState(self.seed)
        self._metric_name = metric_name
        self._gp_kwargs = dict(
            bounds=self.bounds, types=self.types, configspace=self.configspace, kernel=None, rng=self._rng
        )

        self._n_tasks = len(metadata) + 1
        self._base_models: List[GaussianProcess]
        self._train_models_on_metatask(metadata)

    def _train_on_data(self, data: Dict[str, np.ndarray]) -> GaussianProcess:
        # x_train must be the shape of (n_samples, n_feats)
        X = np.asarray([vals for key, vals in data.items() if key != self._metric_name]).T
        y_scaled = self._standardize(data[self._metric_name])
        model = get_gaussian_process(**self._gp_kwargs)
        model.train(X=X, Y=y_scaled)
        return model

    def _train_models_on_metatask(self, metadata: Dict[str, Dict[str, np.ndarray]]) -> None:
        base_models: List[GaussianProcess] = [self._train_on_data(data) for data in metadata.values()]
        self._base_models = base_models

    def _standardize(self, y: np.ndarray) -> np.ndarray:
        mean = y.mean()
        std = y.std()
        if std == 0:
            std = 1

        self._y_mean = mean
        self._y_std = std
        y_scaled = (y - self._y_mean) / self._y_std
        return y_scaled.flatten()

    def _train(self, X: np.ndarray, Y: np.ndarray) -> BaseEPM:
        y_scaled = self._standardize(Y)
        self._target_model = get_gaussian_process(**self._gp_kwargs)
        self._target_model.train(X, y_scaled)

        if X.shape[0] < 3:  # Not sufficient data points
            self._weights = np.ones(self._n_tasks) / self._n_tasks
        else:
            self._weights = compute_rank_weights(
                x_train=X,
                y_train=y_scaled,
                base_models=self._base_models,
                target_model=self._target_model,
                n_samples=self._n_samples,
                max_evals=self._max_evals,
                rng=self._rng,
            )

        return self

    def _predict(self, X: np.ndarray, cov_return_type='diagonal_cov') -> Tuple[np.ndarray, np.ndarray]:
        self._weights /= self._weights.sum()
        weighted_means, weighted_covs = [], []

        for idx, w in enumerate(self._weights):
            if w ** 2 <= 0:
                continue

            model = self._base_models[idx] if idx < self._n_tasks - 1 else self._target_model
            mean, cov = model._predict(X, cov_return_type)

            weighted_means.append(w * mean)
            weighted_covs.append(cov * w ** 2)

        mean = np.sum(np.stack(weighted_means), axis=0) * self._y_std + self._y_mean
        cov = np.sum(weighted_covs, axis=0) * (self._y_std ** 2)
        return mean, cov

    def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray:
        self._weights /= self._weights.sum()
        samples = []
        for idx, w in enumerate(self._weights):
            if w ** 2 <= 0:
                continue

            model = self._base_models[idx] if idx < self._n_tasks - 1 else self._target_model
            # preds.shape -> (n_samples, 1)
            preds = model.sample_functions(X_test, n_funcs)
            samples.append(w * preds)

        return np.sum(samples, axis=0)
