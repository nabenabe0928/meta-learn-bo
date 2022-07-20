from typing import Dict, List, Tuple

import numpy as np

from smac.epm.base_epm import BaseEPM
from smac.epm.gaussian_process import GaussianProcess

from src.utils import get_gaussian_process


EPS = 1e-8


def drop_ranking_loss(
    ranking_loss: np.ndarray, n_evals: int, max_evals: int, rng: np.random.RandomState,
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


def leave_one_out_cross_validation(x: np.ndarray, y: np.ndarray, target_model: GaussianProcess) -> List[float]:
    n_samples = len(x)
    masks = np.eye(n_samples, dtype=np.bool)
    keys = ["configspace", "bounds", "types", "rng", "kernel"]
    loo_preds: List[float] = []
    for mask in masks:
        _x_train, _y_train, _x_test = x[~mask], y[~mask], x[mask]
        loo_model = get_gaussian_process(**{getattr(target_model, key) for key in keys})
        loo_model._train(X=_x_train, y=_y_train, do_optimize=False)
        # predict returns the array with the shape of (n_samples, n_objectives)
        pred_val: float = loo_model.predict(_x_test, cov_return_type=None)[0][0]
        loo_preds.append(pred_val)

    return loo_preds


def compute_rank_weights(ranking_loss: np.ndarray) -> np.ndarray:
    (n_tasks, n_samples) = ranking_loss.shape
    sample_wise_min = np.min(ranking_loss, axis=0)
    best_counts = np.zeros(n_tasks)
    best_task_masks = (ranking_loss == sample_wise_min)

    for best_task_mask in best_task_masks:
        best_counts[best_task_mask] += 1. / np.sum(best_task_mask)

    return best_counts / n_samples


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

    def _bootstrap(self, preds: List[np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_tasks = len(preds) + 1
        loo_preds = leave_one_out_cross_validation(x=x, y=y, target_model=self._target_model)
        preds.append(loo_preds)
        preds = np.asarray(preds)
        (_, n_points) = preds.shape  # (n_tasks, n_points)
        bs_indices = self._rng.choice(n_points, size=(self._n_samples, n_points), replace=True)

        bs_preds = [pred[bs_indices] for pred in preds]  # (n_tasks, n_samples, n_points)
        bs_targets = y[bs_indices].reshape((self._n_samples, len(y)))

        ranking_loss = np.zeros((n_tasks, self._n_samples))
        for i in range(len(n_tasks - 1)):
            # TODO: Check
            pass

        return ranking_loss

    def _compute_rank_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.shape[0] < 3:  # Not sufficient data points
            return np.ones(self._n_tasks) / self._n_tasks

        preds = [model.predict(x, cov_return_type=None).flatten() for model in self._base_models]
        ranking_loss = self._bootstrap(preds=preds, x=x, y=y)
        ranking_loss = drop_ranking_loss(
            ranking_loss=ranking_loss, n_evals=len(x), max_evals=self._max_evals, rng=self._rng,
        )
        return compute_rank_weights(ranking_loss=ranking_loss)

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

        for idx, (w, model) in enumerate(zip(self._weights, self._base_models + [self._target_model])):
            if w ** 2 <= EPS:
                continue

            mean, cov = model._predict(X, cov_return_type)
            weighted_means.append(w * mean)
            weighted_covs.append(cov * w ** 2)

        mean = np.sum(np.stack(weighted_means), axis=0) * self._y_std + self._y_mean
        cov = np.sum(weighted_covs, axis=0) * (self._y_std ** 2)
        return mean, cov

    def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray:
        self._weights /= self._weights.sum()
        samples = []
        for idx, (w, model) in enumerate(zip(self._weights, self._base_models + [self._target_model])):
            if w ** 2 <= EPS:
                continue

            preds = model.sample_functions(X_test, n_funcs)  # preds.shape -> (n_samples, 1)
            samples.append(w * preds)

        return np.sum(samples, axis=0)
