from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from smac.epm.base_epm import BaseEPM
from smac.epm.gaussian_process import GaussianProcess

from src.utils import get_gaussian_process


EPS = 1e-8


class BaseWeightedGP(BaseEPM):
    def __init__(
        self,
        metadata: Dict[str, Dict[str, np.ndarray]],
        max_evals: int,
        metric_name: str,
        **kwargs: Dict[str, Any],
    ):
        """Base class for Weighted Gaussian Process Ensemble.

        Args:
            metadata (Dict[str, Dict[str, np.ndarray]]):
                Dict[task name, Dict[hp_name/metric_name, the array of the corresponding var]]
            max_evals (int):
                How many function call we do during the whole optimization.
            metric_name (str):
                The name of the objective func.
        """
        super().__init__(**kwargs)
        self._max_evals = max_evals
        self._rng = np.random.RandomState(self.seed)
        self._metric_name = metric_name
        self._gp_kwargs = dict(
            bounds=self.bounds, types=self.types, configspace=self.configspace, kernel=None, rng=self._rng
        )
        self._n_tasks = len(metadata) + 1
        self.base_models: List[GaussianProcess]
        self._train_models_on_metatask(metadata)

    @abstractmethod
    def _compute_rank_weights(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _train_on_data(self, data: Dict[str, np.ndarray]) -> GaussianProcess:
        # x_train must be the shape of (n_samples, n_feats)
        X = np.asarray([vals for key, vals in data.items() if key != self._metric_name]).T
        y_scaled = self._standardize(data[self._metric_name])
        model = get_gaussian_process(**self._gp_kwargs)
        model.train(X=X, Y=y_scaled)
        return model

    def _train_models_on_metatask(self, metadata: Dict[str, Dict[str, np.ndarray]]) -> None:
        base_models: List[GaussianProcess] = [self._train_on_data(data) for data in metadata.values()]
        self.base_models = base_models

    def _standardize(self, Y: np.ndarray) -> np.ndarray:
        mean = Y.mean()
        std = Y.std()
        if std == 0:
            std = 1

        self._y_mean = mean
        self._y_std = std
        y_scaled = (Y - self._y_mean) / self._y_std
        return y_scaled.flatten()

    def _train(self, X: np.ndarray, Y: np.ndarray) -> BaseEPM:
        y_scaled = self._standardize(Y)
        self.target_model = get_gaussian_process(**self._gp_kwargs)
        self.target_model.train(X, y_scaled)
        self.weights = self._compute_rank_weights(X=X, Y=y_scaled)
        return self

    def _predict(self, X: np.ndarray, cov_return_type: str = "diagonal_cov") -> Tuple[np.ndarray, np.ndarray]:
        self.weights /= self.weights.sum()
        weighted_means, weighted_covs = [], []

        for idx, (w, model) in enumerate(zip(self.weights, self.base_models + [self.target_model])):
            if w**2 <= EPS:
                continue

            mean, cov = model._predict(X, cov_return_type)
            weighted_means.append(w * mean)
            weighted_covs.append(cov * w**2)

        mean = np.sum(np.stack(weighted_means), axis=0) * self._y_std + self._y_mean
        cov = np.sum(weighted_covs, axis=0) * (self._y_std**2)
        return mean, cov

    def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray:
        self.weights /= self.weights.sum()
        samples = []
        for idx, (w, model) in enumerate(zip(self.weights, self.base_models + [self.target_model])):
            if w**2 <= EPS:
                continue

            preds = model.sample_functions(X_test, n_funcs)  # preds.shape -> (n_samples, 1)
            samples.append(w * preds)

        return np.sum(samples, axis=0)
