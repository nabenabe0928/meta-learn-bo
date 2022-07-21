from typing import Any, Dict

import numpy as np

from src.base_weighted_gp import BaseWeightedGP


class TwoStageTransferWithRanking(BaseWeightedGP):
    def __init__(
        self,
        metadata: Dict[str, Dict[str, np.ndarray]],
        metric_name: str,
        bandwidth: float = 0.1,
        max_evals: int = 50,
        **kwargs: Dict[str, Any],
    ):
        """Two-stage transfer surrogate with ranking from "Scalable Gaussian process-based
        transfer surrogates for hyperparameter optimization" by Wistuba, Schilling and
        Schmidt-Thieme, Machine Learning 2018,
        https://link.springer.com/article/10.1007/s10994-017-5684-y

        Args:
            metadata (Dict[str, Dict[str, np.ndarray]]):
                Dict[task name, Dict[hp_name/metric_name, the array of the corresponding var]]
            max_evals (int):
                How many function call we do during the whole optimization.
            bandwidth (float):
                rho in the original paper.
            metric_name (str):
                The name of the objective func.
        """
        super().__init__(metadata=metadata, max_evals=max_evals, metric_name=metric_name, **kwargs)
        self._bandwidth = bandwidth

    def _compute_rank_weights(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self._n_tasks == 1 or X.shape[0] < 2:  # Not sufficient data points
            return np.ones(self._n_tasks) / self._n_tasks

        n_samples = Y.size
        n_pairs = n_samples * (n_samples - 1)
        # preds.shape = (n_tasks - 1, n_samples)
        preds = np.asarray([model.predict(X)[0].flatten() for model in self.base_models])
        order_info = (preds[:, :, np.newaxis] < preds[:, np.newaxis, :]) ^ (Y[:, np.newaxis] < Y)
        ts = np.sum(order_info, axis=(1, 2)) / (n_pairs * self._bandwidth)
        ts = np.minimum(ts, 1)

        weights = np.ones(self._n_tasks) * 0.75
        weights[:-1] *= 1 - ts**2
        return weights / np.sum(weights)  # normalize and return
