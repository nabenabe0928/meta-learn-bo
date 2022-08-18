from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

import torch

from fast_pareto import nondominated_rank

from meta_learn_bo.base_weighted_gp import BaseWeightedGP
from meta_learn_bo.utils import EHVI, PAREGO, sample


class TwoStageTransferWithRanking(BaseWeightedGP):
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
        self._bandwidth = bandwidth
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
