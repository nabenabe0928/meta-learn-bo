from typing import List, Union

import torch

from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.analytic import MultiObjectiveAnalyticAcquisitionFunction


EPS = 1e-8


class TransferAcquisitionFunction(MultiObjectiveAnalyticAcquisitionFunction):
    def __init__(
        self, acq_fn_list: List[Union[ExpectedHypervolumeImprovement, ExpectedImprovement]], weights: torch.Tensor
    ):
        assert torch.isclose(weights.sum(), torch.tensor(1.0))
        super().__init__(model=None)
        self._acq_fn_list = acq_fn_list
        self._weights = weights

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size = X.shape[0]
        out = torch.zeros((batch_size,), dtype=torch.float64)
        for acq_fn, weight in zip(self._acq_fn_list, self._weights):
            if weight > EPS:  # basically, if weight is non-zero, we compute
                out += weight * acq_fn(X)

        return out
