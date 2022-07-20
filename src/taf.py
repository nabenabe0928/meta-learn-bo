from typing import Any, Dict

import numpy as np

from smac.epm.base_epm import BaseEPM
from smac.optimizer.acquisition import AbstractAcquisitionFunction, EI


class TAF(AbstractAcquisitionFunction):
    def __init__(self, model: BaseEPM):
        """Transfer acquisition function from "Scalable Gaussian process-based transfer surrogates
        for hyperparameter optimization" by Wistuba, Schilling and Schmidt-Thieme,
        Machine Learning 2018, https://link.springer.com/article/10.1007/s10994-017-5684-y

        Works both with TST-R and RGPE weighting.
        """
        super().__init__(model)
        self.long_name = "Transfer Acquisition Function"
        self.eta = None
        self.acq: EI = EI(model=None)

    def update(self, **kwargs: Dict[str, Any]) -> None:
        X = kwargs["X"]
        preds = self.model.target_model.predict(X)[0]
        assert id(kwargs["model"]) == id(self.model)
        self.acq.model = None
        self.acq.update(model=self.model.target_model, eta=np.min(preds))
        best_vals = [
            None if weight == 0 else np.min(base_model.predict(X)[0])
            for weight, base_model in zip(self.model.weights, self.model.base_models)
        ]
        self._best_vals = best_vals

    def _compute(self, X: np.ndarray, **kwargs: Dict[str, Any]) -> np.ndarray:
        ei = self.acq._compute(X)
        if self.model.weights[-1] == 1:
            return ei

        improvements = []
        for w, best_val, base_model in zip(self.model.weights, self._best_vals, self.model.base_models):
            if w == 0:
                continue

            preds = base_model._predict(X)[0]
            improvements.append(w * np.maximum(best_val - preds, 0).flatten())

        return ei.flatten() * self.model.weights[-1] + np.sum(improvements, axis=0).reshape((-1, 1))
