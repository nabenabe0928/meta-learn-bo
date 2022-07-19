from typing import Dict, List, Tuple, Union

import numpy as np

from ConfigSpace import Configuration
from smac.configspace import convert_configurations_to_array
from smac.epm.base_epm import BaseEPM
from smac.epm.gaussian_process import GaussianProcess
from rgpe.utils import get_gaussian_process


def roll_col(X: np.ndarray, shift: int) -> np.ndarray:
    """
    Rotate columns to right by shift.
    """
    return np.concatenate((X[:, -shift:], X[:, :-shift]), axis=1)


def weight_dilution(
    base_models: List[GaussianProcess],
    ranking_loss: np.ndarray,
    n_evals: int,
    max_evals: int,
    rng: np.random.RandomState,
) -> List[float]:
    p_drop: List[float] = []
    for i in range(len(base_models)):
        better_than_target = np.sum(ranking_loss[i, :] < ranking_loss[-1, :])
        worse_than_target = np.sum(ranking_loss[i, :] >= ranking_loss[-1, :])
        proba_keep = better_than_target / (better_than_target + worse_than_target)
        proba_keep = proba_keep * (1 - n_evals / max_evals)
        proba_drop = 1 - proba_keep
        p_drop.append(proba_drop)

        if rng.rand() < proba_drop:
            ranking_loss[i, :] = np.max(ranking_loss) * 2 + 1

    return p_drop


def leave_one_out_cross_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    target_model: GaussianProcess,
) -> List[float]:
    n_samples = len(x_train)
    masks = np.eye(n_samples, dtype=np.bool)
    loo_preds: List[float] = []
    for mask in masks:
        _x_train, _y_train, _x_test = x_train[~mask], y_train[~mask], x_train[mask]
        loo_model = get_gaussian_process(
            configspace=target_model.configspace,
            bounds=target_model.bounds,
            types=target_model.types,
            rng=target_model.rng,
            kernel=target_model.kernel,
        )
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

    ranking_losses = np.zeros((n_tasks, n_samples))
    for i in range(len(n_tasks - 1)):

        for j in range(len(y_train)):
            ranking_losses[i] += np.sum(
                (
                    roll_col(bootstrap_predictions[i], j) < bootstrap_predictions[i])
                    ^ (roll_col(bootstrap_targets, j) < bootstrap_targets
                ), axis=1
            )
    for j in range(len(y_train)):
        ranking_losses[-1] += np.sum(
            (
                (roll_col(bootstrap_predictions[-1], j) < bootstrap_targets)
                ^ (roll_col(bootstrap_targets, j) < bootstrap_targets)
            ), axis=1
        )

    return ranking_losses


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
    ranking_loss = bootstrap(preds)

    # perform model pruning
    p_drop = weight_dilution(
        base_models=base_models,
        ranking_loss=ranking_loss,
        n_evals=len(x_train),
        max_evals=max_evals,
        rng=rng,
    )

    # compute best model (minimum ranking loss) for each sample
    # this differs from v1, where the weight is given only to the target model in case of a tie.
    # Here, we distribute the weight fairly among all participants of the tie.
    minima = np.min(ranking_loss, axis=0)
    assert len(minima) == n_samples
    best_models = np.zeros(len(base_models) + 1)
    for i, minimum in enumerate(minima):
        minimum_locations = ranking_loss[:, i] == minimum
        sample_from = np.where(minimum_locations)[0]

        for sample in sample_from:
            best_models[sample] += 1. / len(sample_from)

    # compute proportion of samples for which each model is best
    rank_weights = best_models / n_samples
    return rank_weights, p_drop


class RGPE(BaseEPM):

    def __init__(
        self,
        training_data: Dict[int, Dict[str, Union[List[Configuration], np.ndarray]]],
        n_samples: int,
        max_evals: int,
        **kwargs
    ):
        """Ranking-Weighted Gaussian Process Ensemble.

        Parameters
        ----------
        training_data
            Dictionary containing the training data for each meta-task. Mapping from an integer (
            task ID) to a dictionary, which is a mapping from configuration to performance.
        n_samples (int):
            The number of samples to draw for the approximation of the posterior of a model.
        max_evals (int):
            How many function call we do during the whole optimization.
        """

        if kwargs.get('instance_features') is not None:
            raise NotImplementedError()
        super().__init__(**kwargs)
        self.training_data = training_data

        self.max_evals = max_evals
        self.n_samples = n_samples
        self.rng = np.random.RandomState(self.seed)

        base_models = []
        for task in training_data:
            model = get_gaussian_process(
                bounds=self.bounds,
                types=self.types,
                configspace=self.configspace,
                rng=self.rng,
                kernel=None,
            )
            y_scaled = self._standardize(training_data[task]['y'])

            configs = training_data[task]['configurations']
            X = convert_configurations_to_array(configs)

            model.train(
                X=X,
                Y=y_scaled,
            )
            base_models.append(model)
        self.base_models = base_models

    def _standardize(self, y: np.ndarray) -> np.ndarray:
        mean = y.mean()
        std = y.std()
        if std == 0:
            std = 1

        self.Y_mean_ = mean
        self.Y_std_ = std
        y_scaled = (y - self.Y_mean_) / self.Y_std_
        return y_scaled.flatten()

    def _train(self, X: np.ndarray, Y: np.ndarray) -> BaseEPM:
        """SMAC training function"""
        y_scaled = self._standardize(Y)

        target_model = get_gaussian_process(
            bounds=self.bounds,
            types=self.types,
            configspace=self.configspace,
            rng=self.rng,
            kernel=None,
        )
        self.target_model = target_model.train(X, y_scaled)
        self.model_list_ = self.base_models + [target_model]

        if X.shape[0] < 3:
            self.weights_ = np.ones(len(self.model_list_)) / len(self.model_list_)
            p_drop = np.ones((len(self.base_models, ))) * np.NaN
        else:
            self.weights_, p_drop = compute_rank_weights(
                x_train=X,
                y_train=y_scaled,
                base_models=self.base_models,
                target_model=target_model,
                n_samples=self.n_samples,
                max_evals=self.max_evals,
                rng=self.rng,
            )

        return self

    def _predict(self, X: np.ndarray, cov_return_type='diagonal_cov') -> Tuple[np.ndarray, np.ndarray]:
        """SMAC predict function"""

        # compute posterior for each model
        weighted_means = []
        weighted_covars = []

        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights_ ** 2 > 0).nonzero()[0]
        non_zero_weights = self.weights_[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()

        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            weight = non_zero_weights[non_zero_weight_idx]
            mean, covar = self.model_list_[raw_idx]._predict(X, cov_return_type)

            weighted_means.append(weight * mean)
            weighted_covars.append(covar * weight ** 2)

        if len(weighted_covars) == 0:
            _, covar = self.model_list_[-1]._predict(X, cov_return_type=cov_return_type)
            weighted_covars.append(covar)

        mean_x = np.sum(np.stack(weighted_means), axis=0) * self.Y_std_ + self.Y_mean_
        covar_x = np.sum(weighted_covars, axis=0) * (self.Y_std_ ** 2)
        return mean_x, covar_x

    def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray:
        """
        Sample function values from the posterior of the specified test points.
        """

        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights_ ** 2 > 0).nonzero()[0]
        non_zero_weights = self.weights_[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()

        samples = []
        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            weight = non_zero_weights[non_zero_weight_idx]

            funcs = self.model_list_[raw_idx].sample_functions(X_test, n_funcs)
            funcs = funcs * weight
            samples.append(funcs)
        samples = np.sum(samples, axis=0)
        return samples
