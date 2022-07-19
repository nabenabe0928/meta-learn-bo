from typing import Dict, List, Tuple, Union, Callable

import numpy as np

from ConfigSpace import Configuration
from smac.configspace import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM
from smac.epm.gaussian_process import GaussianProcess
from rgpe.utils import get_gaussian_process, sample_sobol


def roll_col(X: np.ndarray, shift: int) -> np.ndarray:
    """
    Rotate columns to right by shift.
    """
    return np.concatenate((X[:, -shift:], X[:, :-shift]), axis=1)


def compute_ranking_loss(
    f_samps: np.ndarray,
    target_y: np.ndarray,
    target_model: bool,
) -> np.ndarray:
    """
    Compute ranking loss for each sample from the posterior over target points.
    """
    y_stack = np.tile(target_y.reshape((-1, 1)), f_samps.shape[0]).transpose()
    rank_loss = np.zeros(f_samps.shape[0])
    if not target_model:
        for i in range(1, target_y.shape[0]):
            rank_loss += np.sum(
                (roll_col(f_samps, i) < f_samps) ^ (roll_col(y_stack, i) < y_stack),
                axis=1
            )
    else:
        for i in range(1, target_y.shape[0]):
            rank_loss += np.sum(
                (roll_col(f_samps, i) < y_stack) ^ (roll_col(y_stack, i) < y_stack),
                axis=1
            )

    return rank_loss


def get_target_model_loocv_sample_preds(
    train_x: np.ndarray,
    train_y: np.ndarray,
    num_samples: int,
    model: GaussianProcess,
    engine_seed: int,
) -> np.ndarray:
    """
    Use LOOCV to fit len(train_y) independent GPs and sample from their posterior to obtain an
    approximate sample from the target model.

    This sampling does not take into account the correlation between observations which occurs
    when the predictive uncertainty of the Gaussian process is unequal zero.
    """
    masks = np.eye(len(train_x), dtype=np.bool)
    train_x_cv = np.stack([train_x[~m] for m in masks])
    train_y_cv = np.stack([train_y[~m] for m in masks])
    test_x_cv = np.stack([train_x[m] for m in masks])

    samples = np.zeros((num_samples, train_y.shape[0]))
    for i in range(train_y.shape[0]):
        loo_model = get_gaussian_process(
            configspace=model.configspace,
            bounds=model.bounds,
            types=model.types,
            rng=model.rng,
            kernel=model.kernel,
        )
        loo_model._train(X=train_x_cv[i], y=train_y_cv[i], do_optimize=False)

        samples_i = sample_sobol(loo_model, test_x_cv[i], num_samples, engine_seed).flatten()

        samples[:, i] = samples_i

    return samples


def compute_target_model_ranking_loss(
    train_x: np.ndarray,
    train_y: np.ndarray,
    num_samples: int,
    model: GaussianProcess,
    engine_seed: int,
) -> np.ndarray:
    """
    Use LOOCV to fit len(train_y) independent GPs and sample from their posterior to obtain an
    approximate sample from the target model.

    This function does joint draws from all observations (both training data and left out sample)
    to take correlation between observations into account, which can occur if the predictive
    variance of the Gaussian process is unequal zero. To avoid returning a tensor, this function
    directly computes the ranking loss.
    """
    masks = np.eye(len(train_x), dtype=np.bool)
    train_x_cv = np.stack([train_x[~m] for m in masks])
    train_y_cv = np.stack([train_y[~m] for m in masks])

    ranking_losses = np.zeros(num_samples, dtype=np.int)
    for i in range(train_y.shape[0]):
        loo_model = get_gaussian_process(
            configspace=model.configspace,
            bounds=model.bounds,
            types=model.types,
            rng=model.rng,
            kernel=model.kernel,
        )
        loo_model._train(X=train_x_cv[i], y=train_y_cv[i], do_optimize=False)
        samples_i = sample_sobol(loo_model, train_x, num_samples, engine_seed)

        for j in range(len(train_y)):
            ranking_losses += (samples_i[:, i] < samples_i[:, j]) ^ (train_y[i] < train_y[j])

    return ranking_losses


def weight_dilution(
    base_models: List[GaussianProcess],
    ranking_loss: np.ndarray,
    alpha: float,
    n_evals: int,
    max_evals: int,
    rng: np.random.RandomState,
) -> List[float]:
    p_drop: List[float] = []
    for i in range(len(base_models)):
        better_than_target = np.sum(ranking_loss[i, :] < ranking_loss[-1, :])
        worse_than_target = np.sum(ranking_loss[i, :] >= ranking_loss[-1, :])
        correction_term = alpha * (better_than_target + worse_than_target)
        proba_keep = better_than_target / (better_than_target + worse_than_target + correction_term)
        proba_keep = proba_keep * (1 - n_evals / max_evals)
        proba_drop = 1 - proba_keep
        p_drop.append(proba_drop)

        if rng.rand() < proba_drop:
            ranking_loss[i, :] = np.max(ranking_loss) * 2 + 1

    return p_drop


def leave_one_out_cross_validation(
    train_x: np.ndarray,
    train_y: np.ndarray,
    target_model: GaussianProcess,
) -> List[float]:
    n_samples = len(train_x)
    masks = np.eye(n_samples, dtype=np.bool)
    loo_preds: List[float] = []
    for mask in masks:
        _x_train, _y_train, _x_test = train_x[~mask], train_y[~mask], train_x[mask]
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
    num_samples: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:

    n_tasks = len(preds) + 1
    loo_preds = leave_one_out_cross_validation(
        train_x=train_x, train_y=train_y, target_model=target_model
    )
    preds.append(loo_preds)
    preds = np.asarray(preds)

    bootstrap_indices = rng.choice(preds.shape[1],
                                   size=(num_samples, preds.shape[1]),
                                   replace=True)

    bootstrap_predictions = []
    bootstrap_targets = train_y[bootstrap_indices].reshape((num_samples, len(train_y)))
    for m in range(n_tasks):
        bootstrap_predictions.append(preds[m, bootstrap_indices])

    ranking_losses = np.zeros((n_tasks, num_samples))
    for i in range(len(n_tasks - 1)):

        for j in range(len(train_y)):
            ranking_losses[i] += np.sum(
                (
                    roll_col(bootstrap_predictions[i], j) < bootstrap_predictions[i])
                    ^ (roll_col(bootstrap_targets, j) < bootstrap_targets
                ), axis=1
            )
    for j in range(len(train_y)):
        ranking_losses[-1] += np.sum(
            (
                (roll_col(bootstrap_predictions[-1], j) < bootstrap_targets)
                ^ (roll_col(bootstrap_targets, j) < bootstrap_targets)
            ), axis=1
        )

    return ranking_losses


def compute_rank_weights(
    train_x: np.ndarray,
    train_y: np.ndarray,
    base_models: List[GaussianProcess],
    target_model: GaussianProcess,
    num_samples: int,
    number_of_function_evaluations,
    rng: np.random.RandomState,
    alpha: float = 0.0,
) -> np.ndarray:
    """
    Compute ranking weights for each base model and the target model
    (using LOOCV for the target model).

    Returns
    -------
    weights : np.ndarray
    """
    preds = [model.predict(train_x)[0].flatten() for model in base_models]
    ranking_loss = bootstrap(preds)

    # perform model pruning
    p_drop = weight_dilution(
        base_models=base_models,
        ranking_loss=ranking_loss,
        alpha=alpha,
        n_evals=len(train_x),
        max_evals=number_of_function_evaluations,
        rng=rng,
    )

    # compute best model (minimum ranking loss) for each sample
    # this differs from v1, where the weight is given only to the target model in case of a tie.
    # Here, we distribute the weight fairly among all participants of the tie.
    minima = np.min(ranking_loss, axis=0)
    assert len(minima) == num_samples
    best_models = np.zeros(len(base_models) + 1)
    for i, minimum in enumerate(minima):
        minimum_locations = ranking_loss[:, i] == minimum
        sample_from = np.where(minimum_locations)[0]

        for sample in sample_from:
            best_models[sample] += 1. / len(sample_from)

    # compute proportion of samples for which each model is best
    rank_weights = best_models / num_samples
    return rank_weights, p_drop


class RGPE(AbstractEPM):

    def __init__(
        self,
        training_data: Dict[int, Dict[str, Union[List[Configuration], np.ndarray]]],
        num_posterior_samples: int,
        number_of_function_evaluations: int,
        sampling_mode: str = 'correct',
        variance_mode: str = 'average',
        alpha: float = 0.0,
        **kwargs
    ):
        """Ranking-Weighted Gaussian Process Ensemble.

        Parameters
        ----------
        training_data
            Dictionary containing the training data for each meta-task. Mapping from an integer (
            task ID) to a dictionary, which is a mapping from configuration to performance.
        num_posterior_samples
            Number of samples to draw for approximating the posterior probability of a model
            being the best model to explain the observations on the target task.
        number_of_function_evaluations
            Optimization horizon - used to compute the time-dependent factor in the probability
            of dropping base models for the weight dilution prevention strategy
            ``'probabilistic-ld'``.
        variance_mode
            Can be either ``'average'`` to return the weighted average of the variance
            predictions of the individual models or ``'target'`` to only obtain the variance
            prediction of the target model. Changing this is only necessary to use the model
            together with the expected improvement.
        alpha
            Regularization hyperparameter to increase aggressiveness of dropping base models when
            using the weight dilution strategies ``'probabilistic-ld'`` or ``'probabilistic'``.
        """

        if kwargs.get('instance_features') is not None:
            raise NotImplementedError()
        super().__init__(**kwargs)
        self.training_data = training_data

        self.number_of_function_evaluations = number_of_function_evaluations
        self.num_posterior_samples = num_posterior_samples
        self.rng = np.random.RandomState(self.seed)
        self.variance_mode = variance_mode
        self.alpha = alpha

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
        self.weights_over_time = []
        self.p_drop_over_time = []

    def _standardize(self, y: np.ndarray) -> np.ndarray:
        mean = y.mean()
        std = y.std()
        if std == 0:
            std = 1

        self.Y_mean_ = mean
        self.Y_std_ = std
        y_scaled = (y - self.Y_mean_) / self.Y_std_
        return y_scaled.flatten()

    def _train(self, X: np.ndarray, Y: np.ndarray) -> AbstractEPM:
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
            try:
                self.weights_, p_drop = compute_rank_weights(
                    train_x=X,
                    train_y=y_scaled,
                    base_models=self.base_models,
                    target_model=target_model,
                    num_samples=self.num_posterior_samples,
                    number_of_function_evaluations=self.number_of_function_evaluations,
                    rng=self.rng,
                    alpha=self.alpha,
                )
            except Exception as e:
                print(e)
                self.weights_ = np.zeros((len(self.model_list_, )))
                self.weights_[-1] = 1
                p_drop = np.ones((len(self.base_models, ))) * np.NaN

        print('Weights', self.weights_)
        self.weights_over_time.append(self.weights_)
        self.p_drop_over_time.append(p_drop)

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

            if self.variance_mode == 'average':
                weighted_covars.append(covar * weight ** 2)
            elif self.variance_mode == 'target':
                if raw_idx + 1 == len(self.weights_):
                    weighted_covars.append(covar)
            else:
                raise ValueError()

        if len(weighted_covars) == 0:
            if self.variance_mode != 'target':
                raise ValueError(self.variance_mode)
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
