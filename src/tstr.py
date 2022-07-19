from typing import Dict, List, Tuple, Union

from ConfigSpace import Configuration
import numpy as np
from smac.configspace import convert_configurations_to_array
from smac.epm.base_epm import BaseEPM
from rgpe.utils import get_gaussian_process


class TSTR(BaseEPM):

    def __init__(
        self,
        training_data: Dict[int, Dict[str, Union[List[Configuration], np.ndarray]]],
        bandwidth: float = 0.1,
        variance_mode: str = 'target',
        weight_dilution_strategy: Union[int, str] = 'None',
        number_of_function_evaluations: float = 50,
        **kwargs
    ):
        """
        Two-stage transfer surrogate with ranking from "Scalable Gaussian process-based
        transfer surrogates for hyperparameter optimization" by Wistuba, Schilling and
        Schmidt-Thieme, Machine Learning 2018,
        https://link.springer.com/article/10.1007/s10994-017-5684-y

        Parameters
        ----------
        training_data
            Dictionary containing the training data for each meta-task. Mapping from an integer (
            task ID) to a dictionary, which is a mapping from configuration to performance.
        bandwidth
            rho in the original paper
        variance_mode
            Can be either ``'average'`` to return the weighted average of the variance
            predictions of the individual models or ``'target'`` to only obtain the variance
            prediction of the target model. Changing this is only necessary to use the model
            together with the expected improvement.
        weight_dilution_strategy
            Can be one of the following four:
            * ``'probabilistic-ld'``: the method presented in the paper
            * ``'probabilistic'``: the method presented in the paper, but without the time-dependent
              pruning of meta-models
            * an integer: a deterministic strategy described in https://arxiv.org/abs/1802.02219v1
            * ``None``: no weight dilution prevention
        number_of_function_evaluations
            Optimization horizon - used to compute the time-dependent factor in the probability
            of dropping base models for the weight dilution prevention strategy
            ``'probabilistic-ld'``.
        """

        if kwargs.get('instance_features') is not None:
            raise NotImplementedError()
        super().__init__(**kwargs)
        self.training_data = training_data

        self.bandwidth = bandwidth
        self.rng = np.random.RandomState(self.seed)
        self.variance_mode = variance_mode
        self.weight_dilution_strategy = weight_dilution_strategy
        self.number_of_function_evaluations = number_of_function_evaluations

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

        weights = np.zeros(len(self.model_list_))
        weights[-1] = 0.75

        discordant_pairs_per_task = {}

        for model_idx, model in enumerate(self.base_models):
            if X.shape[0] < 2:
                weights[model_idx] = 0.75
            else:
                mean, _ = model.predict(X)
                discordant_pairs = 0
                total_pairs = 0
                for i in range(X.shape[0]):
                    for j in range(i + 1, X.shape[0]):
                        if (Y[i] < Y[j]) ^ (mean[i] < mean[j]):
                            discordant_pairs += 1
                        total_pairs += 1
                t = discordant_pairs / total_pairs / self.bandwidth
                discordant_pairs_per_task[model_idx] = discordant_pairs
                if t < 1:
                    weights[model_idx] = 0.75 * (1 - t ** 2)
                else:
                    weights[model_idx] = 0

        # perform model pruning
        # use this only for ablation
        if X.shape[0] >= 2:
            p_drop = []
            if self.weight_dilution_strategy in ['probabilistic', 'probabilistic-ld']:
                for i in range(len(self.base_models)):
                    concordant_pairs = total_pairs - discordant_pairs_per_task[i]
                    proba_keep = concordant_pairs / total_pairs
                    if self.weight_dilution_strategy == 'probabilistic-ld':
                        proba_keep = proba_keep * (1 - len(X) / float(self.number_of_function_evaluations))
                    proba_drop = 1 - proba_keep
                    p_drop.append(proba_drop)
                    r = self.rng.rand()
                    if r < proba_drop:
                        weights[i] = 0
            elif self.weight_dilution_strategy == 'None':
                pass
            else:
                raise ValueError(self.weight_dilution_strategy)

        weights /= np.sum(weights)
        print(weights)
        self.weights_ = weights

        self.weights_over_time.append(weights)
        # create model and acquisition function
        return self

    def _predict(self, X: np.ndarray, cov_return_type: str = 'diagonal_cov') -> Tuple[np.ndarray, np.ndarray]:

        if cov_return_type != 'diagonal_cov':
            raise NotImplementedError(cov_return_type)

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
            mean, covar = self.model_list_[raw_idx]._predict(X)

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
            _, covar = self.model_list_[-1]._predict(X, cov_return_type)
            weighted_covars.append(covar)

        # set mean and covariance to be the rank-weighted sum the means and covariances
        # of the base models and target model
        mean_x = np.sum(np.stack(weighted_means), axis=0) * self.Y_std_ + self.Y_mean_
        covar_x = np.sum(weighted_covars, axis=0) * (self.Y_std_ ** 2)
        return mean_x, covar_x

    def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray:
        """
        Samples F function values from the current posterior at the N
        specified test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        n_funcs: int
            Number of function values that are drawn at each test point.

        Returns
        ----------
        function_samples: np.array(F, N)
            The F function values drawn at the N test points.
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
