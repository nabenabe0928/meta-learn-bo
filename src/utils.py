import copy
import typing

import botorch.sampling.qmc
from ConfigSpace import ConfigurationSpace
import numpy as np
from smac.epm.gaussian_process import GaussianProcess
from smac.epm.gp_base_prior import HorseshoePrior, LognormalPrior
from smac.epm.gp_kernels import ConstantKernel, Matern, WhiteKernel, HammingKernel


def get_gaussian_process(
    configspace: ConfigurationSpace,
    types: typing.List[int],
    bounds: typing.List[typing.Tuple[float, float]],
    rng: np.random.RandomState,
    kernel,
) -> GaussianProcess:
    """Get the default GP class from SMAC. Sets the kernel and its hyperparameters for the
    problem at hand."""

    if kernel is None:
        cov_amp = ConstantKernel(
            2.0,
            constant_value_bounds=(np.exp(-10), np.exp(2)),
            prior=LognormalPrior(mean=0.0, sigma=1, rng=rng),
        )

        cont_dims = np.where(np.array(types) == 0)[0]
        cat_dims = np.where(np.array(types) != 0)[0]

        if len(cont_dims) > 0:
            exp_kernel = Matern(
                np.ones([len(cont_dims)]),
                [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in
                 range(len(cont_dims))],
                nu=2.5,
                operate_on=cont_dims,
            )

        if len(cat_dims) > 0:
            ham_kernel = HammingKernel(
                np.ones([len(cat_dims)]),
                [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in
                 range(len(cat_dims))],
                operate_on=cat_dims,
            )

        assert len(cat_dims) + len(cont_dims) == len(configspace.get_hyperparameters()), (
            len(cat_dims) + len(cont_dims), len(configspace.get_hyperparameters())
        )

        noise_kernel = WhiteKernel(
            noise_level=1e-8,
            noise_level_bounds=(np.exp(-25), np.exp(2)),
            prior=HorseshoePrior(scale=0.1, rng=rng),
        )

        if len(cont_dims) > 0 and len(cat_dims) > 0:
            # both
            kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
        elif len(cont_dims) > 0 and len(cat_dims) == 0:
            # only cont
            kernel = cov_amp * exp_kernel + noise_kernel
        elif len(cont_dims) == 0 and len(cat_dims) > 0:
            # only cont
            kernel = cov_amp * ham_kernel + noise_kernel
        else:
            raise ValueError()
    else:
        kernel = copy.deepcopy(kernel)

    gp = GaussianProcess(
        kernel=kernel,
        normalize_y=True,
        seed=rng.randint(0, 2 ** 20),
        types=types,
        bounds=bounds,
        configspace=configspace,
    )
    return gp


def sample_sobol(loo_model, locations, num_samples, engine_seed):
    """Sample from a Sobol sequence. Wraps the sampling to deal with the issue that the predictive
    covariance matrix might not be decomposable and fixes this by adding a small amount of noise
    to the diagonal."""

    y_mean, y_cov = loo_model.predict(locations, cov_return_type='full_cov')
    initial_noise = 1e-14
    while initial_noise < 1:
        try:
            L = np.linalg.cholesky(y_cov + np.eye(len(locations)) * 1e-14)
            break
        except np.linalg.LinAlgError:
            initial_noise *= 10
            continue
    if initial_noise >= 1:
        rval = np.tile(y_mean, reps=num_samples).transpose()
        return rval

    engine = botorch.sampling.qmc.NormalQMCEngine(len(y_mean), seed=engine_seed, )
    samples_alt = y_mean.flatten() + (engine.draw(num_samples).numpy() @ L)
    return samples_alt
