from copy import deepcopy
from typing import List, Optional, Tuple

from ConfigSpace import ConfigurationSpace
import numpy as np
from smac.epm.gaussian_process import GaussianProcess
from smac.epm.gaussian_process.utils.prior import HorseshoePrior, LognormalPrior
from smac.epm.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel, HammingKernel
from sklearn.gaussian_process.kernels import Kernel


LENGTH_SCALE_BOUNDS = (np.exp(-6.754111155189306), np.exp(0.0858637988771976))


def get_const_and_noise_kernels(rng: np.random.RandomState) -> Tuple[ConstantKernel, WhiteKernel]:
    const_kernel = ConstantKernel(
        2.0,
        constant_value_bounds=(np.exp(-10), np.exp(2)),
        prior=LognormalPrior(mean=0.0, sigma=1, rng=rng),
    )
    noise_kernel = WhiteKernel(
        noise_level=1e-8,
        noise_level_bounds=(np.exp(-25), np.exp(2)),
        prior=HorseshoePrior(scale=0.1, rng=rng),
    )
    return const_kernel, noise_kernel


def get_main_kernel(
    numerical_indices: np.ndarray, categorical_indices: np.ndarray
) -> Kernel:

    n_numericals, n_categoricals = numerical_indices.size, categorical_indices.size

    if n_numericals > 0:
        numerical_kernel = Matern(
            np.ones(n_numericals),
            [LENGTH_SCALE_BOUNDS for _ in range(n_numericals)],
            nu=2.5,
            operate_on=numerical_indices,
        )
    else:
        numerical_kernel = None

    if n_categoricals > 0:
        categorical_kernel = HammingKernel(
            np.ones(n_categoricals),
            [LENGTH_SCALE_BOUNDS for _ in range(n_categoricals)],
            operate_on=categorical_indices,
        )
    else:
        categorical_kernel = None

    if numerical_kernel is None and categorical_kernel is None:
        raise ValueError("Dimension must be a positive number")
    elif numerical_kernel is None:
        return categorical_kernel
    elif categorical_kernel is None:
        return numerical_kernel
    else:
        return numerical_kernel * categorical_kernel


def get_kernel(
    types: List[int],
    bounds: List[Tuple[float, float]],
    rng: np.random.RandomState,
) -> Kernel:
    const_kernel, noise_kernel = get_const_and_noise_kernels(rng)
    dim = len(types)
    numerical_indices = np.arange(dim)[np.array(types) == 0]
    categorical_indices = np.arange(dim)[np.array(types) != 0]
    main_kernel = get_main_kernel(numerical_indices, categorical_indices)
    return const_kernel * main_kernel + noise_kernel


def get_gaussian_process(
    configspace: ConfigurationSpace,
    types: List[int],
    bounds: List[Tuple[float, float]],
    rng: np.random.RandomState,
    kernel: Optional[Kernel],
) -> GaussianProcess:
    """Get the default GP class from SMAC. Sets the kernel and its hyperparameters for the
    problem at hand."""

    kernel = get_kernel(types=types, bounds=bounds, rng=rng) if kernel is None else deepcopy(kernel)

    gp = GaussianProcess(
        kernel=kernel,
        normalize_y=True,
        seed=rng.randint(0, 1 << 20),
        types=types,
        bounds=bounds,
        configspace=configspace,
    )
    return gp
