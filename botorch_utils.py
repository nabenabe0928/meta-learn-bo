from typing import Dict, List, Optional, Tuple

import numpy as np

import torch

from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.models import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


def normalize(
    observations: Dict[str, np.ndarray],
    bounds: Dict[str, Tuple[float, float]],
    hp_names: List[str],
) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray(
            [
                (observations[hp_name] - bounds[hp_name][0]) / (bounds[hp_name][1] - bounds[hp_name][0])
                for hp_name in hp_names
            ]
        )
    )


def denormalize(
    X: torch.Tensor,
    bounds: Dict[str, Tuple[float, float]],
    hp_names: List[str],
) -> Dict[str, float]:
    shape = (len(hp_names),)
    if X.shape != (len(hp_names),):
        raise ValueError(f"The shape of X must be {shape}, but got {X.shape}")

    return {
        hp_name: X[idx] * (bounds[hp_name][1] - bounds[hp_name][0]) + bounds[hp_name][0]
        for idx, hp_name in enumerate(hp_names)
    }


def scalarize(
    Y_train: torch.Tensor,
    weights: torch.Tensor,
    rho: float = 0.05,
) -> torch.Tensor:
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    # Y_train.shape = (n_obj, n_samples), Y_weighted.shape = (n_obj, n_samples)
    Y_weighted = Y_train * weights[:, None]
    # NOTE: since Y is "Larger is better", so we take min of Y_weighted
    return torch.amin(Y_weighted, axis=0) + rho * torch.sum(Y_weighted, axis=0)


def get_train_data(
    observations: Dict[str, np.ndarray],
    bounds: Dict[str, Tuple[float, float]],
    hp_names: List[str],
    minimize: Dict[str, bool],
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # NOTE: Y_train will be transformed so that larger is better for botorch
    X_train = normalize(observations=observations, bounds=bounds, hp_names=hp_names).T
    Y_train = torch.as_tensor(
        np.asarray([(1 - 2 * do_min) * observations[obj_name] for obj_name, do_min in minimize.items()])
    )
    if weights is None:
        Y_mean = Y_train.mean(axis=-1)
        Y_std = Y_train.std(axis=-1)
        return X_train, (Y_train - Y_mean[:, None]) / Y_std[:, None], Y_mean, Y_std
    else:  # scalarization
        Y_train = scalarize(Y_train=Y_train, weights=weights)
        Y_mean = Y_train.mean()
        Y_std = Y_train.std()
        return X_train, (Y_train - Y_mean) / Y_std, Y_mean, Y_std


def fit_gp_model(X_train: torch.Tensor, Y: torch.Tensor, y_mean: float, y_std: float) -> FixedNoiseGP:
    Y_train = Y[:, None]
    Y_var_train = torch.full_like(Y_train, y_std**2)
    model = FixedNoiseGP(train_X=X_train, train_Y=Y_train, train_Yvar=Y_var_train)
    model.Y_mean, model.Y_std = y_mean, y_std
    return model


def fit_model(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    Y_mean: torch.Tensor,
    Y_std: torch.Tensor,
    scalarize: bool = False,
) -> Tuple[ModelListGP, FixedNoiseGP]:

    if scalarize:  # ParEGO
        model = fit_gp_model(X_train=X_train, Y=Y_train, y_mean=Y_mean, y_std=Y_std)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        return model
    else:  # EHVI
        models: List[FixedNoiseGP] = []
        for Y, y_mean, y_std in zip(Y_train, Y_mean, Y_std):
            _model = fit_gp_model(X_train=X_train, Y=Y, y_mean=y_mean, y_std=y_std)
            models.append(_model)

        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        return model


def get_parego(
    observations: torch.Tensor,
    bounds: Dict[str, Tuple[float, float]],
    hp_names: List[str],
    minimize: Dict[str, bool],
    weights: torch.Tensor,
):
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    X_train, Y_train, y_mean, y_std = get_train_data(
        observations=observations,
        bounds=bounds,
        hp_names=hp_names,
        minimize=minimize,
        weights=weights,
    )
    model = fit_model(X_train=X_train, Y_train=Y_train, Y_mean=y_mean, Y_std=y_std, scalarize=True)
    acq_fn = ExpectedImprovement(model=model, best_f=Y_train.amax())
    return acq_fn


def get_ehvi(
    observations: torch.Tensor,
    bounds: Dict[str, Tuple[float, float]],
    hp_names: List[str],
    minimize: Dict[str, bool],
) -> ExpectedHypervolumeImprovement:
    X_train, Y_train, Y_mean, Y_std = get_train_data(
        observations=observations, bounds=bounds, hp_names=hp_names, minimize=minimize
    )
    model = fit_model(X_train=X_train, Y_train=Y_train, Y_mean=Y_mean, Y_std=Y_std)

    with torch.no_grad():
        pred = model.posterior(X_train).mean

    # NOTE: botorch maximizes all objectives and notice that Y.min() is alywas negative
    ref_point = torch.as_tensor([Y.min() * 1.1 for Y in Y_train])
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pred)
    acq_fn = ExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
    )
    return acq_fn


def optimize_acq_fn(
    acq_fn: ExpectedHypervolumeImprovement,
    bounds: Dict[str, Tuple[float, float]],
    hp_names: List[str],
) -> Dict[str, float]:
    kwargs = dict(q=1, num_restarts=10, raw_samples=1 << 8, return_best_only=True)
    standard_bounds = torch.zeros((2, len(hp_names)))
    standard_bounds[1] = 1
    X, _ = optimize_acqf(acq_function=acq_fn, bounds=standard_bounds, **kwargs)
    eval_config = denormalize(X=X.squeeze(), bounds=bounds, hp_names=hp_names)
    return eval_config