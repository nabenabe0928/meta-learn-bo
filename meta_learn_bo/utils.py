from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Tuple, Union

from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

import numpy as np

import torch


PAREGO, EHVI = "parego", "ehvi"
AcqFuncType = Literal["parego", "ehvi"]
NumericType = Union[int, float]
SingleTaskGPType = Union[SingleTaskGP, MixedSingleTaskGP]
ModelType = Union[SingleTaskGP, MixedSingleTaskGP, ModelListGP]


def validate_weights(weights: torch.Tensor) -> None:
    if not torch.isclose(weights.sum(), torch.tensor(1.0)):
        raise ValueError(f"The sum of the weights must be 1, but got {weights}")


def normalize(
    observations: Dict[str, np.ndarray],
    bounds: Dict[str, Tuple[NumericType, NumericType]],
    hp_names: List[str],
) -> torch.Tensor:
    """
    Normalize the feature tensor so that each dimension is in [0, 1].

    Args:
        observations (Dict[str, np.ndarray]):
            The observations.
            Dict[hp_name/obj_name, the array of the corresponding param].
        bounds (Dict[str, Tuple[NumericType, NumericType]]):
            The lower and upper bounds for each hyperparameter.
            Dict[hp_name, Tuple[lower bound, upper bound]].
        hp_names (List[str]):
            The list of hyperparameter names.
            List[hp_name].

    Returns:
        X (torch.Tensor):
            The transformed feature tensor with the shape (dim, n_samples).
    """
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
    bounds: Dict[str, Tuple[NumericType, NumericType]],
    hp_names: List[str],
) -> Dict[str, float]:
    """
    De-normalize the feature tensor from the range of [0, 1].

    Args:
        X (torch.Tensor):
            The transformed feature tensor with the shape (dim, n_samples).
        bounds (Dict[str, Tuple[NumericType, NumericType]]):
            The lower and upper bounds for each hyperparameter.
            Dict[hp_name, Tuple[lower bound, upper bound]].
        hp_names (List[str]):
            The list of hyperparameter names.
            List[hp_name].

    Returns:
        config (Dict[str, float]):
            The config reverted from X.
            Dict[hp_name/obj_name, the corresponding param value].

    TODO:
        * Map to the exact type (int/float)
        * Support categorical parameters
    """
    shape = (len(hp_names),)
    if X.shape != (len(hp_names),):
        raise ValueError(f"The shape of X must be {shape}, but got {X.shape}")

    return {
        hp_name: float(X[idx]) * (bounds[hp_name][1] - bounds[hp_name][0]) + bounds[hp_name][0]
        for idx, hp_name in enumerate(hp_names)
    }


def sample(model: ModelType, X: torch.Tensor) -> torch.Tensor:
    """
    Sample from the posterior based on the model given X.

    Args:
        model (ModelType):
            The Gaussian process model trained on the provided dataset.
        X (torch.Tensor):
            The feature tensor with the shape of (n_samples, dim) that takes as a condition.
            Basically, we sample from y ~ model(f|X).

    Returns:
        preds (torch.Tensor):
            The array with the shape of (batch size, n_samples, n_objectives).
    """
    with torch.no_grad():
        return model.posterior(X).sample()


def scalarize(
    Y_train: torch.Tensor,
    weights: torch.Tensor,
    rho: float = 0.05,
) -> torch.Tensor:
    """
    Compute the linear combination used for ParEGO.

    Args:
        Y_train (torch.Tensor):
            The preprocessed objective tensor.
            Note that this tensor is preprocessed so that
            `larger is better` for the BoTorch internal implementation.
            Y_train.shape = (n_obj, n_evals).
        weights (torch.Tensor):
            The weights for each objective with the shape of (n_obj, ).
        rho (float):
            The hyperparameter used in ParEGO.

    Returns:
        Y_train (torch.Tensor):
            The linear combined version of the objective tensor.
            The shape is (n_evals, ).
    """
    validate_weights(weights)
    # Y_train.shape = (n_obj, n_samples), Y_weighted.shape = (n_obj, n_samples)
    Y_weighted = Y_train * weights[:, None]
    # NOTE: since Y is "Larger is better", so we take min of Y_weighted
    return torch.amin(Y_weighted, dim=0) + rho * torch.sum(Y_weighted, dim=0)


def get_train_data(
    observations: Dict[str, np.ndarray],
    bounds: Dict[str, Tuple[NumericType, NumericType]],
    hp_names: List[str],
    minimize: Dict[str, bool],
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess the observations so that BoTorch can train
    Gaussian process using this data.

    Args:
        observations (Dict[str, np.ndarray]):
            The observations.
            Dict[hp_name/obj_name, the array of the corresponding param].
        bounds (Dict[str, Tuple[NumericType, NumericType]]):
            The lower and upper bounds for each hyperparameter.
            Dict[hp_name, Tuple[lower bound, upper bound]].
        hp_names (List[str]):
            The list of hyperparameter names.
            List[hp_name].
        minimize (Dict[str, bool]):
            The direction of the optimization for each objective.
            Dict[obj_name, whether to minimize or not].
        weights (Optional[torch.Tensor]):
                The weights used in the scalarization of ParEGO.

    Returns:
        X_train (torch.Tensor):
            The preprocessed hyperparameter configurations tensor.
            X_train.shape = (n_evals, dim).
        Y_train (torch.Tensor):
            The preprocessed objective tensor.
            Note that this tensor is preprocessed so that
            `larger is better` for the BoTorch internal implementation.
            Y_train.shape = (n_obj, n_evals).
    """
    # NOTE: Y_train will be transformed so that larger is better for botorch
    # X_train.shape = (n_samples, dim)
    X_train = normalize(observations=observations, bounds=bounds, hp_names=hp_names).T
    # Y_train.shape = (n_obj, n_samples)
    Y_train = torch.as_tensor(
        np.asarray([(1 - 2 * do_min) * observations[obj_name] for obj_name, do_min in minimize.items()])
    )
    if weights is None:
        Y_mean = torch.mean(Y_train, dim=-1)
        Y_std = torch.std(Y_train, dim=-1)
        return X_train, (Y_train - Y_mean[:, None]) / Y_std[:, None]
    else:  # scalarization
        Y_train = scalarize(Y_train=Y_train, weights=weights)
        Y_mean = torch.mean(Y_train)
        Y_std = torch.std(Y_train)
        return X_train, (Y_train - Y_mean) / Y_std


def fit_model(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    cat_dims: List[int],
    scalarize: bool = False,
    state_dict: Optional[OrderedDict] = None,
) -> ModelType:
    """
    Fit Gaussian process model on the provided data.

    Args:
        X_train (torch.Tensor):
            The preprocessed hyperparameter configurations tensor.
            X_train.shape = (n_evals, dim).
        Y_train (torch.Tensor):
            The preprocessed objective tensor.
            Note that this tensor is preprocessed so that
            `larger is better` for the BoTorch internal implementation.
            Y_train.shape = (n_obj, n_evals).
        cat_dims (List[int]):
            The indices of the categorical parameters.
        scalarize (bool):
            Whether to use the scalarization or not.
        state_dict (Optional[OrderedDict]):
            The state dict to reduce the training time in BoTorch.
            This is used for leave-one-out cross validation.

    Returns:
        model (ModelType):
            The Gaussian process model trained on the provided dataset.
    """
    gp_cls = SingleTaskGP if len(cat_dims) == 0 else MixedSingleTaskGP
    kwargs = dict() if len(cat_dims) == 0 else dict(cat_dims=cat_dims)
    if scalarize:  # ParEGO
        model = gp_cls(train_X=X_train, train_Y=Y_train.squeeze()[:, None], **kwargs)
        mll_cls = ExactMarginalLogLikelihood
    else:  # EHVI
        models: List[SingleTaskGPType] = []
        for Y in Y_train:
            _model = gp_cls(train_X=X_train, train_Y=Y[:, None], **kwargs)
            models.append(_model)

        model = ModelListGP(*models)
        mll_cls = SumMarginalLogLikelihood

    if state_dict is None:
        mll = mll_cls(model.likelihood, model)
        fit_gpytorch_model(mll)
    else:
        model.load_state_dict(state_dict)

    return model


def get_model_and_train_data(
    observations: Dict[str, np.ndarray],
    bounds: Dict[str, Tuple[NumericType, NumericType]],
    hp_names: List[str],
    minimize: Dict[str, bool],
    cat_dims: List[int],
    weights: Optional[torch.Tensor] = None,
) -> Tuple[ModelType, torch.Tensor, torch.Tensor]:
    scalarize = weights is not None
    if weights is not None:
        validate_weights(weights)

    X_train, Y_train = get_train_data(
        observations=observations,
        bounds=bounds,
        hp_names=hp_names,
        minimize=minimize,
        weights=weights,
    )
    model = fit_model(X_train=X_train, Y_train=Y_train, cat_dims=cat_dims, scalarize=scalarize)

    return model, X_train, Y_train


def get_parego(model: SingleTaskGPType, X_train: torch.Tensor, Y_train: torch.Tensor) -> ExpectedImprovement:
    """
    Get the ParEGO acquisition funciton.

    Args:
        model (SingleTaskGPType):
            The Gaussian process model trained on the provided dataset.
        X_train (torch.Tensor):
            The preprocessed hyperparameter configurations tensor.
            X_train.shape = (n_evals, dim).
        Y_train (torch.Tensor):
            The preprocessed objective tensor.
            Note that this tensor is preprocessed so that
            `larger is better` for the BoTorch internal implementation.
            Y_train.shape = (n_obj, n_evals).

    Returns:
        acq_fn (ExpectedImprovement):
            The acquisition function obtained based on the provided dataset and the model.
    """
    acq_fn = ExpectedImprovement(model=model, best_f=Y_train.amax())
    return acq_fn


def get_ehvi(model: ModelListGP, X_train: torch.Tensor, Y_train: torch.Tensor) -> ExpectedHypervolumeImprovement:
    """
    Get the Expected hypervolume improvement acquisition funciton.

    Args:
        model (ModelListGP):
            The Gaussian process model trained on the provided dataset.
        X_train (torch.Tensor):
            The preprocessed hyperparameter configurations tensor.
            X_train.shape = (n_evals, dim).
        Y_train (torch.Tensor):
            The preprocessed objective tensor.
            Note that this tensor is preprocessed so that
            `larger is better` for the BoTorch internal implementation.
            Y_train.shape = (n_obj, n_evals).

    Returns:
        acq_fn (ExpectedHypervolumeImprovement):
            The acquisition function obtained based on the provided dataset and the model.
    """
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


def get_acq_fn(
    model: ModelType, X_train: torch.Tensor, Y_train: torch.Tensor, acq_fn_type: AcqFuncType
) -> Union[ExpectedImprovement, ExpectedHypervolumeImprovement]:
    """
    Get the specified acquisition funciton.

    Args:
        model (ModelType):
            The Gaussian process model trained on the provided dataset.
        X_train (torch.Tensor):
            The preprocessed hyperparameter configurations tensor.
            X_train.shape = (n_evals, dim).
        Y_train (torch.Tensor):
            The preprocessed objective tensor.
            Note that this tensor is preprocessed so that
            `larger is better` for the BoTorch internal implementation.
            Y_train.shape = (n_obj, n_evals).

    Returns:
        acq_fn (Union[ExpectedImprovement, ExpectedHypervolumeImprovement]):
            The acquisition function obtained based on the provided dataset and the model.
    """
    supported_acq_fn_types = {"parego": get_parego, "ehvi": get_ehvi}
    for acq_fn_name, func in supported_acq_fn_types.items():
        if acq_fn_name == acq_fn_type:
            return func(model=model, X_train=X_train, Y_train=Y_train)
    else:
        raise ValueError(f"acq_fn_type must be in {supported_acq_fn_types}, but got {acq_fn_type}")


def optimize_acq_fn(
    acq_fn: ExpectedHypervolumeImprovement,
    bounds: Dict[str, Tuple[NumericType, NumericType]],
    hp_names: List[str],
) -> Dict[str, NumericType]:
    """
    Optimize the given acquisition function and obtain the next configuration to evaluate.

    Args:
        acq_fn (Union[ExpectedImprovement, ExpectedHypervolumeImprovement]):
            The acquisition function obtained based on the provided dataset and the model.
        bounds (Dict[str, Tuple[NumericType, NumericType]]):
            The lower and upper bounds for each hyperparameter.
            Dict[hp_name, Tuple[lower bound, upper bound]].
        hp_names (List[str]):
            The list of hyperparameter names.
            List[hp_name].

    Returns:
        eval_config (Dict[str, float]):
            The config to evaluate.
            Dict[hp_name/obj_name, the corresponding param value].
    """
    kwargs = dict(q=1, num_restarts=10, raw_samples=1 << 8, return_best_only=True)
    standard_bounds = torch.zeros((2, len(hp_names)))
    standard_bounds[1] = 1
    X, _ = optimize_acqf(acq_function=acq_fn, bounds=standard_bounds, **kwargs)
    eval_config = denormalize(X=X.squeeze(), bounds=bounds, hp_names=hp_names)
    return eval_config
