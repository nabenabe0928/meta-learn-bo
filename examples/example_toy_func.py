import warnings

from meta_learn_bo import (
    RankingWeigtedGaussianProcessEnsemble,
    TwoStageTransferWithRanking,
)

from examples.toy_func import get_initial_samples, get_toy_func_info, toy_func


warnings.filterwarnings("ignore")


def optimize(acq_fn_type: str = "parego", rank_weight_type: str = "rgpe") -> None:
    kwargs = get_toy_func_info()
    n_init, max_evals = 10, 20
    observations = get_initial_samples(n_init)
    bo_method = {
        "rgpe": RankingWeigtedGaussianProcessEnsemble,
        "tstr": TwoStageTransferWithRanking,
    }[rank_weight_type]
    model = bo_method(
        init_data=observations,
        metadata={"src": get_initial_samples(n_init=50)},
        max_evals=max_evals,
        acq_fn_type=acq_fn_type,
        target_task_name="target",
        **kwargs,
    )

    for t in range(max_evals - n_init):
        eval_config = model.optimize_acq_fn()
        results = toy_func(eval_config)
        model.update(eval_config=eval_config, results=results)
        print(f"Iteration {t + 1}: ", eval_config, results)

    print(model.observations)


if __name__ == "__main__":
    optimize(acq_fn_type="ehvi")
    optimize(acq_fn_type="parego")