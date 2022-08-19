import unittest

from meta_learn_bo.rgpe import (
    RankingWeigtedGaussianProcessEnsemble,
    compute_ranking_loss,
    compute_rank_weights,
    drop_ranking_loss,
)

import numpy as np

import torch

from utils import get_kwargs_and_observations


def test_compute_rank_weights() -> None:
    kwargs, observations = get_kwargs_and_observations(size=5)
    kwargs.update(n_bootstraps=50)
    metadata = {}
    _, metadata["src20"] = get_kwargs_and_observations(size=10)
    _, metadata["src30"] = get_kwargs_and_observations(size=15)
    rgpe = RankingWeigtedGaussianProcessEnsemble(init_data=observations, metadata=metadata, **kwargs)
    assert getattr(rgpe, "_task_weights") is not None
    assert torch.isclose(torch.sum(rgpe._task_weights), torch.tensor(1.0))

    kwargs, observations = get_kwargs_and_observations(size=2)
    kwargs.update(n_bootstraps=50)
    metadata = {}
    _, metadata["src20"] = get_kwargs_and_observations(size=20)
    _, metadata["src30"] = get_kwargs_and_observations(size=30)
    rgpe = RankingWeigtedGaussianProcessEnsemble(init_data=observations, metadata=metadata, **kwargs)
    assert getattr(rgpe, "_task_weights") is not None
    assert torch.isclose(torch.sum(rgpe._task_weights), torch.tensor(1.0))

    kwargs, observations = get_kwargs_and_observations(size=10)
    metadata = {}
    rgpe = RankingWeigtedGaussianProcessEnsemble(init_data=observations, metadata=metadata, **kwargs)
    assert getattr(rgpe, "_task_weights") is not None
    assert torch.allclose(rgpe._task_weights, torch.ones(1))


def test_drop_ranking_loss() -> None:
    rng = np.random.RandomState(0)
    # ranking_loss.shape = (n_tasks, n_bootstraps)
    ranking_loss = torch.tensor(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ]
    )
    # ==> p_keep = 0.5 * (1 - 1) = 0
    res = drop_ranking_loss(
        ranking_loss.clone(),
        n_evals=100,
        max_evals=100,
        rng=rng,
    )
    assert torch.allclose(res, torch.tensor([[3, 3, 3, 3], [0, 1, 1, 0]]))

    # ==> p_keep = 0.5 * (1 - 0.5) = 0.25
    count = 0
    tot = 1000
    for _ in range(tot):
        res = drop_ranking_loss(
            ranking_loss.clone(),
            n_evals=50,
            max_evals=100,
            rng=rng,
        )
        count += torch.allclose(res, torch.tensor([[3, 3, 3, 3], [0, 1, 1, 0]]))

    p_drop = 1 - 0.25
    # variance of binomial ==> tot * p_drop * p_keep
    # ===> std of the mean ==> sqrt(p_drop * p_keep / tot)
    # = sqrt(0.25 * 0.75 / 1000) = 0.025 * sqrt(3/10)
    # ==> 5σ will rarely happen!!
    assert p_drop - 0.05 <= count / tot <= p_drop + 0.05


def test_compute_rank_weights_func() -> None:
    ranking_loss = torch.tensor(
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 0, 0],
        ]
    )
    rank_weights = compute_rank_weights(ranking_loss)
    ans = torch.tensor([(2 + 1 / 3 + 1 / 2) / 8, (1 + 1 / 3) / 8, (3 + 1 / 3 + 1 / 2) / 8])
    assert torch.allclose(rank_weights, ans)


def test_compute_ranking_loss() -> None:
    """
    For bootstrap 1 in task 0:
        F(0/0) F(1/0) F(2/0) F(3/0) F(4/0)
        T(0/1) F(1/1) F(2/1) F(3/1) F(4/1)
        T(0/2) T(1/2) F(2/2) F(3/2) F(4/2)
        T(0/3) T(1/3) T(2/3) F(3/3) F(4/3)
        T(0/4) T(1/4) T(2/4) T(3/4) F(4/4)

        F(2/2) F(3/2) T(1/2) F(5/2) F(4/2)
        T(2/3) F(3/3) T(1/3) F(5/3) F(4/3)
        F(2/1) F(3/1) F(1/1) F(5/1) F(4/1)
        T(2/5) T(3/5) T(1/5) F(5/5) T(4/5)
        T(2/4) T(3/4) T(1/4) F(5/4) F(4/4)

        ==> XOR (discordant info)
        F(F/F) F(F/F) T(F/T) F(F/F) F(F/F)
        F(T/T) F(F/F) T(F/T) F(F/F) F(F/F)
        T(T/F) T(T/F) F(F/F) F(F/F) F(F/F)
        F(T/T) F(T/T) F(T/T) F(F/F) T(F/T)
        F(T/T) F(T/T) F(T/T) T(T/F) F(F/F)
        => True (6)

    For bootstrap 2 in task 0:
        F(2/2) F(3/2) F(2/2) F(3/2) F(2/2)
        T(2/3) F(3/3) T(2/3) F(3/3) T(2/3)
        F(2/2) F(3/2) F(2/2) F(3/2) F(2/2)
        T(2/3) F(3/3) T(2/3) F(3/3) T(2/3)
        F(2/2) F(3/2) F(2/2) F(3/2) F(2/2)

        F(1/1) F(5/1) F(1/1) F(5/1) F(1/1)
        T(1/5) F(5/5) T(1/5) F(5/5) T(1/5)
        F(1/1) F(5/1) F(1/1) F(5/1) F(1/1)
        T(1/5) F(5/5) T(1/5) F(5/5) T(1/5)
        F(1/1) F(5/1) F(1/1) F(5/1) F(1/1)

        ==> XOR (discordant info)
        All same
        => True (0)

    For bootstrap 3 in task 0:
        F(0/0) F(2/0) F(4/0) F(2/0) F(3/0)
        T(0/2) F(2/2) F(4/2) F(2/2) F(3/2)
        T(0/4) T(2/4) F(4/4) T(2/4) T(3/4)
        T(0/2) F(2/2) F(4/2) F(2/2) F(3/2)
        T(0/3) T(2/3) F(4/3) T(2/3) F(3/3)

        F(2/2) T(1/2) F(4/2) T(1/2) F(5/2)
        F(2/1) F(1/1) F(4/1) F(1/1) F(5/1)
        T(2/4) T(1/4) F(4/4) T(1/4) F(5/4)
        F(2/1) F(1/1) F(4/1) F(1/1) F(5/1)
        T(2/5) T(1/5) T(4/5) T(1/5) F(5/5)

        ==> XOR (discordant info)
        F(F/F) T(F/T) F(F/F) T(F/T) F(F/F)
        T(T/F) F(F/F) F(F/F) F(F/F) F(F/F)
        F(T/T) F(T/T) F(F/F) F(T/T) T(T/F)
        T(T/F) F(F/F) F(F/F) F(F/F) F(F/F)
        F(T/T) F(T/T) T(F/T) F(T/T) F(F/F)
        => True (6)

    Task 1 ==> 0 as the preds are identical to the target.

    For bootstrap, we use (preds vs targets) and (targets vs targets)
    For bootstrap 1 in the target task:
        preds = [0, 1, 3, 2, 4]
        targets = [0, 1, 2, 3, 4]
        preds < targets
        F(0/0) F(1/0) F(3/0) F(2/0) F(4/0)
        T(0/1) F(1/1) F(3/1) F(2/1) F(4/1)
        T(0/2) T(1/2) F(3/2) F(2/2) F(4/2)
        T(0/3) T(1/3) F(3/3) T(2/3) F(4/3)
        T(0/4) T(1/4) T(3/4) T(2/4) F(4/4)

        targets vs targets
        F(0/0) F(1/0) F(2/0) F(3/0) F(4/0)
        T(0/1) F(1/1) F(2/1) F(3/1) F(4/1)
        T(0/2) T(1/2) F(2/2) F(3/2) F(4/2)
        T(0/3) T(1/3) T(2/3) F(3/3) F(4/3)
        T(0/4) T(1/4) T(2/4) T(3/4) F(4/4)

        ==> XOR (discordant info)
        F(F/F) F(F/F) F(F/F) F(F/F) F(F/F)
        F(T/T) F(F/F) F(F/F) F(F/F) F(F/F)
        F(T/T) F(T/T) F(F/F) F(F/F) F(F/F)
        F(T/T) F(T/T) T(F/T) T(T/F) F(F/F)
        F(T/T) F(T/T) F(T/T) F(T/T) F(F/F)
        => True (2)

    For bootstrap 2 in the target task:
        preds = [3, 2, 3, 2, 3]
        targets = [2, 3, 2, 3, 2]
        preds < targets
        F(3/2) F(2/2) F(3/2) F(2/2) F(3/2)
        F(3/3) T(2/3) F(3/3) T(2/3) F(3/3)
        F(3/2) F(2/2) F(3/2) F(2/2) F(3/2)
        F(3/3) T(2/3) F(3/3) T(2/3) F(3/3)
        F(3/2) F(2/2) F(3/2) F(2/2) F(3/2)

        targets vs targets
        F(2/2) F(3/2) F(2/2) F(3/2) F(2/2)
        T(2/3) F(3/3) T(2/3) F(3/3) T(2/3)
        F(2/2) F(3/2) F(2/2) F(3/2) F(2/2)
        T(2/3) F(3/3) T(2/3) F(3/3) T(2/3)
        F(2/2) F(3/2) F(2/2) F(3/2) F(2/2)

        ==> XOR (discordant info)
        F(F/F) F(F/F) F(F/F) F(F/F) F(F/F)
        T(F/T) T(T/F) T(F/T) T(T/F) T(F/T)
        F(F/F) F(F/F) F(F/F) F(F/F) F(F/F)
        T(F/T) T(T/F) T(F/T) T(T/F) T(F/T)
        F(F/F) F(F/F) F(F/F) F(F/F) F(F/F)
        => True (10)

    For bootstrap 3 in the target task:
        preds = [0, 3, 4, 3, 2]
        targets = [0, 2, 4, 2, 3]
        preds < targets
        T(0/0) F(3/0) F(4/0) F(3/0) F(2/0)
        T(0/2) F(3/2) F(4/2) F(3/2) F(2/2)
        T(0/4) T(3/4) F(4/4) T(3/4) T(2/4)
        T(0/2) F(3/2) F(4/2) F(3/2) F(2/2)
        T(0/3) F(3/3) F(4/3) F(3/3) T(2/3)

        targets vs targets
        F(0/0) F(2/0) F(4/0) F(2/0) F(3/0)
        T(0/2) F(2/2) F(4/2) F(2/2) F(3/2)
        T(0/4) T(2/4) F(4/4) T(2/4) T(3/4)
        T(0/2) F(2/2) F(4/2) F(2/2) F(3/2)
        T(0/3) T(2/3) F(4/3) T(2/3) F(3/3)

        ==> XOR (discordant info)
        T(T/F) F(F/F) F(F/F) F(F/F) F(F/F)
        F(T/T) F(F/F) F(F/F) F(F/F) F(F/F)
        F(T/T) F(T/T) F(F/F) F(T/T) F(T/T)
        F(T/T) F(F/F) F(F/F) F(F/F) F(F/F)
        F(T/T) F(F/T) F(F/F) T(F/T) T(T/F)
        => True (3)
    """
    rank_preds = torch.tensor(
        [
            [2, 3, 1, 5, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 3, 2, 4],
        ]
    )
    rank_targets = torch.tensor([0, 1, 2, 3, 4])
    # bs_indices.shape = (n_bootstraps, n_evals = 5)
    bs_indices = np.array([[0, 1, 2, 3, 4]] * 15 + [[2, 3, 2, 3, 2]] * 15 + [[0, 2, 4, 2, 3]] * 15)
    ranking_loss = compute_ranking_loss(rank_preds, rank_targets, bs_indices=bs_indices)

    ans = torch.tensor(
        [
            [6.0] * 15 + [0.0] * 15 + [6.0] * 15,
            [0.0] * 45,
            [2.0] * 15 + [10.0] * 15 + [3.0] * 15,
        ]
    )
    assert torch.allclose(ranking_loss, ans)


if __name__ == "__main__":
    unittest.main()
