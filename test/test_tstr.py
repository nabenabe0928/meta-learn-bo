import unittest

from meta_learn_bo.tstr import TwoStageTransferWithRanking, compute_ranking_loss

import numpy as np

import torch

from utils import get_kwargs_and_observations


def test_compute_rank_weights() -> None:
    kwargs, observations = get_kwargs_and_observations(size=10)
    metadata = {}
    _, metadata["src20"] = get_kwargs_and_observations(size=20)
    _, metadata["src30"] = get_kwargs_and_observations(size=30)
    tstr = TwoStageTransferWithRanking(init_data=observations, metadata=metadata, **kwargs)
    assert getattr(tstr, "_task_weights") is not None
    assert torch.isclose(torch.sum(tstr._task_weights), torch.tensor(1.0))

    kwargs, observations = get_kwargs_and_observations(size=10)
    metadata = {}
    tstr = TwoStageTransferWithRanking(init_data=observations, metadata=metadata, **kwargs)
    assert getattr(tstr, "_task_weights") is not None
    assert torch.allclose(tstr._task_weights, torch.ones(1))


def test_compute_ranking_loss() -> None:
    """
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

    Therefore, 6 / (5 * (5 - 1) * 0.1) = 60 / 20 = 3.0

    Note that if everything is same, then the ranking loss will be zero.
    """
    bandwidth = 0.1
    rank_preds = np.array(
        [
            [0, 1, 2, 3, 4],
            [2, 3, 1, 5, 4],
        ]
    )
    rank_targets = np.array([0, 1, 2, 3, 4])
    ranking_loss = compute_ranking_loss(rank_preds, rank_targets, bandwidth=bandwidth)
    assert torch.allclose(ranking_loss, torch.tensor([0.0, 3.0], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
