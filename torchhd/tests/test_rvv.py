import torch

import torchhd


def test_rvv_fallback_bind_and_hamming():
    x = torch.tensor([1, 0, 1, 0], dtype=torch.uint8)
    y = torch.tensor([1, 1, 0, 0], dtype=torch.uint8)

    z = torchhd.rvv.bind(x, y)
    assert torch.equal(z, torch.tensor([0, 1, 1, 0], dtype=torch.uint8))
    assert torchhd.rvv.hamming(x, y).item() == 2

    M = torch.stack([x, y])
    scores = torchhd.rvv.query(M, x)
    assert torch.equal(scores, torch.tensor([0, 2], dtype=torch.int64))
