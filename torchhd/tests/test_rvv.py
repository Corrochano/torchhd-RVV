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


def test_map_bind_uses_rvv_when_available_on_integer_map_tensors():
    a = torch.tensor([1, -1, 1, -1], dtype=torch.int8)
    b = torch.tensor([-1, -1, 1, 1], dtype=torch.int8)

    out = torchhd.MAPTensor(a).bind(torchhd.MAPTensor(b))

    assert torch.equal(out, torch.tensor([-1, 1, 1, -1], dtype=torch.int8))
