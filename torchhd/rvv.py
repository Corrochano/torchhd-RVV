"""Optional RVV extension wrapper for the torchhd library.

This module exposes the RVV-backed operations when built on a RISC-V target.
On other architectures it falls back to the equivalent PyTorch kernels so the
package remains importable and testable from a non-RISC-V development machine.
"""

from __future__ import annotations

from typing import Any

import torch


def is_available() -> bool:
    """Return True when the native RVV extension was successfully loaded."""
    try:
        import torchhd._C as _C  # type: ignore
    except ImportError:
        return False
    return hasattr(_C, "bind") and hasattr(_C, "hamming") and hasattr(_C, "query")


def _extension():
    try:
        import torchhd._C as _C  # type: ignore
    except ImportError:
        return None

    return _C


def bind(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the HDC bind operation using the RVV extension when available."""
    if x.device.type != "cpu" or y.device.type != "cpu":
        raise ValueError("RVV bindings require CPU tensors")

    ext = _extension()
    if ext is not None:
        return ext.bind(x, y)

    if x.shape != y.shape:
        raise ValueError("Inputs must have identical shapes")
    if x.dtype != y.dtype:
        raise ValueError("Inputs must have matching dtypes")

    if x.dtype == torch.bool:
        return torch.logical_xor(x, y)

    if torch.is_floating_point(x):
        raise TypeError("RVV bind fallback only supports integral and boolean tensors")

    return torch.bitwise_xor(x, y)


def hamming(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the Hamming distance between two vectors using the RVV extension when available."""
    if x.device.type != "cpu" or y.device.type != "cpu":
        raise ValueError("RVV Hamming requires CPU tensors")

    ext = _extension()
    if ext is not None:
        return ext.hamming(x, y)

    if x.shape != y.shape:
        raise ValueError("Inputs must have identical shapes")
    if x.dtype != y.dtype:
        raise ValueError("Inputs must have matching dtypes")

    if x.dtype == torch.bool:
        return torch.sum(torch.logical_xor(x, y), dtype=torch.int64)

    if torch.is_floating_point(x):
        raise TypeError("RVV hamming fallback only supports integral and boolean tensors")

    return torch.sum(x != y, dtype=torch.int64)


def query(memory: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Score all memory vectors against a query using the RVV extension when available."""
    if memory.device.type != "cpu" or query.device.type != "cpu":
        raise ValueError("RVV query requires CPU tensors")

    ext = _extension()
    if ext is not None:
        return ext.query(memory, query)

    if memory.dim() != 2:
        raise ValueError("Memory matrix must be 2D [nvec, words]")
    if query.dim() != 1:
        raise ValueError("Query vector must be 1D [words]")
    if memory.shape[1] != query.shape[0]:
        raise ValueError("Memory dimension 1 must match query size")
    if memory.dtype != query.dtype:
        raise ValueError("Inputs must have matching dtypes")

    scores = []
    for item in memory:
        scores.append(hamming(item, query))

    return torch.stack(scores, dim=0).to(dtype=torch.int64)


__all__ = ["is_available", "bind", "hamming", "query"]
