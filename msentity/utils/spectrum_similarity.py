from __future__ import annotations
from typing import Optional, Sequence, Tuple
import torch

from ..core.MSDataset import MSDataset


def cosine_similarity_matrix(
    ds1: MSDataset,
    index1: torch.Tensor,
    ds2: MSDataset,
    index2: torch.Tensor,
    *,
    bin_width: float = 0.01,
    intensity_exponent: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Paired cosine similarity (vectorized, no Python loop).

    Matching rule:
      - peaks match if |mz1 - mz2| <= bin_width
      - unmatched peaks contribute 0 to dot
      - cosine = dot / (||v1|| * ||v2||)

    Returns:
      sim: Tensor [K]
    """

    assert index1.ndim == 1 and index2.ndim == 1
    assert index1.size(0) == index2.size(0)

    subset1 = ds1[index1]
    subset2 = ds2[index2]

    ps1 = subset1.peaks
    ps2 = subset2.peaks

    if device is None:
        device = ps1.device

    K = index1.numel()
    if K == 0:
        return torch.empty(0, device=device)

    # --------- flatten all peaks ---------
    mz1 = ps1.mz.to(device)
    it1 = ps1.intensity.to(device).float()
    mz2 = ps2.mz.to(device)
    it2 = ps2.intensity.to(device).float()

    # intensity power transform
    if intensity_exponent != 1.0:
        it1 = torch.clamp(it1, min=0.0) ** intensity_exponent
        it2 = torch.clamp(it2, min=0.0) ** intensity_exponent

    # spectrum id per peak
    spec_id1 = torch.repeat_interleave(
        torch.arange(K, device=device),
        ps1.length
    )
    spec_id2 = torch.repeat_interleave(
        torch.arange(K, device=device),
        ps2.length
    )

    # --------- compute pairwise mz diff ---------
    # broadcast
    diff = mz1[:, None] - mz2[None, :]
    mask = (diff.abs() <= bin_width)

    # only keep matches within same spectrum pair
    same_spec = spec_id1[:, None] == spec_id2[None, :]
    valid = mask & same_spec

    # dot product contributions
    dot_matrix = (it1[:, None] * it2[None, :]) * valid

    # sum per spectrum
    dot = torch.zeros(K, device=device)
    dot.scatter_add_(
        0,
        spec_id1.repeat_interleave(mz2.numel()),
        dot_matrix.reshape(-1)
    )

    # --------- compute norms ----------
    norm1 = torch.zeros(K, device=device)
    norm2 = torch.zeros(K, device=device)

    norm1.scatter_add_(0, spec_id1, it1 * it1)
    norm2.scatter_add_(0, spec_id2, it2 * it2)

    norm = torch.sqrt(norm1 * norm2).clamp(min=1e-12)

    return dot / norm

def cosine_similarity_all_pairs_matrix(
    ds1: MSDataset,
    ds2: MSDataset,
    *,
    bin_width: float = 0.01,
    intensity_exponent: float = 1.0,
    device: Optional[torch.device] = None,
    max_pairs_per_call: int = 2_000_000,
) -> torch.Tensor:
    """
    Compute full N1 x N2 cosine similarity matrix by calling the paired function:

        cosine_similarity_matrix(ds1, index1, ds2, index2)

    This implementation is memory-safe for huge n2 because it NEVER materializes
    the full idx1/idx2 arrays of length n1*n2. Instead, it generates idx1/idx2
    per chunk from flattened pair indices p.

    Returns:
        Tensor of shape [N1, N2]
    """

    # Determine number of spectra in each dataset
    n1 = int(ds1.n_rows)
    n2 = int(ds2.n_rows)

    # Return empty matrix if one dataset is empty
    if n1 == 0 or n2 == 0:
        dev = device if device is not None else ds1.peaks.device
        return torch.empty((n1, n2), dtype=torch.float32, device=dev)

    dev = device if device is not None else ds1.peaks.device

    # Allocate output similarity matrix
    S = torch.empty((n1, n2), dtype=torch.float32, device=dev)

    total_pairs = n1 * n2
    if max_pairs_per_call <= 0:
        raise ValueError("max_pairs_per_call must be a positive integer")

    start = 0
    flat_S = S.view(-1)

    while start < total_pairs:
        end = min(total_pairs, start + max_pairs_per_call)

        # Build flattened pair indices p in [start, end)
        p = torch.arange(start, end, device=dev, dtype=torch.int64)

        # Convert flattened index p -> (i, j), where p = i * n2 + j
        idx1 = torch.div(p, n2, rounding_mode="floor")
        idx2 = torch.remainder(p, n2)

        sims = cosine_similarity_matrix(
            ds1, idx1,
            ds2, idx2,
            bin_width=bin_width,
            intensity_exponent=intensity_exponent,
            device=dev,
        )

        # Scatter computed similarities into the flattened output matrix
        flat_S[p] = sims

        start = end

    return S