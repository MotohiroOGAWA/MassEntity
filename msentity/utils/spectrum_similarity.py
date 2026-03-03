from __future__ import annotations
from typing import Optional, Sequence, Tuple
import torch
from tqdm import tqdm

from ..core.MSDataset import MSDataset


def cosine_similarity_pair(
    ds1,
    index1: torch.Tensor,
    ds2,
    index2: torch.Tensor,
    *,
    bin_width: float = 0.01,
    intensity_exponent: float = 1.0,
    max_cum_peaks: int = 200_000,
    device: Optional[torch.device] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Paired cosine similarity computed in chunks.

    This function splits the (index1, index2) pairs into chunks so that the
    cumulative number of peaks in *either* ds1 or ds2 does not exceed max_cum_peaks.
    Each chunk is computed by _cosine_similarity_pair_core and then concatenated
    back into a single [K] tensor.
    """
    assert index1.ndim == 1 and index2.ndim == 1
    assert index1.numel() == index2.numel()

    K = index1.numel()
    if K == 0:
        return torch.empty(0, device=device or torch.device("cpu"))

    if device is None:
        device = ds1.peaks.device

    ds1 = ds1.to(device=device)
    ds2 = ds2.to(device=device)

    # Peak counts per paired spectrum (on CPU for cheap cumsum)
    len1 = ds1[index1].peaks.length.to("cpu", dtype=torch.long)
    len2 = ds2[index2].peaks.length.to("cpu", dtype=torch.long)

    out = torch.empty(K, device=device, dtype=torch.float32)

    start = 0
    if show_progress:
        pbar = tqdm(total=K, desc="Computing paired cosine similarity")
    else:
        pbar = None

    while start < K:
        # Build cumulative peak counts starting from `start`
        c1 = torch.cumsum(len1[start:], dim=0)
        c2 = torch.cumsum(len2[start:], dim=0)

        # Find the first position where either cumulative sum exceeds the budget
        exceed = (c1 > max_cum_peaks) | (c2 > max_cum_peaks)

        if torch.any(exceed):
            # first_exceed is the offset (>=0) where it first becomes True
            first_exceed = int(torch.nonzero(exceed, as_tuple=False)[0].item())
            # Use the range that stays within budget; ensure at least one pair
            end = start + max(1, first_exceed)
        else:
            end = K

        # Compute this chunk using the core routine (no chunking inside)
        sim_chunk = _cosine_similarity_pair_core(
            ds1,
            index1[start:end],
            ds2,
            index2[start:end],
            bin_width=bin_width,
            intensity_exponent=intensity_exponent,
            device=device,
        )
        out[start:end] = sim_chunk

        if pbar is not None:
            pbar.update(end - start)
            
        start = end


    if pbar is not None:
        pbar.close()

    return out


def _cosine_similarity_pair_core(
    ds1,
    index1: torch.Tensor,
    ds2,
    index2: torch.Tensor,
    *,
    bin_width: float,
    intensity_exponent: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Core paired cosine similarity (vectorized, no Python loop over pairs).

    Notes
    -----
    This implementation forms an (n_peaks1 x n_peaks2) broadcasted matrix,
    so it must only be called on chunks that fit in memory.
    """
    assert index1.ndim == 1 and index2.ndim == 1
    assert index1.numel() == index2.numel()

    subset1 = ds1[index1]
    subset2 = ds2[index2]

    ps1 = subset1.peaks
    ps2 = subset2.peaks

    K = index1.numel()
    if K == 0:
        return torch.empty(0, device=device)

    # Flatten all peaks
    mz1 = ps1.mz.to(device)
    it1 = ps1.intensity.to(device).float()
    mz2 = ps2.mz.to(device)
    it2 = ps2.intensity.to(device).float()

    # Intensity power transform
    if intensity_exponent != 1.0:
        it1 = torch.clamp(it1, min=0.0) ** intensity_exponent
        it2 = torch.clamp(it2, min=0.0) ** intensity_exponent

    # Spectrum id per peak within the chunk
    spec_id1 = torch.repeat_interleave(
        torch.arange(K, device=device),
        ps1.length.to(device)
    )
    spec_id2 = torch.repeat_interleave(
        torch.arange(K, device=device),
        ps2.length.to(device)
    )

    # Pairwise m/z difference (broadcast)
    diff = mz1[:, None] - mz2[None, :]
    mask = diff.abs() <= bin_width

    # Only keep matches within the same paired spectrum
    same_spec = spec_id1[:, None] == spec_id2[None, :]
    valid = mask & same_spec

    # Dot-product contributions
    dot_matrix = (it1[:, None] * it2[None, :]) * valid

    # Sum dot per spectrum pair
    dot = torch.zeros(K, device=device)
    dot.scatter_add_(
        0,
        spec_id1.repeat_interleave(mz2.numel()),
        dot_matrix.reshape(-1),
    )

    # Norms per spectrum pair
    norm1 = torch.zeros(K, device=device)
    norm2 = torch.zeros(K, device=device)

    norm1.scatter_add_(0, spec_id1, it1 * it1)
    norm2.scatter_add_(0, spec_id2, it2 * it2)

    denom = torch.sqrt(norm1 * norm2).clamp(min=1e-12)
    return dot / denom

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

        sims = cosine_similarity_pair(
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