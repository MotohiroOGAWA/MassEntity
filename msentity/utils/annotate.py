import torch
import pandas as pd
import numpy as np
from ..core.MSDataset import MSDataset

def set_spec_id(dataset:MSDataset, prefix:str = "") -> bool:
    if not isinstance(prefix, str):
        raise ValueError("Prefix must be string.")
    if 'SpecID' in dataset._spectrum_meta_ref.columns:
        print("Warning: 'SpecID' column already exists in the dataset.")
        return False
    
    n = len(dataset)
    width = len(str(n))
    spec_ids = [f"{prefix}{i+1:0{width}d}" for i in range(n)]
    dataset['SpecID'] = spec_ids
    return True


def set_peak_id(
    dataset: MSDataset,
    *,
    col_name: str = "PeakID",
    overwrite: bool = False,
    start: int = 0,
) -> bool:
    """
    Assign PeakID for each peak in each spectrum.

    PeakID policy:
      - For each spectrum: PeakID = start..start+(n_peaks-1)
      - Resets for every spectrum.

    Writes into: dataset.peaks[col_name]

    Returns:
        bool: True if assigned, False if skipped (already exists and overwrite=False)
    """
    ps = dataset.peaks

    # Ensure metadata exists
    if ps._metadata_ref is None:
        # Trigger metadata creation via __setitem__ by writing later
        pass

    # If column exists and not overwriting -> skip
    if (ps._metadata_ref is not None) and (col_name in ps._metadata_ref.columns) and (not overwrite):
        print(f"Warning: '{col_name}' column already exists in peaks metadata.")
        return False

    # offsets_ref: [n_spectra+1], int64
    offsets = ps._offsets_ref
    if not isinstance(offsets, torch.Tensor) or offsets.ndim != 1 or offsets.dtype != torch.int64:
        raise ValueError("dataset.peaks._offsets_ref must be int64 1D torch.Tensor")

    # lengths per spectrum: [n_spectra]
    lens = offsets[1:] - offsets[:-1]  # int64

    # total peaks
    total = int(lens.sum().item())
    if total == 0:
        # still create empty column
        ps[col_name] = []
        return True

    # Build PeakID vector efficiently:
    # Example: lens = [3,1,4] -> [0,1,2, 0, 0,1,2,3]
    # Using repeat_interleave to get local indices
    # local_idx = arange(total) - repeat_interleave(offsets[:-1], lens)
    peak_global = torch.arange(total, dtype=torch.int64, device=offsets.device)
    base = torch.repeat_interleave(offsets[:-1], lens)  # same device/dtype
    peak_id = peak_global - base + int(start)

    # Write back to pandas metadata via PeakSeries.__setitem__
    # convert to CPU numpy to avoid pandas issues
    ps[col_name] = peak_id.cpu().numpy()
    ps[col_name] = ps.metadata[col_name].astype(np.int64)  # ensure int64 dtype in pandas
    return True
