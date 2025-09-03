import torch
import pandas as pd
from typing import Tuple, Iterator, Optional, Sequence, Literal
from .PeakEntry import PeakEntry

class PeakSeries:
    """
    Represents a (possibly sliced) view of mass spectral peaks.
    Holds reference to original data, offsets, and metadata,
    plus an index selecting which spectra are visible.
    """
    def __init__(
        self,
        data: torch.Tensor,
        offsets: torch.Tensor,
        metadata: Optional[pd.DataFrame],
        index: Optional[torch.Tensor] = None,
        is_sorted: bool = False,
        device: Optional[torch.device | str] = None,
    ):
        # setup device
        _device = torch.device(device) if device is not None else None
        assert isinstance(data, torch.Tensor), "data must be torch.Tensor"
        assert data.ndim == 2 and data.shape[1] == 2, "data must be shape (n_peaks, 2)"
        assert isinstance(offsets, torch.Tensor), "offsets must be torch.Tensor"
        assert offsets.ndim == 1 and offsets.dtype == torch.int64
        if metadata is not None:
            assert isinstance(metadata, pd.DataFrame)
            assert metadata.shape[0] == data.shape[0]

        self._data_ref = data if device is None else data.to(_device)         # always original
        self._offsets_ref = offsets if device is None else offsets.to(_device)  # always original
        self._metadata_ref = metadata  # always original

        if index is None:
            self._index = torch.arange(offsets.size(0) - 1, dtype=torch.int64)
        else:
            self._index = index.clone()

        if _device is not None:
            self._index = self._index.to(_device)

        if is_sorted:
            self.sort_by_mz(in_place=True)

    def __len__(self) -> int:
        return self._index.numel()

    def __repr__(self):
        return f"PeakSeries(rows={len(self)}, npeaks={self.n_all_peaks})"
    
    def __iter__(self) -> Iterator["PeakSeries"]:
        """
        Iterate over each spectrum as a PeakSeries.
        """
        for idx in range(len(self)):
            yield self[idx]

    # --- properties that map index -> actual slices ---
    @property
    def _offsets(self) -> torch.Tensor:
        # compute new offsets relative to selected spectra
        lengths = (self._offsets_ref[1:] - self._offsets_ref[:-1])[self._index]
        new_offsets = torch.empty(len(self) + 1, dtype=torch.int64, device=self.device)
        new_offsets[0] = 0
        torch.cumsum(lengths, dim=0, out=new_offsets[1:])
        return new_offsets

    @property
    def _data(self) -> torch.Tensor:
        parts = [slice(self._offsets_ref[i].item(), self._offsets_ref[i+1].item()) for i in self._index]
        return torch.cat([self._data_ref[p] for p in parts], dim=0) if parts else self._data_ref[0:0]

    @_data.setter
    def _data(self, value: torch.Tensor):
        # ensure shape consistency
        if value.shape != self._data.shape:
            raise ValueError(
                f"Assigned tensor has shape {value.shape}, expected {self._data.shape}"
            )

        # write back into the underlying reference tensor segment by segment
        offset = 0
        for i in self._index:
            s = self._offsets_ref[i].item()
            e = self._offsets_ref[i + 1].item()
            seg_len = e - s
            self._data_ref[s:e] = value[offset:offset + seg_len]
            offset += seg_len

    @property
    def _metadata(self) -> Optional[pd.DataFrame]:
        if self._metadata_ref is None:
            return None
        parts = [self._metadata_ref.iloc[self._offsets_ref[i].item():self._offsets_ref[i+1].item()] for i in self._index]
        return pd.concat(parts, ignore_index=True) if parts else self._metadata_ref.iloc[0:0]
    
    @_metadata.setter
    def _metadata(self, value: pd.DataFrame):
        if self._metadata_ref is None:
            raise AttributeError("No metadata_ref exists to update.")

        # shape consistency check
        if len(value) != len(self._metadata):
            raise ValueError(
                f"Assigned metadata has {len(value)} rows, "
                f"expected {len(self._metadata)} rows"
            )

        # write back into the underlying reference DataFrame segment by segment
        offset = 0
        parts = [
            slice(self._offsets_ref[i].item(), self._offsets_ref[i + 1].item())
            for i in self._index
        ]
        for p in parts:
            seg_len = p.stop - p.start
            self._metadata_ref.iloc[p] = value.iloc[offset:offset + seg_len]
            offset += seg_len

    @property
    def count(self) -> int:
        return len(self)

    @property
    def n_all_peaks(self) -> int:
        return int((self._offsets_ref[1:] - self._offsets_ref[:-1])[self._index].sum())

    def n_peaks(self, index: int) -> int:
        i = self._index[index].item()
        return int(self._offsets_ref[i+1] - self._offsets_ref[i])

    # --- slicing ---
    def __getitem__(self, i: int | slice | Sequence[int] | torch.Tensor) -> "PeakSeries":
        if isinstance(i, int):
            new_index = self._index[i:i+1]
        elif isinstance(i, slice):
            new_index = self._index[i]
        else:
            new_index = self._index[torch.as_tensor(i, dtype=torch.int64)]
        return PeakSeries(self._data_ref, self._offsets_ref, self._metadata_ref, index=new_index)

    # --- copy materializes real data ---
    def copy(self) -> "PeakSeries":
        # materialize actual sliced data
        data = self._data.clone()
        offsets = self._offsets.clone()
        meta = None if self._metadata is None else self._metadata.copy()

        # construct fully independent PeakSeries
        return PeakSeries(data, offsets, meta, index=None)
    
    def to(self, device: torch.device | str) -> "PeakSeries":
        """
        Return a new PeakSeries with data and offsets moved to the given device.
        Metadata (pandas DataFrame) remains on CPU.
        """
        device = torch.device(device)
        data = self._data.clone().to(device)
        offsets = self._offsets.clone().to(device)
        meta = None if self._metadata is None else self._metadata.copy()

        return PeakSeries(data, offsets, meta, index=None)
    
    @property
    def device(self) -> torch.device:
        return self._data.device

    def normalize(
        self,
        scale: float = 1.0,
        in_place: bool = False,
        method: Literal["for", "vectorized"] = "vectorized"
    ) -> "PeakSeries":
        """
        Normalize intensities in each spectrum so that the maximum intensity = scale.
        Works segment-wise according to self._offsets.

        Args:
            scale (float): Target max intensity after normalization.
            in_place (bool): Modify this object if True.
            method (str): "for" (loop) or "vectorized".

        Returns:
            PeakSeries: normalized series (self if in_place=True, otherwise a new copy).
        """
        data = self._data if in_place else self._data.clone()
        meta = self._metadata if in_place else (
            None if self._metadata is None else self._metadata.copy()
        )

        if method == "vectorized":
            # intensities of all peaks
            intensities = data[:, 1]

            # build spectrum IDs for each peak
            spectrum_ids = torch.repeat_interleave(
                torch.arange(len(self), device=data.device),
                self._offsets[1:] - self._offsets[:-1]
            )

            # compute maximum intensity per spectrum (avoid for-loop)
            maxima = torch.full((len(self),), 1.0, dtype=data.dtype, device=data.device)
            maxima = maxima.scatter_reduce(
                0, spectrum_ids, intensities, reduce="amax", include_self=False
            )

            # broadcast normalization factors to each peak
            norm_factors = maxima[spectrum_ids].clamp(min=1e-12)
            data[:, 1] = data[:, 1] / norm_factors * scale

        elif method == "for":
            for s, e in zip(self._offsets[:-1], self._offsets[1:]):
                if e > s:
                    seg = data[s:e, 1]
                    max_val = seg.max()
                    if max_val > 0:
                        seg /= max_val
                        seg *= scale
        else:
            raise ValueError(f"Invalid method '{method}'")

        if in_place:
            self._data = data
            self._metadata = meta
            return self
        else:
            return PeakSeries(
                data.clone(),               # normalized data
                self._offsets.clone(),          # offsets also copied
                None if meta is None else meta.copy(),  # metadata also copied
                index=None,                     # this is an independent copy, not a view
                is_sorted=False
            )

    def sort_by_mz(self, ascending: bool = True, in_place: bool = False) -> "PeakSeries":
        """
        Sort peaks by m/z within each spectrum segment.
        """
        data = self._data if in_place else self._data.clone()
        meta = self._metadata if in_place else (
            None if self._metadata is None else self._metadata.copy()
        )
        perm = torch.empty(data.shape[0], dtype=torch.int64)

        for i in range(self._offsets.size(0) - 1):
            s, e = self._offsets[i].item(), self._offsets[i + 1].item()
            order = torch.argsort(data[s:e, 0], stable=True)
            if not ascending:
                order = torch.flip(order, dims=[0])
            perm[s:e] = order + s

        data = data[perm]
        if meta is not None:
            meta = meta.iloc[perm.tolist()].reset_index(drop=True)

        if in_place:
            self._data = data
            self._metadata = meta
            return self
        else:
            return PeakSeries(
                data.clone(),               # normalized data
                self._offsets.clone(),          # offsets also copied
                None if meta is None else meta.copy(),  # metadata also copied
                index=None,                     # this is an independent copy, not a view
                is_sorted=False
            )

    def sorted_by_intensity(self, ascending: bool = True, in_place: bool = False) -> "PeakSeries":
        """
        Sort peaks by intensity within each spectrum segment.
        """
        data = self._data if in_place else self._data.clone()
        meta = self._metadata if in_place else (
            None if self._metadata is None else self._metadata.copy()
        )
        perm = torch.empty(data.shape[0], dtype=torch.int64)

        # Build a global permutation by concatenating per-segment argsort indices
        for i in range(self._offsets.size(0) - 1):
            s, e = self._offsets[i].item(), self._offsets[i + 1].item()
            order = torch.argsort(data[s:e, 1], stable=True)
            if not ascending:
                order = torch.flip(order, dims=[0])
            perm[s:e] = order + s

        # Apply permutation
        data = data[perm]
        if meta is not None:
            meta = meta.iloc[perm.tolist()].reset_index(drop=True)

        if in_place:
            # Update this PeakSeries in place
            self._data = data
            self._metadata = meta
            return self
        else:
            # Return a new independent PeakSeries with sorted data
            return PeakSeries(
                data.clone(),                   # sorted data
                self._offsets.clone(),          # keep offsets
                None if meta is None else meta.copy(),  # sorted metadata
                index=None,                     # independent copy (not a view)
                is_sorted=False
            )


