from __future__ import annotations
import torch
import pandas as pd
import numpy as np
from typing import overload, Tuple, Iterator, Optional, Sequence, Literal, Union, Any
from .PeakEntry import PeakEntry
from .PeakCondition import PeakCondition

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
        metadata: Optional[pd.DataFrame] = None,
        index: Optional[torch.Tensor] = None,
        is_sorted: bool = False,
        device: Optional[Union[torch.device, str]] = None,
    ):
        # setup device
        _device = torch.device(device) if device is not None else data.device
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

        self._index = self._index.to(_device)

        if is_sorted:
            self.sort_by_mz(in_place=True)

    def __len__(self) -> int:
        return self._index.numel()

    def __repr__(self):
        return f"PeakSeries(rows={len(self)}, npeaks={self.n_all_peaks})"
    
    def __iter__(self) -> Iterator[SpectrumPeaks]:
        """
        Iterate over each spectrum as a SpectrumPeaks.
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
    def mz(self) -> torch.Tensor:
        return self._data[:, 0]
    
    @mz.setter
    def mz(self, value: torch.Tensor):
        if value.shape != self.mz.shape:
            raise ValueError(f"Assigned tensor has shape {value.shape}, expected {self.mz.shape}")
        self._data[:, 0] = value

    @property
    def intensity(self) -> torch.Tensor:
        return self._data[:, 1]
    @intensity.setter
    def intensity(self, value: torch.Tensor):
        if value.shape != self.intensity.shape:
            raise ValueError(f"Assigned tensor has shape {value.shape}, expected {self.intensity.shape}")
        self._data[:, 1] = value

    @property
    def _metadata(self) -> Optional[pd.DataFrame]:
        if self._metadata_ref is None:
            return None
        parts = [self._metadata_ref.iloc[self._offsets_ref[i].item():self._offsets_ref[i+1].item()] for i in self._index]
        return pd.concat(parts, ignore_index=True) if parts else self._metadata_ref.iloc[0:0]
    
    @_metadata.setter
    def _metadata(self, value: pd.DataFrame):
        if self._metadata_ref is None:
            if value is None:
                return
            else:
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

    @property
    def length(self) -> torch.Tensor:
        lengths = self._offsets_ref[1:] - self._offsets_ref[:-1]
        return lengths[self._index]

    def n_peaks(self, index: int) -> int:
        i = self._index[index].item()
        return int(self._offsets_ref[i+1] - self._offsets_ref[i])

    @overload
    def __getitem__(self, i: int) -> SpectrumPeaks: ...
    @overload
    def __getitem__(self, i: slice) -> "PeakSeries": ...
    @overload
    def __getitem__(self, i: Sequence[int]) -> "PeakSeries": ...
    @overload
    def __getitem__(self, i: torch.Tensor) -> "PeakSeries": ...

    def __getitem__(self, i: Union[int, slice, Sequence[int], torch.Tensor]):
        """
        Indexing for PeakSeries:
        - int              → SpectrumPeaks (peaks of a single spectrum)
        - slice / sequence → PeakSeries subset
        """
        if isinstance(i, int):
            # Return SpectrumPeaks for a single spectrum
            if not (0 <= i < len(self)):
                raise IndexError(f"Index {i} out of range for {len(self)} spectra")
            return SpectrumPeaks(self, i)

        elif isinstance(i, slice):
            # Return a PeakSeries view for multiple spectra
            new_index = self._index[i]
            return PeakSeries(self._data_ref, self._offsets_ref, self._metadata_ref, index=new_index)
        
        elif isinstance(i, torch.Tensor):
            new_index = self._index[i]
            return PeakSeries(self._data_ref, self._offsets_ref, self._metadata_ref, index=new_index)
        else:
            # Convert fancy indexing into tensor of indices
            idx_tensor = torch.as_tensor(i, dtype=torch.int64, device=self.device)
            new_index = self._index[idx_tensor]
            return PeakSeries(self._data_ref, self._offsets_ref, self._metadata_ref, index=new_index)

    # --- copy materializes real data ---
    def copy(self) -> "PeakSeries":
        # materialize actual sliced data
        data = self._data.clone()
        offsets = self._offsets.clone()
        meta = None if self._metadata is None else self._metadata.copy()

        # construct fully independent PeakSeries
        return PeakSeries(data, offsets, meta, index=None, device=self.device)

    def to(self, device: Union[torch.device, str], in_place: bool = True) -> "PeakSeries":
        """
        Return a new PeakSeries with data and offsets moved to the given device.
        Metadata (pandas DataFrame) remains on CPU.
        """
        device = torch.device(device)
        data = self._data_ref.to(device)
        offsets = self._offsets_ref.to(device)
        meta = None if self._metadata_ref is None else self._metadata_ref.copy()
        index = self._index.to(device)

        if in_place:
            self._data_ref = data
            self._offsets_ref = offsets
            self._metadata_ref = meta
            self._index = index
            return self
        else:
            return PeakSeries(data, offsets, meta, index=index)

    @property
    def device(self) -> torch.device:
        return self._data_ref.device

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
            # self._metadata = meta
            return self
        else:
            return PeakSeries(
                data.clone(),               # normalized data
                self._offsets.clone(),          # offsets also copied
                None if meta is None else meta.copy(),  # metadata also copied
                index=None,                     # this is an independent copy, not a view
                is_sorted=False
            )

    def _sort_by(
        self,
        key: Literal["mz", "intensity"] = "mz",
        ascending: bool = True,
        in_place: bool = False,
        method: Optional[Literal["loop", "batch"]] = None,
        bin_size: int = 100,
        return_index: bool = False,
    ) -> "PeakSeries":
        """
        Sort peaks within each spectrum segment by a given key (mz or intensity).

        Args:
            key (str): Which column to sort by ("mz" or "intensity").
            ascending (bool): Sort order, ascending if True.
            in_place (bool): If True, update this PeakSeries in place.
            method (str|None): Sorting method:
                - "loop": per-segment loop-based implementation
                - "batch": batched vectorized implementation
            bin_size (int): Base bin size for batch sorting (default=100).
            return_index (bool): If True, also return the permutation indices.
        """
        if key == "mz":
            col = 0
        elif key == "intensity":
            col = 1
        else:
            raise ValueError(f"Unsupported key '{key}', must be 'mz' or 'intensity'.")

        data = self._data if in_place else self._data.clone()
        meta = self._metadata if in_place else (
            None if self._metadata is None else self._metadata.copy()
        )

        # Decide method
        if method is None:
            if self.device.type == "cpu":
                method = "loop"
            elif self.device.type == "cuda":
                method = "batch"
            else:
                raise ValueError(f"Unsupported device type '{self.device.type}'")
            
        perm = None

        if method == "loop":
            perm = torch.empty(data.shape[0], dtype=torch.int64)

            # Per-segment argsort on CPU
            for i in range(self._offsets.size(0) - 1):
                s, e = self._offsets[i].item(), self._offsets[i + 1].item()
                order = torch.argsort(data[s:e, col], stable=True)
                if not ascending:
                    order = torch.flip(order, dims=[0])
                perm[s:e] = order + s

            data = data[perm]
            if meta is not None:
                meta = meta.iloc[perm.tolist()].reset_index(drop=True)

        elif method == "batch":
            seg_lens = (self._offsets[1:] - self._offsets[:-1]).to(torch.long)
            n_segments = seg_lens.numel()

            # bin assignment: bin_size*2^n <= L < bin_size*2^(n+1)
            bins = torch.floor(torch.log2(torch.clamp(seg_lens.float() / bin_size, min=1))).to(torch.long)

            perm = torch.arange(data.size(0), device=data.device)

            for b in torch.unique(bins):
                seg_mask = (bins == b)
                seg_ids_bin = torch.nonzero(seg_mask, as_tuple=True)[0]
                if seg_ids_bin.numel() == 0:
                    continue

                seg_lens_bin = seg_lens[seg_ids_bin]
                max_len = seg_lens_bin.max().item()
                n_segments_bin = seg_ids_bin.numel()

                # Build mapping peak->segment
                seg_ids_all = torch.repeat_interleave(torch.arange(n_segments, device=data.device), seg_lens)
                peak_mask = torch.isin(seg_ids_all, seg_ids_bin)
                seg_ids_bin_repeat = seg_ids_all[peak_mask]
                seg_ids_bin_sorted, _ = torch.sort(seg_ids_bin)
                seg_ids_bin_local = torch.searchsorted(seg_ids_bin_sorted, seg_ids_bin_repeat)
                peak_ids_bin = perm[peak_mask]

                idx_within = torch.arange(max_len, device=data.device).repeat(n_segments_bin)
                mask = idx_within < seg_lens_bin.repeat_interleave(max_len)

                # padded matrix
                pad_val = float("inf") if ascending else float("-inf")
                value_matrix = torch.full(
                    (n_segments_bin, max_len),
                    pad_val,
                    dtype=data.dtype,
                    device=data.device,
                )
                index_matrix = torch.full(
                    (n_segments_bin, max_len),
                    -1,
                    dtype=torch.long,
                    device=data.device,
                )

                value_matrix[seg_ids_bin_local, idx_within[mask]] = data[peak_ids_bin, col]
                index_matrix[seg_ids_bin_local, idx_within[mask]] = peak_ids_bin

                # sort inside each bin
                order = torch.argsort(value_matrix, dim=1, descending=not ascending, stable=True)
                sorted_idx_bin = torch.gather(index_matrix, 1, order).reshape(-1)
                sorted_idx_bin = sorted_idx_bin[sorted_idx_bin >= 0]

                perm[peak_mask] = sorted_idx_bin

            data = data[perm]
            if meta is not None:
                meta = meta.iloc[perm.tolist()].reset_index(drop=True)

        else:
            raise ValueError(f"Invalid method '{method}'")

        if in_place:
            self._data = data
            self._metadata = meta
            return (self, perm) if return_index else self
        else:
            result = PeakSeries(
                data.clone(),
                self._offsets.clone(),
                None if meta is None else meta.copy(),
                index=None,
                is_sorted=False
            )
            return (result, perm.clone()) if return_index else result

    def sort_by_mz(
        self,
        ascending: bool = True,
        in_place: bool = False,
        method: Optional[Literal["loop", "batch"]] = None,
        bin_size: int = 100,
        return_index: bool = False,
    ) -> "PeakSeries":
        """
        Sort peaks by m/z within each spectrum segment.

        Args:
            ascending (bool): Sort order, ascending if True.
            in_place (bool): If True, update this PeakSeries in place.
            method (str|None): Sorting method ("loop" or "batch").
            bin_size (int): Base bin size for batch sorting.
            return_index (bool): If True, also return the permutation indices.
        """
        return self._sort_by(
            key="mz",
            ascending=ascending,
            in_place=in_place,
            method=method,
            bin_size=bin_size,
            return_index=return_index,
        )


    def sort_by_intensity(
        self,
        ascending: bool = False,
        in_place: bool = False,
        method: Optional[Literal["loop", "batch"]] = None,
        bin_size: int = 100,
        return_index: bool = False,
    ) -> "PeakSeries":
        """
        Sort peaks by intensity within each spectrum segment.

        Args:
            ascending (bool): Sort order, ascending if True.
            in_place (bool): If True, update this PeakSeries in place.
            method (str|None): Sorting method ("loop" or "batch").
            bin_size (int): Base bin size for batch sorting.
            return_index (bool): If True, also return the permutation indices.
        """
        return self._sort_by(
            key="intensity",
            ascending=ascending,
            in_place=in_place,
            method=method,
            bin_size=bin_size,
            return_index=return_index,
        )

        
    def reorder(self, new_order: Sequence[int]) -> "PeakSeries":
        """
        Reorder the spectra according to new_order.
        The resulting PeakSeries will have spectra in the new order,
        and the indices will be re-labeled 0..N-1.

        Args:
            new_order (Sequence[int]): Desired order of spectra (permutation of range(len(self))).

        Returns:
            PeakSeries: Reordered PeakSeries.
        """
        new_order_tensor = torch.as_tensor(new_order, dtype=torch.int64, device=self.device)

        if (set(new_order_tensor.tolist()) != set(range(len(self)))) or (len(new_order_tensor) != len(self)):
            raise ValueError(
                f"new_order must be a permutation of range(len(self)) = {list(range(len(self)))}"
            )

        # reorder the index
        new_index = self._index[new_order_tensor]

        # return a new PeakSeries with reordered index
        return PeakSeries(
            self._data_ref,
            self._offsets_ref,
            self._metadata_ref,
            index=new_index
        )
    
    def filter(self, condition: "PeakCondition") -> "PeakSeries":
        """
        Return a new PeakSeries containing only peaks that satisfy the given condition.

        Args:
            condition (PeakCondition): A condition to evaluate on this PeakSeries.

        Returns:
            PeakSeries: A new PeakSeries with only the peaks that satisfy the condition.
        """
        mask = condition.evaluate(self)  # torch.BoolTensor

        if mask.numel() != self._data.shape[0]:
            raise ValueError(
                f"Condition returned mask of shape {mask.shape}, "
                f"expected {self._data.shape[0]}"
            )

        # apply mask to peaks
        filtered_data = self._data[mask]

        # recompute offsets
        spectrum_ids = torch.repeat_interleave(
            torch.arange(len(self), device=mask.device),
            self._offsets[1:] - self._offsets[:-1]
        )
        kept_counts = torch.bincount(
            spectrum_ids[mask], minlength=len(self)
        )
        new_offsets = torch.zeros(len(self) + 1, dtype=torch.int64, device=mask.device)
        new_offsets[1:] = torch.cumsum(kept_counts, dim=0)

        # filter metadata if available
        if self._metadata is not None:
            filtered_meta = self._metadata.iloc[mask.cpu().numpy()].reset_index(drop=True)
        else:
            filtered_meta = None

        return PeakSeries(
            filtered_data,
            new_offsets,
            filtered_meta,
            index=None,
        )



class SpectrumPeaks:
    """
    A view of peaks belonging to a single spectrum inside a PeakSeries.
    Provides convenient access to its peaks and metadata.
    """

    def __init__(self, peak_series: PeakSeries, index: int):
        self._peak_series = peak_series
        self._index = index

        # calculate segment boundaries
        self._s = self._peak_series._offsets_ref[index].item()
        self._e = self._peak_series._offsets_ref[index + 1].item()

    def __len__(self) -> int:
        """Number of peaks in this spectrum."""
        return self._e - self._s

    def __iter__(self) -> Iterator[PeakEntry]:
        """Iterate over all peaks as PeakEntry instances."""
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i: int) -> PeakEntry:
        """Return the i-th peak as PeakEntry."""
        if not (0 <= i < len(self)):
            raise IndexError(f"Index {i} out of range for {len(self)} peaks")
        j = self._s + i
        mz, inten = self._peak_series._data_ref[j].tolist()
        meta = None
        if self._peak_series._metadata_ref is not None:
            meta = self._peak_series._metadata_ref.iloc[j]
        return PeakEntry(mz, inten, dict(meta) if meta is not None else None)
    
    def __setitem__(self, key: str, value: Union[Sequence, pd.Series, Any]):
        """
        Set a metadata column for all peaks in this spectrum.

        Args:
            key (str): Metadata column name.
            value (Sequence|pd.Series|Any): New values for the metadata column.
                If a single value is provided, it will be broadcast to all peaks.
                If a sequence or pd.Series is provided, its length must match the number of peaks.
        """
        n_peaks = len(self)
        if self._peak_series._metadata_ref is None:
            self._peak_series._metadata_ref = pd.DataFrame(index=range(self._peak_series._data_ref.shape[0]))
        if key not in self._peak_series._metadata_ref.columns:
            self._peak_series._metadata_ref[key] = pd.NA

        # Prepare the value(s) to assign
        idx = range(self._s, self._e)
        if isinstance(value, (list, np.ndarray, torch.Tensor, pd.Series)):
            if len(value) != n_peaks:
                raise ValueError(
                    f"Length of values ({len(value)}) must match number of peaks in this view ({n_peaks})"
                )
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            # Assign only for indices of this view
            self._peak_series._metadata_ref.loc[idx, key] = value
        else:
            # scalar → assign only for this view
            self._peak_series._metadata_ref.loc[idx, key] = value
    
    def __str__(self) -> str:
        # collect peak rows
        rows = []
        for peak in self:
            row = {
                "mz": f"{peak.mz:.4f}",
                "intensity": f"{peak.intensity:.4f}",
            }
            if peak.metadata is not None:
                for k, v in peak.metadata.items():
                    row[str(k)] = str(v)
            rows.append(row)

        # determine all column names
        colnames = ["mz", "intensity"]
        if rows and len(rows[0]) > 2:
            extra_cols = [k for k in rows[0].keys() if k not in colnames]
            colnames.extend(extra_cols)

        # compute column widths
        widths = {c: max(len(c), max(len(r.get(c, "")) for r in rows)) for c in colnames}

        # build header
        header = "  ".join(f"{c:<{widths[c]}}" for c in colnames)

        # build rows
        body = []
        for r in rows:
            body.append("  ".join(f"{r.get(c, ''):<{widths[c]}}" for c in colnames))

        lines = [header] + body
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SpectrumPeaks(n_peaks={len(self)})"

    @property
    def device(self) -> torch.device:
        """Return the device of the underlying data tensor."""
        return self._peak_series._data_ref.device

    @property
    def data(self) -> torch.Tensor:
        """Return the raw peak tensor [n_peaks, 2]."""
        return self._peak_series._data_ref[self._s:self._e]
    
    @data.setter
    def data(self, value: torch.Tensor):
        if value.shape != (len(self), 2):
            raise ValueError(f"Assigned tensor has shape {value.shape}, expected ({len(self)}, 2)")
        self._peak_series._data_ref[self._s:self._e] = value

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        """Return the peak metadata for this spectrum (if available)."""
        if self._peak_series._metadata_ref is None:
            return None
        return self._peak_series._metadata_ref.iloc[self._s:self._e].reset_index(drop=True)

    @property
    def mz(self) -> torch.Tensor:
        """Return the m/z values of the peaks."""
        return self._peak_series._data_ref[self._s:self._e, 0]
    
    @mz.setter
    def mz(self, value: torch.Tensor):
        self._peak_series._data_ref[self._s:self._e, 0] = value

    @property
    def intensity(self) -> torch.Tensor:
        """Return the intensity values of the peaks."""
        return self._peak_series._data_ref[self._s:self._e, 1]
    
    @intensity.setter
    def intensity(self, value: torch.Tensor):
        self._peak_series._data_ref[self._s:self._e, 1] = value

    def normalize(
        self,
        scale: float = 1.0,
        in_place: bool = False
    ) -> "SpectrumPeaks":
        """
        Normalize intensities in this spectrum so that the max intensity = scale.
        """
        data = self.data.clone()
        max_val = data[:, 1].max()
        if max_val > 0:
            data[:, 1] = data[:, 1] / max_val * scale

        if in_place:
            self._peak_series._data_ref[self._s:self._e] = data
            return self
        else:
            # return a detached SpectrumPeaks over a new PeakSeries
            new_ps = PeakSeries(
                data,
                torch.tensor([0, len(data)], dtype=torch.int64, device=self.device),
                self.metadata.copy() if self.metadata is not None else None,
                index=None
            )
            return SpectrumPeaks(new_ps, 0)
        
    def sort_by_mz(
        self,
        ascending: bool = True,
        in_place: bool = False
    ) -> "SpectrumPeaks":
        """
        Sort peaks by m/z within this spectrum.
        """
        data = self.data.clone()
        order = torch.argsort(data[:, 0], stable=True)
        if not ascending:
            order = torch.flip(order, dims=[0])
        data = data[order]

        meta = self.metadata
        if meta is not None:
            meta = meta.iloc[order.cpu().numpy()].reset_index(drop=True)

        if in_place:
            self._peak_series._data_ref[self._s:self._e] = data
            if meta is not None:
                self._peak_series._metadata_ref.iloc[self._s:self._e] = meta
            return self
        else:
            # return a detached SpectrumPeaks over a new PeakSeries
            new_ps = PeakSeries(
                data,
                torch.tensor([0, len(data)], dtype=torch.int64, device=self.device),
                meta.copy() if meta is not None else None,
                index=None
            )
            return SpectrumPeaks(new_ps, 0)
    
    def sort_by_intensity(
        self,
        ascending: bool = True,
        in_place: bool = False
    ) -> "SpectrumPeaks":
        """
        Sort peaks by intensity within this spectrum.
        """
        data = self.data.clone()
        order = torch.argsort(data[:, 1], stable=True)
        if not ascending:
            order = torch.flip(order, dims=[0])
        data = data[order]

        meta = self.metadata
        if meta is not None:
            meta = meta.iloc[order.cpu().numpy()].reset_index(drop=True)

        if in_place:
            self._peak_series._data_ref[self._s:self._e] = data
            if meta is not None:
                self._peak_series._metadata_ref.iloc[self._s:self._e] = meta
            return self
        else:
            # return a detached SpectrumPeaks over a new PeakSeries
            new_ps = PeakSeries(
                data,
                torch.tensor([0, len(data)], dtype=torch.int64, device=self.device),
                meta.copy() if meta is not None else None,
                index=None
            )
            return SpectrumPeaks(new_ps, 0)
