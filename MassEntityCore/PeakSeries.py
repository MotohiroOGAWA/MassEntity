import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Iterator, Any, Literal, Union, Optional
from collections.abc import Sequence
from .PeakEntry import PeakEntry


class PeakSeries:
    """
    Represents a series of mass spectral peaks.
    """
    def __init__(self, data: np.ndarray, offsets: np.ndarray, metadata: Optional[pd.DataFrame], is_sorted: bool = False):
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        assert data.ndim == 2 and data.shape[1] == 2, "data must be a 2D array with shape (n_peaks, 2)"
        assert isinstance(offsets, np.ndarray), "offsets must be a numpy array"
        assert offsets.ndim == 1, "offsets must be a 1D array"
        assert offsets.dtype == np.int64, "offsets must be an array of int64"
        if metadata is not None:
            assert isinstance(metadata, pd.DataFrame), "metadata must be a pandas DataFrame"
            assert metadata.shape[0] == data.shape[0], "metadata must have the same number of rows as data"

        self._data = np.array([[mz, intensity] for mz, intensity in data])

        if offsets[0] != 0:
            offsets = np.insert(offsets, 0, 0)
        elif offsets[-1] != data.shape[0]:
            offsets = np.append(offsets, data.shape[0])
        self._offsets = offsets
        self._metadata = metadata
        
        if is_sorted:
            self.sort_by_mz(in_place=True)

    def __len__(self) -> int:
        return self.count

    def __repr__(self):
        contents = [f'\t{line}' for line in str(self).splitlines()]
        content = "\n".join(contents)
        return f"PeakSeries(rows={len(self)}, npeaks={self.n_all_peaks},\n{content}\n)"
    
    def __getitem__(self, i: int | slice | Sequence[int] | np.ndarray) -> "PeakSeries":
        """
        Select spectra by index and return a new PeakSeries consisting of those spectra.
        Supports:
        - int (single spectrum)
        - slice (e.g., 3:7, 8:2:-1)
        - Sequence[int] / np.ndarray[int]
        - boolean np.ndarray with shape == (#spectra,)
        """
        S = self._offsets.size - 1  # number of spectra

        # ---------- normalize i to idx: np.ndarray of spectrum indices ----------
        if isinstance(i, int):
            if i < 0:
                i += S
            if not (0 <= i < S):
                raise IndexError(f"Index {i} out of range (0..{S-1})")
            idx = np.array([i], dtype=int)

        elif isinstance(i, slice):
            start, stop, step = i.indices(S)
            if step == 1:
                # ---- fast path: contiguous slice → O(1) cut via offsets ----
                s = int(self._offsets[start]); e = int(self._offsets[stop])
                new_data = self._data[s:e]  # view
                new_offsets = (self._offsets[start:stop+1] - self._offsets[start]).astype(self._offsets.dtype)
                new_meta = None if self._metadata is None else self._metadata.iloc[s:e].reset_index(drop=True)
                return PeakSeries(new_data, new_offsets, new_meta, is_sorted=False)
            else:
                idx = np.arange(start, stop, step, dtype=int)

        elif isinstance(i, np.ndarray):
            if i.dtype == bool:
                if i.size != S:
                    raise ValueError(f"Boolean mask length {i.size} != #spectra {S}")
                idx = np.nonzero(i)[0].astype(int)
            else:
                idx = i.astype(int)

        elif isinstance(i, Sequence) and not isinstance(i, (str, bytes)):
            idx = np.asarray(i, dtype=int)

        else:
            raise TypeError("Index must be int | slice | Sequence[int] | np.ndarray")

        # support negative indices in sequences
        idx[idx < 0] += S
        if (idx < 0).any() or (idx >= S).any():
            raise IndexError("One or more indices out of range")

        # ---------- build new concatenated PeakSeries for (possibly) non-contiguous idx ----------
        if idx.size == 0:
            # empty selection
            empty_data = self._data[0:0]
            empty_offsets = np.array([0], dtype=self._offsets.dtype)
            empty_meta = None if self._metadata is None else self._metadata.iloc[0:0].copy()
            return PeakSeries(empty_data, empty_offsets, empty_meta, is_sorted=False)

        lengths = (self._offsets[1:] - self._offsets[:-1])[idx]
        new_offsets = np.empty(idx.size + 1, dtype=self._offsets.dtype)
        new_offsets[0] = 0
        np.cumsum(lengths, out=new_offsets[1:])

        parts = [self._data[int(self._offsets[j]):int(self._offsets[j+1])] for j in idx]
        new_data = np.vstack(parts) if parts else self._data[0:0]

        if self._metadata is None:
            new_meta = None
        else:
            meta_parts = [self._metadata.iloc[int(self._offsets[j]):int(self._offsets[j+1])] for j in idx]
            new_meta = pd.concat(meta_parts, ignore_index=True) if meta_parts else self._metadata.iloc[0:0].copy()

        return PeakSeries(new_data, new_offsets, new_meta, is_sorted=False)

        
    def __iter__(self) -> Iterator[Tuple[PeakEntry, Optional[pd.Series]]]:
        """
        Iterate over all peaks as tuples of (m/z, intensity).
        """
        for i, p in enumerate(self._data):
            peak = PeakEntry(mz=p[0], intensity=p[1])
            m = None if self._metadata is None else self._metadata.iloc[i]
            yield peak, m

    @property
    def count(self) -> int:
        return len(self._offsets) - 1

    @property
    def n_all_peaks(self) -> int:
        return self._data.shape[0]
    
    def n_peaks(self, index: int) -> int:
        """
        Return the number of peaks in the specified spectrum segment.
        """
        assert 0 <= index < len(self), f"Index {index} out of range for PeakSeries with {len(self)} peaks."
        return int(self._offsets[index+1] - self._offsets[index])

    @property
    def np(self) -> np.ndarray:
        """
        Return the underlying numpy array of m/z and intensity values.
        """
        return self._data.copy()
    
    @staticmethod
    def parse(peak_str: str) -> 'PeakSeries':
        """
        Create a PeakSeries object from a string with optional formulas.

        Supports format:
            "mz,intensity;..." or
            "mz,intensity,formula;..."

        Args:
            peak_str (str): String like "100.0,200.0,C6H12O6;150.0,300.0,C7H14O2"

        Returns:
            PeakSeries: A new PeakSeries instance with optional formulas.
        """
        assert isinstance(peak_str, str), "peak_str must be a string"

        peak_list = []
        metadata = {}
        meta_specs = []

        entries = peak_str.strip().split(";")
        for entry in entries:
            parts = entry.strip().split(",")
            assert len(parts) == 2, f"Invalid peak entry: '{entry}'"
            mz = float(parts[0])
            intensity = float(parts[1])
            peak_list.append([mz, intensity])

        data = np.array(peak_list)
        return data
    
    def normalize(
        self,
        scale: float = 1.0,
        in_place: bool = False,
        method: Literal["for", "vectorized"] = "for"
    ) -> "PeakSeries":
        """
        Normalize intensities in each spectrum so that the maximum intensity = scale.

        Args:
            scale (float): Target maximum intensity value after normalization (default=1.0).
            in_place (bool): If True, normalize self._data directly.
                             If False, return a new PeakSeries instance.
            method (Literal["for", "vectorized"]): 
                "for"        = per-spectrum loop (default, stable for large data).
                "vectorized" = NumPy vectorized implementation (faster for small data).

        Returns:
            PeakSeries: Normalized PeakSeries (if in_place=False).
        """
        data = self._data if in_place else self._data.copy()
        meta = self._metadata if in_place else (None if self._metadata is None else self._metadata.copy())

        if method == "vectorized":
            maxima = np.array([
                data[s:e, 1].max() if e > s else 1.0
                for s, e in zip(self._offsets[:-1], self._offsets[1:])
            ])
            spectrum_ids = np.repeat(np.arange(len(self)), np.diff(self._offsets))
            valid = maxima[spectrum_ids] > 0
            data[valid, 1] = data[valid, 1] / maxima[spectrum_ids][valid] * scale

        elif method == "for":
            for s, e in zip(self._offsets[:-1], self._offsets[1:]):
                if e > s:
                    max_intensity = data[s:e, 1].max()
                    if max_intensity > 0:
                        data[s:e, 1] = data[s:e, 1] / max_intensity * scale
        else:
            raise ValueError(f"Invalid method '{method}', choose 'for' or 'vectorized'.")

        if in_place:
            return self
        else:
            return PeakSeries(data, self._offsets.copy(), meta, is_sorted=False)

    def sort_by_mz(self, ascending: bool = True, in_place: bool = False) -> "PeakSeries":
        """
        Sort peaks by m/z within each spectrum segment [offsets[i]:offsets[i+1]).
        Uses a segment-wise permutation so metadata (if any) stays aligned.
        """
        P = self._data.shape[0]
        perm = np.empty(P, dtype=np.int64)

        # build a global permutation by concatenating per-segment argsort indices
        for i in range(self._offsets.size - 1):
            s, e = int(self._offsets[i]), int(self._offsets[i+1])
            order = np.argsort(self._data[s:e, 0], kind="mergesort")  # stable for equal m/z
            if not ascending:
                order = order[::-1]
            perm[s:e] = s + order

        if in_place:
            self._data[:] = self._data[perm]
            if self._metadata is not None:
                self._metadata = self._metadata.iloc[perm].reset_index(drop=True)
            return self
        else:
            new_meta = None if self._metadata is None else self._metadata.iloc[perm].reset_index(drop=True)
            return PeakSeries(self._data[perm].copy(), self._offsets.copy(), new_meta, is_sorted=False)

    def sorted_by_intensity(self, ascending: bool = True, in_place: bool = False) -> 'PeakSeries':
        P = self._data.shape[0]
        perm = np.empty(P, dtype=np.int64)

        # build a global permutation by concatenating per-segment argsort indices
        for i in range(self._offsets.size - 1):
            s, e = int(self._offsets[i]), int(self._offsets[i+1])
            order = np.argsort(self._data[s:e, 1], kind="mergesort")  # stable for equal m/z
            if not ascending:
                order = order[::-1]
            perm[s:e] = s + order

        if in_place:
            self._data[:] = self._data[perm]
            if self._metadata is not None:
                self._metadata = self._metadata.iloc[perm].reset_index(drop=True)
            return self
        else:
            new_meta = None if self._metadata is None else self._metadata.iloc[perm].reset_index(drop=True)
            return PeakSeries(self._data[perm].copy(), self._offsets.copy(), new_meta, is_sorted=False)
        
    def copy(self):
        return PeakSeries(self._data.copy(), self._offsets.copy(), self._metadata.copy() if self._metadata is not None else None)
