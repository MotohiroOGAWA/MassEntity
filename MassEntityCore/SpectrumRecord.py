from __future__ import annotations
import torch
from typing import List, Iterator, Literal, Union, Any
from collections.abc import Sequence
from .PeakEntry import PeakEntry
from .SpectrumPeaks import SpectrumPeaks
from .MSDataset import MSDataset


class SpectrumRecord:
    """
    A single spectrum record (view into MSDataset).

    Provides access to spectrum-level metadata and its associated peaks.
    """

    def __init__(self, ms_dataset: MSDataset, index: int):
        self._ms_dataset = ms_dataset
        self._index = index

    # ------------------- metadata access -------------------
    def _datai(self, key: str):
        """Get metadata value at the current index."""
        assert key in self._ms_dataset._columns, f"Key '{key}' not in columns"
        return self._ms_dataset._spectrum_meta_ref.iloc[self._index][key]

    def __contains__(self, item: str) -> bool:
        return item in self._ms_dataset._columns

    def __setitem__(self, key: str, value):
        """Update metadata in place."""
        assert key in self._ms_dataset._columns, f"Key '{key}' not in columns"
        self._ms_dataset._spectrum_meta_ref.iloc[
            self._index, self._ms_dataset._spectrum_meta_ref.columns.get_loc(key)
        ] = value

    def __str__(self) -> str:
        # Format metadata
        res = ""
        max_len = max(len(k) for k in self._ms_dataset._columns)
        for d in self._ms_dataset._columns: 
            res += f"{d:<{max_len+1}}:\t{self._datai(d)}\n"
        res += '\n'
        res += str(self.peaks)
        return res

    def __repr__(self) -> str:
        res = {k: self._datai(k) for k in self._ms_dataset._columns} 
        contents = [f"{k}={v}" for k, v in res.items()]
        return f"SpectrumRecord(n_peaks={self.n_peaks}, {', '.join(contents)})"

    # ------------------- peaks access -------------------
    def __getitem__(self, key: str) -> Any:
        """
        Access spectrum-level metadata by column name.
        """
        if isinstance(key, str):
            if key in self._ms_dataset._columns:
                return self._datai(key)
            else:
                raise KeyError(f"Key '{key}' not found in spectrum metadata.")
        else:
            raise TypeError(f"Invalid index type: {type(key)}. Only str allowed.")

    # ------------------- properties -------------------
    @property
    def n_peaks(self) -> int:
        return len(self.peaks)

    @property
    def peaks(self) -> SpectrumPeaks:
        """Return peaks of this spectrum as SpectrumPeaks."""
        return SpectrumPeaks(self._ms_dataset._peak_series, self._index)

    @property
    def is_int_mz(self) -> bool:
        """Check if all m/z values are integers."""
        return torch.all(self.peaks.data[:, 0] % 1 == 0)

    # ------------------- utility -------------------
    def copy(self) -> SpectrumRecord:
        """Return an independent copy of this spectrum record."""
        peaks_copy = self.peaks.normalize(in_place=False)  # SpectrumPeaks -> PeakSeries
        return SpectrumRecord(
            MSDataset(
                self._ms_dataset._spectrum_meta_ref.iloc[[self._index]]
                .reset_index(drop=True)
                .copy(),
                peaks_copy._peak_series,  # PeakSeries inside SpectrumPeaks
                columns=self._ms_dataset._columns,
            ),
            index=0,
        )