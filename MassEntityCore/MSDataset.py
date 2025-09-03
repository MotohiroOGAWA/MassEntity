import torch
import pandas as pd
from typing import Optional, Sequence, Union
from .PeakSeries import PeakSeries


class MSDataset:
    def __init__(
        self,
        spectrum_meta: pd.DataFrame,
        peak_series: PeakSeries,
        columns: Optional[Sequence[str]] = None
    ):
        assert spectrum_meta.shape[0] == len(peak_series._offsets_ref) - 1, \
            "Number of spectra in metadata must match PeakSeries"
        
        if columns is None:
            columns = spectrum_meta.columns.tolist()
        assert all(col in spectrum_meta.columns for col in (columns or [])), \
            "All specified columns must be in the spectrum metadata"
        self._spectrum_meta_ref = spectrum_meta
        self._peak_series = peak_series
        self._columns = columns  # None = all columns

    @property
    def meta_copy(self) -> pd.DataFrame:
        """
        Return spectrum-level metadata view (row selection by PeakSeries index,
        and restricted to the selected columns). Editing the result updates
        the original DataFrame directly.
        """
        # Row selection by index
        meta = self._spectrum_meta_ref.iloc[self._peak_series._index.tolist()]

        # Column selection (always defined)
        return meta[self._columns]

    @property
    def peak_series(self) -> PeakSeries:
        """Return PeakSeries view."""
        return self._peak_series

    def __getitem__(self, i) -> "MSDataset":
        """Subselect spectra by index/slice."""
        return MSDataset(
            self._spectrum_meta_ref,
            self._peak_series[i],
            columns=self._columns
        )

    def copy(self) -> "MSDataset":
        """Return independent copy of both metadata and peaks."""
        return MSDataset(
            self.meta_copy.copy(),
            self._peak_series.copy(),
            columns=list(self._columns)
        )

    def __len__(self) -> int:
        """Number of spectra in this dataset (rows of spectrum_meta)."""
        return len(self._peak_series)

    @property
    def shape(self) -> tuple[int, int]:
        """(n_spectra, n_columns) like DataFrame.shape."""
        cols = self._columns if self._columns is not None else []
        return (len(self), len(cols))

    def __repr__(self) -> str:
        return (
            f"MSDataset(n_spectra={len(self)}, "
            f"n_peaks={self._peak_series.n_all_peaks}, "
            f"columns={self._columns})"
        )
