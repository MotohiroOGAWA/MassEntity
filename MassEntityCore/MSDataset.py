import io
import torch
import pandas as pd
import numpy as np
import h5py
from typing import overload, Optional, Sequence, Union
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
    
    @overload
    def __getitem__(self, i: int) -> "MSDataset": ...
    @overload
    def __getitem__(self, i: slice) -> "MSDataset": ...
    @overload
    def __getitem__(self, i: Sequence[int]) -> "MSDataset": ...
    @overload
    def __getitem__(self, i: str) -> pd.Series: ...

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
    
    def to_hdf5(self, path: str):
        """Save MSDataset to one HDF5 file, embedding Parquet as binary."""
        with h5py.File(path, "w") as f:
            # ---- save peak data ----
            f.create_dataset("peaks/data", data=self._peak_series._data_ref.cpu().numpy())
            f.create_dataset("peaks/offsets", data=self._peak_series._offsets_ref.cpu().numpy())

            # ---- save peak metadata (Parquet binary) ----
            if self._peak_series._metadata_ref is not None:
                buf = io.BytesIO()
                self._peak_series._metadata_ref.to_parquet(buf, engine="pyarrow")
                f.create_dataset("peaks/metadata_parquet", data=np.void(buf.getvalue()))

            # ---- save spectrum metadata (Parquet binary) ----
            buf = io.BytesIO()
            self._spectrum_meta_ref.to_parquet(buf, engine="pyarrow")
            f.create_dataset("spectrum_meta_parquet", data=np.void(buf.getvalue()))

    @staticmethod
    def from_hdf5(path: str, device: Optional[Union[str, torch.device]] = None) -> "MSDataset":
        """Load MSDataset from one HDF5 file with embedded Parquet binaries."""
        with h5py.File(path, "r") as f:
            # ---- load peak data ----
            data = torch.tensor(f["peaks/data"][:], dtype=torch.float32, device=device)
            offsets = torch.tensor(f["peaks/offsets"][:], dtype=torch.int64, device=device)

            # ---- load peak metadata ----
            peak_meta = None
            if "peaks/metadata_parquet" in f:
                buf = io.BytesIO(f["peaks/metadata_parquet"][()].tobytes())
                peak_meta = pd.read_parquet(buf, engine="pyarrow")

            # ---- load spectrum metadata ----
            buf = io.BytesIO(f["spectrum_meta_parquet"][()].tobytes())
            spectrum_meta = pd.read_parquet(buf, engine="pyarrow")

        ps = PeakSeries(data, offsets, peak_meta)
        return MSDataset(spectrum_meta, ps)