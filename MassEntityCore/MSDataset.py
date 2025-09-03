import torch
import pandas as pd
import numpy as np
import h5py
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

    def save(self, path: str):
        """Save MSDataset (including PeakSeries) to an HDF5 file."""
        ps_copy = self._peak_series.copy()
        with h5py.File(path, "w") as f:
            # Save peak data
            f.create_dataset("peaks/data", data=ps_copy._data_ref.cpu().numpy())
            f.create_dataset("peaks/offsets", data=ps_copy._offsets_ref.cpu().numpy())

            # Save peak metadata if present
            if ps_copy._metadata_ref is not None:
                md_grp = f.create_group("peaks/metadata")
                for col in ps_copy._metadata_ref.columns:
                    arr = ps_copy._metadata_ref[col].to_numpy()
                    if arr.dtype.kind in {"U", "O"}:  # Unicode or object → convert to bytes
                        arr = arr.astype("S")  # <-- convert to byte-strings
                        md_grp.create_dataset(
                            col, data=arr, dtype=h5py.string_dtype("utf-8")
                        )
                    else:
                        md_grp.create_dataset(col, data=arr)
                md_grp.attrs["columns"] = list(ps_copy._metadata_ref.columns)
                md_grp.attrs["dtypes"] = [str(dt) for dt in ps_copy._metadata_ref.dtypes]

            # Save spectrum metadata
            sm_grp = f.create_group("spectrum_meta")
            for col in self._spectrum_meta_ref.columns:
                arr = self._spectrum_meta_ref[col].to_numpy()
                if arr.dtype.kind in {"U", "O"}:
                    arr = arr.astype("S")  # convert to byte-strings
                    sm_grp.create_dataset(
                        col, data=arr, dtype=h5py.string_dtype("utf-8")
                    )
                else:
                    sm_grp.create_dataset(col, data=arr)
            sm_grp.attrs["columns"] = list(self._spectrum_meta_ref.columns)
            sm_grp.attrs["dtypes"] = [str(dt) for dt in self._spectrum_meta_ref.dtypes]

    @staticmethod
    def load(path: str, device: Optional[Union[str, torch.device]] = None) -> "MSDataset":
        """Load MSDataset from an HDF5 file."""
        with h5py.File(path, "r") as f:
            # Load peak tensors
            data = torch.tensor(f["peaks/data"][:], dtype=torch.float32, device=device)
            offsets = torch.tensor(f["peaks/offsets"][:], dtype=torch.int64, device=device)

            # Load peak metadata if present
            peak_meta = None
            if "peaks/metadata" in f:
                md_grp = f["peaks/metadata"]
                cols = list(md_grp.attrs["columns"])
                dtypes = list(md_grp.attrs["dtypes"])
                md_dict = {}
                for col, dt in zip(cols, dtypes):
                    arr = md_grp[col][:]
                    # Decode bytes → str if needed
                    if arr.dtype.kind in {"S", "O"}:
                        arr = np.array([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
                                        for x in arr], dtype=object)
                    series = pd.Series(arr).astype(dt)
                    md_dict[col] = series
                peak_meta = pd.DataFrame(md_dict, columns=cols)

            # Load spectrum metadata
            sm_grp = f["spectrum_meta"]
            sm_cols = list(sm_grp.attrs["columns"])
            sm_dtypes = list(sm_grp.attrs["dtypes"])
            sm_dict = {}
            for col, dt in zip(sm_cols, sm_dtypes):
                arr = sm_grp[col][:]
                if arr.dtype.kind in {"S", "O"}:
                    arr = np.array([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
                                    for x in arr], dtype=object)
                series = pd.Series(arr).astype(dt)
                sm_dict[col] = series
            spectrum_meta = pd.DataFrame(sm_dict, columns=sm_cols)

        ps = PeakSeries(data, offsets, peak_meta)
        return MSDataset(spectrum_meta, ps)