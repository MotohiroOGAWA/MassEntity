from __future__ import annotations
import io
import torch
import pandas as pd
import numpy as np
import h5py
from typing import overload, Optional, Sequence, Union, Any
from .PeakSeries import PeakSeries
from .PeakSeries import PeakSeries, SpectrumPeaks, PeakCondition
from .SpecCondition import SpecCondition


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
    def __getitem__(self, i: int) -> SpectrumRecord: ...
    @overload
    def __getitem__(self, i: slice) -> "MSDataset": ...
    @overload
    def __getitem__(self, i: Sequence[int]) -> "MSDataset": ...
    @overload
    def __getitem__(self, i: str) -> pd.Series: ...

    def __getitem__(self, i: Union[int, slice, Sequence[int], str, torch.Tensor, np.ndarray]):
        """
        Flexible indexing for MSDataset.

        - int      → return SpectrumRecord (single spectrum view)
        - str      → return pandas.Series (column of metadata for all spectra)
        - slice    → return MSDataset subset
        - list[int]/ndarray[int]/Tensor[int] → return MSDataset subset
        """
        if isinstance(i, int):
            # Return a SpectrumRecord representing a single spectrum
            if not (0 <= i < len(self)):
                raise IndexError(f"Index {i} out of range for {len(self)} spectra")
            return SpectrumRecord(self, self._peak_series._index[i].item())

        elif isinstance(i, str):
            # Return metadata column as pandas.Series
            if i not in self._columns:
                raise KeyError(f"Column '{i}' not in available columns {self._columns}")
            view = self._spectrum_meta_ref.iloc[self._peak_series._index.tolist()]
            return view[i].reset_index(drop=True)

        elif isinstance(i, (slice, Sequence, torch.Tensor, np.ndarray)):
            # Return MSDataset subset with the same column selection
            return MSDataset(
                self._spectrum_meta_ref,
                self._peak_series[i],
                columns=self._columns
            )
        elif isinstance(i, pd.Series) and i.dtype == bool:
            # Boolean mask Series
            if len(i) != len(self):
                raise ValueError(f"Boolean index length {len(i)} does not match number of spectra {len(self)}")
            mask = i.to_numpy()
            indices = np.nonzero(mask)[0]
            return MSDataset(
                self._spectrum_meta_ref,
                self._peak_series[indices],
                columns=self._columns
            )
        else:
            raise TypeError(f"Invalid index type: {type(i)}")
        
    def __setitem__(self, key: str, value: Union[Sequence, pd.Series, Any]):
        """
        Add or update a metadata column for the current MSDataset view.

        - If key exists, update values only for this view (subset).
        - If key is new, create it with NaN for all rows, then fill only this view.
        """
        n_view = len(self._peak_series)           # number of rows in this view
        idx = self._peak_series._index            # indices of this view
        n_total = len(self._spectrum_meta_ref)    # total rows in the underlying DataFrame

        # Ensure the column exists in the underlying metadata
        if key not in self._spectrum_meta_ref.columns:
            self._spectrum_meta_ref[key] = np.full(n_total, np.nan)

        # Prepare the value(s) to assign
        if isinstance(value, (list, np.ndarray, torch.Tensor, pd.Series)):
            if len(value) != n_view:
                raise ValueError(
                    f"Length of values ({len(value)}) must match number of spectra in this view ({n_view})"
                )
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            # Assign only for indices of this view
            self._spectrum_meta_ref.loc[idx, key] = value
        else:
            # scalar → assign only for this view
            self._spectrum_meta_ref.loc[idx, key] = value

        # Track column in self._columns
        if key not in self._columns:
            self._columns.append(key)

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
    
    def to(self, device: Union[torch.device, str], in_place=True) -> "MSDataset":
        """
        Move PeakSeries data and offsets to the given device.

        Metadata (pandas DataFrame) remains on CPU.

        If in_place=True, modify this MSDataset's PeakSeries in place and return self.
        If in_place=False, return a new MSDataset with PeakSeries moved to device.
        """
        if in_place:
            self._peak_series.to(device, in_place=True)
            return self
        else:
            return MSDataset(
                self._spectrum_meta_ref,
                self._peak_series.to(device, in_place=False),
                columns=self._columns
            )
    
    def sort_by(self, column: str, ascending: bool = True) -> "MSDataset":
        if column not in self._columns:
            raise KeyError(f"Column '{column}' not in available columns {self._columns}")

        # Get sorted index order based on column values
        order = self[column].sort_values(ascending=ascending).index.to_numpy()

        # Apply to PeakSeries (reorder spectra accordingly)
        return MSDataset(
            self._spectrum_meta_ref,
            self._peak_series[order],
            columns=self._columns
        )

    def filter(self, condition: Union[PeakCondition, SpecCondition]) -> "MSDataset":
        """
        Filter peaks in each spectrum based on a PeakCondition or filter spectra
        based on a SpecCondition.

        Peaks that do not satisfy the condition are removed.

        Example:
            cond = IntensityThreshold(threshold=100.0)
            filtered_ds = ds.filter(cond)
        """
        if not isinstance(condition, (PeakCondition, SpecCondition)):
            raise TypeError("condition must be an instance of PeakCondition or SpecCondition")

        if isinstance(condition, PeakCondition):
            filtered_peak_series = self._peak_series.filter(condition)
            return MSDataset(
                self._spectrum_meta_ref.iloc[filtered_peak_series._index][self._columns],
                filtered_peak_series,
                columns=list(self._columns)
            )
        elif isinstance(condition, SpecCondition):
            mask = condition.evaluate(self)  # torch.BoolTensor of shape [n_spectra]
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1).cpu().numpy()
            return MSDataset(
                self._spectrum_meta_ref.iloc[self._peak_series._index[indices]][self._columns],
                self._peak_series[indices].copy(),
                columns=list(self._columns)
            )
        else:
            raise TypeError("Unsupported condition type")

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

    def __setitem__(self, key: str, value: Any):
        """
        Update or add metadata value for this spectrum only.

        - If key exists, overwrite the value.
        - If key does not exist, create a new column and set value.
        """
        if key not in self._ms_dataset._spectrum_meta_ref.columns:
            # Add new column with None for all rows
            self._ms_dataset[key] = [None] * len(self._ms_dataset)

        # Set value for the current spectrum (row in DataFrame)
        self._ms_dataset._spectrum_meta_ref.iat[
            self._index, self._ms_dataset._spectrum_meta_ref.columns.get_loc(key)
        ] = value

        # Ensure the new column is tracked in self._columns
        if key not in self._ms_dataset._columns:
            self._ms_dataset._columns.append(key)

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