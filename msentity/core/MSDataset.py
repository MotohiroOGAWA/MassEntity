from __future__ import annotations
import os
import io
import torch
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import h5py
import json
from dataclasses import dataclass
from typing import overload, Optional, Sequence, Union, Any, Tuple, List, Dict, Literal
from .PeakSeries import PeakSeries
from .PeakSeries import PeakSeries, SpectrumPeaks, PeakCondition
from .SpecCondition import SpecCondition


@dataclass(frozen=True)
class MSDatasetMeta:
    description: str
    attributes: Dict[str, str]
    tags: List[str]

class MSDataset:
    # Keep a margin under Arrow's ~2GB cap
    _ARROW_BYTES_LIMIT = 2_147_483_646
    _MAX_PART_BYTES = 1_000_000_000

    def __init__(
        self,
        spectrum_meta: pd.DataFrame,
        peak_series: PeakSeries,
        columns: Optional[Sequence[str]] = None,
        description: str = "",
        attributes: Dict[str, str] = {},
        tags: List[str] = [],

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


        self._description = ''
        self.description = description

        self._attributes = {}
        self.attributes = attributes
        
        self._tags = []
        self.tags = tags

    def __repr__(self) -> str:
        return (
            f"MSDataset(n_spectra={len(self)}, "
            f"n_peaks={self._peak_series.n_all_peaks}, "
            f"columns={self._columns})"
        )

    def __len__(self) -> int:
        """Number of spectra in this dataset (rows of spectrum_meta)."""
        return len(self._peak_series)
    
    def __iter__(self):
        """
        Iterate over spectra in this MSDataset, yielding SpectrumRecord.
        Example:
            for rec in ms_dataset:
                print(rec.n_peaks, rec["Name"])
        """
        for i in range(len(self)):
            yield self[i]

    @property
    def columns(self) -> List[str]:
        """Return the list of columns in this MSDataset view."""
        return list(self._columns) if self._columns is not None else list(self._spectrum_meta_ref.columns)
    
    @columns.setter
    def columns(self, cols: Sequence[str]):
        """Set the columns to include in this MSDataset view."""
        if not all(col in self._spectrum_meta_ref.columns for col in cols):
            raise ValueError("All specified columns must be in the spectrum metadata")
        self._columns = list(cols)

    @property
    def n_rows(self) -> int:
        """Number of spectra in this MSDataset view."""
        return self.shape[0]

    @property
    def n_columns(self) -> int:
        """Number of columns in this MSDataset view."""
        return self.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """(n_spectra, n_columns) like DataFrame.shape."""
        cols = self._columns if self._columns is not None else []
        return (len(self), len(cols))
    
    @property
    def n_all_peaks(self) -> int:
        """Total number of peaks across all spectra in this MSDataset view."""
        return self._peak_series.n_all_peaks

    @property
    def description(self) -> str:
        """Get dataset description."""
        return self._description
    @description.setter
    def description(self, description: str):
        """Set dataset description."""
        if not isinstance(description, str):
            raise TypeError("Description must be a string")
        self._description = description

    @property
    def attributes(self) -> Dict[str, str]:
        """Get dataset attributes."""
        return dict(self._attributes)
    @attributes.setter
    def attributes(self, attributes: Dict[str, str]):
        """Set dataset attributes."""
        if not isinstance(attributes, dict):
            raise TypeError("Attributes must be a dictionary")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in attributes.items()):
            raise TypeError("All attribute keys and values must be strings")
        self._attributes = dict(attributes)
    def set_attribute(self, key: str, value: str) -> bool:
        """Add or update a single attribute. Returns True if new key was added."""
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("Both key and value must be strings")
        self._attributes[key] = value
        return True
    def remove_attribute(self, key: str) -> bool:
        """Remove an attribute if it exists. Returns True if removed."""
        if key in self._attributes:
            del self._attributes[key]
            return True
        return False
    def has_attribute(self, key: str) -> bool:
        """Check if an attribute key exists."""
        return key in self._attributes
    def clear_attributes(self):
        """Remove all attributes."""
        self._attributes.clear()

    @property
    def tags(self) -> List[str]:
        """Get dataset tags."""
        return list(self._tags)
    @tags.setter
    def tags(self, tags: List[str]):
        """Set dataset tags."""
        if not isinstance(tags, list):
            raise TypeError("Tags must be a list")
        if any(not isinstance(t, str) for t in tags):
            raise TypeError("All tags must be strings")
        self._tags = list(tags)
    def add_tag(self, tag: str) -> bool:
        """Add a tag if it doesn't exist. Returns True if added."""
        if not isinstance(tag, str):
            raise TypeError("Tag must be a string")
        if tag not in self._tags:
            self._tags.append(tag)
            return True
        return False
    def remove_tag(self, tag: str) -> bool:
        """Remove a tag if it exists. Returns True if removed."""
        if tag in self._tags:
            self._tags.remove(tag)
            return True
        return False
    def remove_tag_at(self, index: int) -> bool:
        """Remove a tag at a specific index. Returns True if removed."""
        if 0 <= index < len(self._tags):
            del self._tags[index]
            return True
        return False
    def has_tag(self, tag: str) -> bool:
        """Check if a tag exists."""
        return tag in self._tags
    def clear_tags(self):
        """Remove all tags."""
        self._tags.clear()

    @property
    def meta(self) -> pd.DataFrame:
        """
        Return spectrum-level metadata view (row selection by PeakSeries index,
        and restricted to the selected columns). Editing the result updates
        the original DataFrame directly.
        """
        # Row selection by index
        meta = self._spectrum_meta_ref.iloc[self._peak_series._index.tolist()]

        # Column selection (always defined)
        return meta[self._columns].reset_index(drop=True)

    @property
    def peaks(self) -> PeakSeries:
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

    def __getitem__(self, i: Union[int, slice, Sequence[int], str, torch.Tensor, np.ndarray]) -> Union[SpectrumRecord, "MSDataset", pd.Series]:
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
                columns=self._columns,
                description=self.description,
                attributes=self.attributes,
                tags=self.tags,
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
                columns=self._columns,
                description=self.description,
                attributes=self.attributes,
                tags=self.tags,
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
            self.meta.copy(),
            self._peak_series.copy(),
            columns=list(self._columns),
            description=self.description,
            attributes=self.attributes,
            tags=self.tags,
        )
    
    @property
    def device(self) -> torch.device:
        """Device of the PeakSeries data tensor."""
        return self._peak_series.device
    
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
                columns=self._columns,
                description=self.description,
                attributes=self.attributes,
                tags=self.tags,
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
            columns=self._columns,
            description=self.description,
            attributes=self.attributes,
            tags=self.tags,
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
                self._spectrum_meta_ref.iloc[filtered_peak_series._index.cpu()][self._columns].reset_index(drop=True),
                filtered_peak_series,
                columns=list(self._columns),
                description=self.description,
                attributes=self.attributes,
                tags=self.tags,
            )
        elif isinstance(condition, SpecCondition):
            mask = condition.evaluate(self)  # torch.BoolTensor of shape [n_spectra]
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1).cpu().numpy()
            return MSDataset(
                self._spectrum_meta_ref.iloc[self._peak_series._index[indices].cpu()][self._columns].reset_index(drop=True),
                self._peak_series[indices].copy(),
                columns=list(self._columns),
                description=self.description,
                attributes=self.attributes,
                tags=self.tags,
            )
        else:
            raise TypeError("Unsupported condition type")
        
    @classmethod
    def concat(cls, datasets: List["MSDataset"], device: torch.device=None, description: str = "", attributes: Dict[str, str] = {}, tags: List[str] = []) -> "MSDataset":
        """
        Concatenate multiple MSDataset objects into one.

        Args:
            datasets (List[MSDataset]): List of datasets to concatenate.

        Returns:
            MSDataset: A single merged dataset.
        """
        if not datasets:
            raise ValueError("No datasets provided for concatenation")
        
        if device is None:
            device = datasets[0].device

        datasets = [ds.copy() for ds in datasets]

        # --- concatenate spectrum-level metadata ---
        spectrum_meta = pd.concat(
            [ds._spectrum_meta_ref for ds in datasets],
            ignore_index=True
        )

        all_columns = []
        seen = set()
        for ds in datasets:
            for col in ds._columns:
                if col not in seen:
                    all_columns.append(col)
                    seen.add(col)

        # --- concatenate peak-level data ---
        data_list = [ds.peaks._data_ref for ds in datasets]
        offsets_list = [ds.peaks._offsets_ref for ds in datasets]
        peak_meta_list = [
            ds.peaks._metadata_ref
            for ds in datasets
            if ds.peaks._metadata_ref is not None
        ]

        data = torch.cat(data_list, dim=0)

        # Adjust offsets across datasets
        offsets = [0]
        peak_offset = 0
        for ds in datasets:
            seg_offsets = ds.peaks._offsets_ref[1:] + peak_offset
            offsets.extend(seg_offsets.tolist())
            peak_offset += ds.peaks._offsets_ref[-1].item()
        offsets = torch.tensor(offsets, dtype=torch.int64)

        # Concatenate peak metadata if available
        peak_meta = None
        if peak_meta_list:
            peak_meta = pd.concat(peak_meta_list, ignore_index=True)

        all_peak_columns = []
        seen = set()
        for ds in datasets:
            for col in ds.peaks.meta_columns:
                if col not in seen:
                    all_peak_columns.append(col)
                    seen.add(col)

        peak_series = PeakSeries(data, offsets, peak_meta, all_peak_columns, device=device)

        return cls(
            spectrum_meta,
            peak_series,
            columns=all_columns,
            description=description,
            attributes=attributes,
            tags=tags,
        )

    # -------------------------------------------------
    # Parquet <-> bytes helpers
    # -------------------------------------------------
    @staticmethod
    def _dump_parquet_to_bytes(df: pd.DataFrame) -> bytes:
        """Serialize a DataFrame into Parquet bytes."""
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow")
        return buf.getvalue()

    @staticmethod
    def _read_parquet_from_bytes(blob: bytes) -> pd.DataFrame:
        """Deserialize Parquet bytes into a DataFrame."""
        return pd.read_parquet(io.BytesIO(blob), engine="pyarrow")

    @staticmethod
    def _parquet_uncompressed_bytes(blob: bytes) -> int:
        """
        Estimate uncompressed bytes from Parquet metadata.

        This does NOT decode the whole table; it reads only the Parquet footer/metadata.
        """
        pf = pq.ParquetFile(io.BytesIO(blob))
        md = pf.metadata
        total = 0
        for i in range(md.num_row_groups):
            total += md.row_group(i).total_byte_size
        return int(total)

    # -------------------------------------------------
    # HDF5 bytes I/O helpers
    #   - write: always new format (uint8 1D array)
    #   - read : backward-compatible (old np.void scalar / new uint8 array)
    # -------------------------------------------------
    @staticmethod
    def _save_bytes_h5(
        grp: h5py.Group,
        name: str,
        blob: bytes,
        *,
        compression: str | None = "gzip",
        chunks: bool = True,
    ) -> None:
        """
        Save raw bytes into HDF5 as a uint8 1D dataset.

        This is the *new* recommended format.
        Any existing dataset with the same name is overwritten.
        """
        if name in grp:
            del grp[name]

        arr = np.frombuffer(memoryview(blob), dtype=np.uint8)

        kwargs = {}
        if compression is not None:
            kwargs["compression"] = compression
        if chunks:
            kwargs["chunks"] = True

        ds = grp.create_dataset(name, data=arr, **kwargs)

        # Optional attributes for debugging / inspection
        ds.attrs["bytes_format"] = "uint8_1d"
        ds.attrs["nbytes"] = int(len(blob))

    @staticmethod
    def _load_bytes_h5(grp: h5py.Group, name: str) -> bytes:
        """
        Load raw bytes from HDF5.

        Supports:
          - Old format: scalar np.void (shape == ())
          - New format: uint8 1D array
        """
        if name not in grp:
            raise KeyError(f"Dataset '{name}' not found in group '{grp.name}'")

        ds = grp[name]

        # Old format: single scalar of type np.void
        if ds.shape == () or ds.dtype.kind == "V":
            return ds[()].tobytes()

        # New format: uint8 array
        return ds[...].tobytes()

    @classmethod
    def _save_parquet_h5(
        cls,
        grp: h5py.Group,
        name: str,
        df: pd.DataFrame,
        *,
        max_part_bytes: int | None = None,
        initial_rows: int = 2_000_000,
    ) -> None:
        """
        Save a DataFrame as Parquet bytes into HDF5.

        If the serialized Parquet blob exceeds max_part_bytes, split by rows and
        save multiple parts: name__part_000, name__part_001, ...
        """
        if max_part_bytes is None:
            max_part_bytes = cls._MAX_PART_BYTES

        # --- remove existing datasets (single + parts) ---
        if name in grp:
            del grp[name]
        i = 0
        while f"{name}__part_{i:03d}" in grp:
            del grp[f"{name}__part_{i:03d}"]
            i += 1
        for k in (f"{name}__chunked", f"{name}__num_parts"):
            if k in grp.attrs:
                del grp.attrs[k]

        # --- try single-part first ---
        blob = cls._dump_parquet_to_bytes(df)
        uncomp = cls._parquet_uncompressed_bytes(blob)
        cur_bytes = max(len(blob), uncomp)
        if cur_bytes <= max_part_bytes:
            cls._save_bytes_h5(grp, name, blob)  # new format (uint8)
            grp.attrs[f"{name}__chunked"] = False
            grp.attrs[f"{name}__num_parts"] = 1
            return

        # --- split by rows with measured bytes ---
        n = len(df)
        rows = min(initial_rows, max(1, n))
        parts: list[bytes] = []

        start = 0
        while start < n:
            end = min(n, start + rows)
            chunk_df = df.iloc[start:end].reset_index(drop=True)

            chunk_blob = cls._dump_parquet_to_bytes(chunk_df)
            chunk_uncomp = cls._parquet_uncompressed_bytes(chunk_blob)
            chunk_bytes = max(len(chunk_blob), chunk_uncomp)

            # Too big -> reduce rows and retry same start
            if chunk_bytes > max_part_bytes:
                if rows == 1:
                    raise ValueError(
                        f"Even 1 row Parquet exceeds max_part_bytes={max_part_bytes}. "
                        "A cell may contain an extremely large object."
                    )
                rows = max(1, rows // 2)
                continue

            parts.append(chunk_blob)
            start = end

        # --- write parts ---
        for i, part_blob in enumerate(parts):
            cls._save_bytes_h5(grp, f"{name}__part_{i:03d}", part_blob)

        grp.attrs[f"{name}__chunked"] = True
        grp.attrs[f"{name}__num_parts"] = len(parts)

    @classmethod
    def _load_parquet_h5(
        cls,
        grp: h5py.Group,
        name: str,
    ) -> pd.DataFrame:
        """
        Load a DataFrame stored as Parquet bytes in HDF5.

        Supports:
          - chunked format: name__part_000, name__part_001, ...
          - single dataset: name (old np.void or new uint8)
        """
        # --- chunked path ---
        if bool(grp.attrs.get(f"{name}__chunked", False)):
            num_parts = int(grp.attrs.get(f"{name}__num_parts", 0))
            if num_parts <= 0:
                raise ValueError(f"Invalid num_parts for '{name}' in '{grp.name}'")

            dfs: list[pd.DataFrame] = []
            for i in range(num_parts):
                blob = cls._load_bytes_h5(grp, f"{name}__part_{i:03d}")
                dfs.append(cls._read_parquet_from_bytes(blob))
            return pd.concat(dfs, ignore_index=True)

        # --- single dataset path (old/new) ---
        blob = cls._load_bytes_h5(grp, name)

        # If this is a single blob > 2GB, pyarrow cannot read it safely.
        # This file must be re-saved using the chunked format.
        if len(blob) > cls._ARROW_BYTES_LIMIT:
            raise ValueError(
                f"'{name}' is stored as a single Parquet blob of {len(blob)} bytes (> ~2GB). "
                "Cannot load with pyarrow. Re-save with chunked Parquet storage."
            )

        return cls._read_parquet_from_bytes(blob)
    

    def to_hdf5(self, path: str, save_ref: bool = False, mode: Literal["w", "a"] = "w"):
        """
        Save MSDataset into a single HDF5 file.

        Parquet metadata is stored as raw bytes using the new uint8 format.
        """
        assert mode in ("w", "a"), "mode must be 'w' or 'a'"

        dataset = self if save_ref else self.copy()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with h5py.File(path, mode) as f:
            # --- top-level metadata ---
            meta_grp = f["metadata"] if "metadata" in f else f.create_group("metadata")
            meta_grp.attrs["description"] = dataset.description
            meta_grp.attrs["attributes_json"] = json.dumps(dataset.attributes)
            meta_grp.attrs["tags_json"] = json.dumps(dataset.tags)

            # --- select dataset group ---
            existing = [g for g in f.keys() if g.startswith("dataset_")]

            if mode == "a" and existing:
                next_idx = max(int(g.split("_")[1]) for g in existing) + 1
                grp = f.create_group(f"dataset_{next_idx}")
            else:
                if "dataset_0" in f:
                    del f["dataset_0"]
                grp = f.create_group("dataset_0")

            # --- peak arrays ---
            peaks_grp = grp.create_group("peaks")
            peaks_grp.create_dataset("data", data=dataset._peak_series._data_ref.cpu().numpy())
            peaks_grp.create_dataset("offsets", data=dataset._peak_series._offsets_ref.cpu().numpy())
            peaks_grp.create_dataset("index", data=dataset._peak_series._index.cpu().numpy())

            # --- peak metadata (Parquet bytes) ---
            if dataset._peak_series._metadata_ref is not None:
                self._save_parquet_h5(
                    peaks_grp,
                    "metadata_parquet",
                    dataset._peak_series._metadata_ref,
                )

                dt = h5py.string_dtype(encoding="utf-8")
                peaks_grp.create_dataset(
                    "meta_columns",
                    data=np.array(dataset._peak_series._meta_columns, dtype=dt),
                )

            # --- spectrum metadata (Parquet bytes) ---
            self._save_parquet_h5(
                grp,
                "spectrum_meta_parquet",
                dataset._spectrum_meta_ref,
            )

    @staticmethod
    def from_hdf5(
        path: str,
        device: Optional[Union[str, torch.device]] = None,
        load_peak_meta: bool = True,
    ) -> "MSDataset":
        """
        Load MSDataset from HDF5.

        Supports:
        - Old files (np.void scalar Parquet storage)
        - New files (uint8 array Parquet storage)
        - New chunked Parquet storage (name__part_000, name__part_001, ...)
        """
        with h5py.File(path, "r") as f:
            # --- global metadata ---
            description = ""
            attributes = {}
            tags = []

            if "metadata" in f:
                meta_grp = f["metadata"]
                description = meta_grp.attrs.get("description", "")
                attributes = json.loads(meta_grp.attrs.get("attributes_json", "{}"))
                tags = json.loads(meta_grp.attrs.get("tags_json", "[]"))

            dataset_groups = sorted(
                [g for g in f.keys() if g.startswith("dataset_")],
                key=lambda g: int(g.split("_")[1]),
            )
            if not dataset_groups:
                raise ValueError(f"No dataset groups found in {path}. Expected at least 'dataset_0'.")

            datasets = []

            for group_name in dataset_groups:
                grp = f[group_name]

                # --- peak arrays ---
                peaks_grp = grp["peaks"]
                data = torch.tensor(peaks_grp["data"][:], dtype=torch.float32, device=device)
                offsets = torch.tensor(peaks_grp["offsets"][:], dtype=torch.int64, device=device)
                index = torch.tensor(peaks_grp["index"][:], dtype=torch.int64, device=device)

                # --- peak metadata (optional, supports chunked) ---
                peak_meta = None
                if load_peak_meta:
                    # Load if either:
                    # - a single dataset exists (old/new), or
                    # - chunked attributes exist
                    if ("metadata_parquet" in peaks_grp) or bool(peaks_grp.attrs.get("metadata_parquet__chunked", False)):
                        peak_meta = MSDataset._load_parquet_h5(peaks_grp, "metadata_parquet")

                # --- peak meta column names ---
                meta_columns = None
                if "meta_columns" in peaks_grp:
                    meta_columns = [
                        c.decode("utf-8") if isinstance(c, (bytes, np.bytes_)) else c
                        for c in peaks_grp["meta_columns"][:]
                    ]

                # --- spectrum metadata (supports chunked) ---
                # Load if either:
                # - a single dataset exists (old/new), or
                # - chunked attributes exist
                if ("spectrum_meta_parquet" not in grp) and (not bool(grp.attrs.get("spectrum_meta_parquet__chunked", False))):
                    raise KeyError(
                        f"Missing spectrum metadata 'spectrum_meta_parquet' (or chunked parts) in group '{grp.name}'."
                    )
                spectrum_meta = MSDataset._load_parquet_h5(grp, "spectrum_meta_parquet")

                # --- build MSDataset ---
                ps = PeakSeries(data, offsets, peak_meta, meta_columns, index=index, device=device)
                ds = MSDataset(
                    spectrum_meta,
                    ps,
                    description=description,
                    attributes=attributes,
                    tags=tags,
                )
                datasets.append(ds)

        return datasets[0] if len(datasets) == 1 else MSDataset.concat(datasets, device=device)

    @staticmethod
    def read_dataset_meta(path: str) -> MSDatasetMeta:
        """Read only top-level metadata from an MSDataset HDF5 file."""
        with h5py.File(path, "r") as f:
            if "metadata" not in f:
                raise KeyError("HDF5 does not contain '/metadata' group")

            meta_grp = f["metadata"]

            # attrs may be stored as bytes depending on h5py/HDF5 settings
            def _as_str(x) -> str:
                if x is None:
                    return ""
                if isinstance(x, (bytes, bytearray)):
                    return x.decode("utf-8")
                return str(x)

            description = _as_str(meta_grp.attrs.get("description", ""))

            attributes_json = _as_str(meta_grp.attrs.get("attributes_json", "{}"))
            tags_json = _as_str(meta_grp.attrs.get("tags_json", "[]"))

            try:
                attributes = json.loads(attributes_json) if attributes_json else {}
            except json.JSONDecodeError:
                attributes = {}
            try:
                tags = json.loads(tags_json) if tags_json else []
            except json.JSONDecodeError:
                tags = []

            if not isinstance(attributes, dict):
                attributes = {}
            else:
                attributes = {str(k): str(v) for k, v in attributes.items()}

            if not isinstance(tags, list):
                tags = []
            else:
                tags = [str(t) for t in tags]

            return MSDatasetMeta(
                description=description,
                attributes=attributes,
                tags=tags,
            )

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
        
    def __eq__(self, other: SpectrumRecord) -> bool:
        """
        Check equality between two SpectrumRecord objects.

        Two SpectrumRecords are considered equal if:
        - They are both SpectrumRecord instances
        - Their metadata (all columns) are identical
        - Their peak lists (m/z and intensity) are identical
        """
        if not isinstance(other, SpectrumRecord):
            return False

        self_meta = {k: self._datai(k) for k in self._ms_dataset._columns}
        other_meta = {k: other._datai(k) for k in other._ms_dataset._columns}

        if set(self_meta.keys()) != set(other_meta.keys()):
            return False

        for key in self_meta.keys():
            v1, v2 = self_meta[key], other_meta[key]
            if (v1 != v1 and v2 != v2):  # both NaN
                continue
            if v1 != v2:
                return False

        if self.peaks != other.peaks:
            return False

        return True

    # ------------------- properties -------------------
    @property
    def columns(self) -> List[str]:
        """Return the list of columns in the parent MSDataset."""
        return self._ms_dataset._columns.copy()
    
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

    def normalize(self, scale: float = 1.0, in_place: bool = False) -> SpectrumRecord:
        """
        Normalize intensities of this spectrum's peaks.

        Args:
            scale (float): Scale factor for normalization (default=1.0).
            in_place (bool): If True, modify peaks in place. If False, return a new SpectrumPeaks.

        Returns:
            SpectrumRecord: Normalized spectrum (if in_place=False).
        """
        if in_place:
            data = self
        else:
            data = self.copy()
        data.peaks.normalize(scale=scale, in_place=True)
        
        if in_place:
            return self
        else:
            return data
    
    def sort_by_mz(self, ascending: bool = True, in_place: bool = False) -> SpectrumRecord:
        """
        Sort peaks by m/z values.

        Args:
            in_place (bool): If True, sort peaks in place. If False, return a new SpectrumPeaks.

        Returns:
            SpectrumRecord: Sorted spectrum (if in_place=False).
        """
        if in_place:
            data = self
        else:
            data = self.copy()
        data.peaks.sort_by_mz(ascending=ascending, in_place=True)
        
        if in_place:
            return self
        else:
            return data
        
    def sort_by_intensity(self, ascending: bool = False, in_place: bool = False) -> SpectrumRecord:
        """
        Sort peaks by intensity values.

        Args:
            in_place (bool): If True, sort peaks in place. If False, return a new SpectrumPeaks.

        Returns:
            SpectrumRecord: Sorted spectrum (if in_place=False).
        """
        if in_place:
            data = self
        else:
            data = self.copy()
        data.peaks.sort_by_intensity(ascending=ascending, in_place=True)
        
        if in_place:
            return self
        else:
            return data

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
                description=self._ms_dataset.description,
                attributes=self._ms_dataset.attributes,
                tags=self._ms_dataset.tags,
            ),
            index=0,
        )