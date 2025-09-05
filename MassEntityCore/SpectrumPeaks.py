import torch
import pandas as pd
from typing import Iterator, Optional
from .PeakEntry import PeakEntry
from .PeakSeries import PeakSeries


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
        return PeakEntry(mz, inten, dict(meta))
    
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

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        """Return the peak metadata for this spectrum (if available)."""
        if self._peak_series._metadata_ref is None:
            return None
        return self._peak_series._metadata_ref.iloc[self._s:self._e].reset_index(drop=True)

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
