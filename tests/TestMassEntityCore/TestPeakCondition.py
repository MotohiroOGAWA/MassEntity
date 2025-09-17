import unittest
import torch
import pandas as pd
from MassEntity.MassEntityCore.PeakSeries import PeakSeries
from MassEntity.MassEntityCore.PeakCondition import (
    IntensityThreshold,
)


class TestPeakCondition(unittest.TestCase):
    def setUp(self):
        # 4 spectra: sizes (3, 2, 3, 2) = 10 peaks
        self.data = torch.tensor([
            # Spectrum 0 (3 peaks)
            [100.0, 10.0],
            [105.0, 50.0],
            [110.0, 5.0],

            # Spectrum 1 (2 peaks)
            [200.0, 80.0],
            [205.0, 20.0],

            # Spectrum 2 (3 peaks)
            [300.0, 5.0],
            [310.0, 90.0],
            [320.0, 15.0],

            # Spectrum 3 (2 peaks)
            [400.0, 70.0],
            [405.0, 25.0],
        ], dtype=torch.float32)

        self.offsets = torch.tensor([0, 3, 5, 8, 10], dtype=torch.int64)
        self.metadata = pd.DataFrame({"id": list(range(1, 11))})

        # Normal full PeakSeries
        self.ps = PeakSeries(self.data, self.offsets, self.metadata.copy())

        # Subset: only spectra 0 and 2
        self.sub_ps = PeakSeries(
            self.data, self.offsets, self.metadata.copy(),
            index=torch.tensor([0, 2], dtype=torch.int64)
        )

    def test_filter_on_full_series(self):
        cond = IntensityThreshold(threshold=30.0)
        filtered = self.ps.filter(cond)

        # Spectrum 0 -> keep peak 105.0 (intensity 50.0)
        # Spectrum 1 -> keep 200.0 (80.0)
        # Spectrum 2 -> keep 310.0 (90.0)
        # Spectrum 3 -> keep 400.0 (70.0)
        self.assertEqual(filtered.n_all_peaks, 4)
        expected_mz = torch.tensor([105.0, 200.0, 310.0, 400.0])
        torch.testing.assert_close(filtered.mz, expected_mz)

        # Offsets should reflect 4 spectra, each with 1 peak
        expected_offsets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        torch.testing.assert_close(filtered._offsets, expected_offsets)

    def test_filter_on_subset(self):
        cond = IntensityThreshold(threshold=30.0)
        filtered = self.sub_ps.filter(cond)

        # Subset contains only spectrum 0 and 2
        # Spectrum 0 -> keep peak 105.0 (50.0)
        # Spectrum 2 -> keep peak 310.0 (90.0)
        self.assertEqual(filtered.n_all_peaks, 2)
        expected_mz = torch.tensor([105.0, 310.0])
        torch.testing.assert_close(filtered.mz, expected_mz)

        # Offsets should reflect only 2 spectra
        expected_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
        torch.testing.assert_close(filtered._offsets, expected_offsets)

    def test_filter_with_empty_spectrum(self):
        cond = IntensityThreshold(threshold=60.0)  # keep only peaks with intensity >= 60

        filtered = self.ps.filter(cond)

        # Spectrum 0 -> all peaks < 60 → becomes empty
        # Spectrum 1 -> keep 200.0 (80.0)
        # Spectrum 2 -> keep 310.0 (90.0)
        # Spectrum 3 -> keep 400.0 (70.0)
        self.assertEqual(filtered.n_all_peaks, 3)

        expected_mz = torch.tensor([200.0, 310.0, 400.0])
        torch.testing.assert_close(filtered.mz, expected_mz)

        # Offsets should preserve empty spectrum (length 0)
        # Peak counts per spectrum = [0, 1, 1, 1]
        expected_offsets = torch.tensor([0, 0, 1, 2, 3], dtype=torch.int64)
        torch.testing.assert_close(filtered._offsets, expected_offsets)

if __name__ == "__main__":
    unittest.main()
