import unittest
import torch
import pandas as pd
from MassEntityCore.PeakSeries import PeakSeries


class TestPeakSeries(unittest.TestCase):

    def setUp(self):
        # Fixed test data: 4 spectra, variable sizes (4, 6, 3, 7) = 20 peaks
        self.data = torch.tensor([
            # Spectrum 1 (4 peaks, shuffled m/z)
            [107.0, 25.0],
            [101.0, 30.0],
            [110.0, 15.0],
            [105.0, 20.0],

            # Spectrum 2 (6 peaks, shuffled m/z)
            [212.0, 60.0],
            [205.0, 40.0],
            [215.0, 35.0],
            [210.0, 50.0],
            [208.0, 45.0],
            [218.0, 55.0],

            # Spectrum 3 (3 peaks, shuffled m/z)
            [315.0, 65.0],
            [301.0, 70.0],
            [310.0, 80.0],

            # Spectrum 4 (7 peaks, shuffled m/z)
            [412.0, 65.0],
            [402.0, 45.0],
            [415.0, 30.0],
            [408.0, 55.0],
            [420.0, 35.0],
            [405.0, 50.0],
            [418.0, 60.0],
        ], dtype=torch.float32)

        # Offsets
        self.offsets = torch.tensor([0, 4, 10, 13, 20], dtype=torch.int64)

        # Metadata (simple IDs)
        self.metadata = pd.DataFrame({
            "id": list(range(1, 21))
        })

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ps = PeakSeries(self.data, self.offsets, self.metadata.copy(), device=device)

    def test_len_and_counts(self):
        self.assertEqual(len(self.ps), 4)          # spectra count
        self.assertEqual(self.ps.n_all_peaks, 20)  # total peaks
        self.assertEqual(self.ps.n_peaks(0), 4)
        self.assertEqual(self.ps.n_peaks(1), 6)
        self.assertEqual(self.ps.n_peaks(2), 3)
        self.assertEqual(self.ps.n_peaks(3), 7)

    def test_getitem_int(self):
        sub = self.ps[2]  # spectrum 3
        self.assertEqual(len(sub), 1)
        self.assertEqual(sub.n_all_peaks, 3)
        torch.testing.assert_close(sub._data[:, 0],
                                   torch.tensor([315.0, 301.0, 310.0], device=self.ps.device))

    def test_getitem_slice(self):
        sub = self.ps[0:2]  # spectra 1+2
        self.assertEqual(len(sub), 2)
        self.assertEqual(sub.n_all_peaks, 10)

    def test_getitem_list(self):
        sub = self.ps[[1, 3]]
        self.assertEqual(len(sub), 2)
        self.assertEqual(sub.n_all_peaks, 6 + 7)

    def test_iter_returns_spectra(self):
        # Collect spectra from iterator
        spectra = list(self.ps)

        # Number of yielded PeakSeries should equal number of spectra
        self.assertEqual(len(spectra), len(self.ps))

        # Each element should be a PeakSeries
        for sp in spectra:
            self.assertIsInstance(sp, PeakSeries)

        # Check the number of peaks per spectrum matches n_peaks
        for i, sp in enumerate(spectra):
            self.assertEqual(sp.n_all_peaks, self.ps.n_peaks(i))

        # Check that the data matches between iterated PeakSeries and indexing
        for i, sp in enumerate(spectra):
            torch.testing.assert_close(sp._data, self.ps[i]._data)

        # Metadata should also match
        if self.ps._metadata is not None:
            for i, sp in enumerate(spectra):
                self.assertTrue(sp._metadata.equals(self.ps[i]._metadata))

    def test_normalize_vectorized(self):
        # --- Case 1: in_place=False (returns a new PeakSeries) ---
        ps_norm = self.ps.normalize(scale=1.0, method="vectorized", in_place=False)

        # Each segment must be normalized to have max intensity = 1.0
        for i in range(len(ps_norm)):
            s, e = ps_norm._offsets[i].item(), ps_norm._offsets[i + 1].item()
            self.assertAlmostEqual(ps_norm._data[s:e, 1].max().item(), 1.0)

        # The original PeakSeries should remain unchanged
        for i in range(len(self.ps)):
            s, e = self.ps._offsets[i].item(), self.ps._offsets[i + 1].item()
            orig_max = self.ps._data[s:e, 1].max().item()
            self.assertNotAlmostEqual(orig_max, 1.0)

        # --- Case 2: in_place=True (modifies self.ps directly) ---
        self.ps.normalize(scale=13.0, method="vectorized", in_place=True)

        # Now the original PeakSeries should also be normalized
        for i in range(len(self.ps)):
            s, e = self.ps._offsets[i].item(), self.ps._offsets[i + 1].item()
            self.assertAlmostEqual(self.ps._data[s:e, 1].max().item(), 13.0)

    def test_normalize_for(self):
        # --- Case 1: in_place=False (returns a new PeakSeries) ---
        ps_norm = self.ps.normalize(scale=1.0, method="for", in_place=False)

        # Each segment must be normalized to have max intensity = 1.0
        for i in range(len(ps_norm)):
            s, e = ps_norm._offsets[i].item(), ps_norm._offsets[i + 1].item()
            self.assertAlmostEqual(ps_norm._data[s:e, 1].max().item(), 1.0)

        # The original PeakSeries should remain unchanged
        for i in range(len(self.ps)):
            s, e = self.ps._offsets[i].item(), self.ps._offsets[i + 1].item()
            orig_max = self.ps._data[s:e, 1].max().item()
            self.assertNotAlmostEqual(orig_max, 1.0)

        # --- Case 2: in_place=True (modifies self.ps directly) ---
        self.ps.normalize(scale=13.0, method="for", in_place=True)

        # Now the original PeakSeries should also be normalized
        for i in range(len(self.ps)):
            s, e = self.ps._offsets[i].item(), self.ps._offsets[i + 1].item()
            self.assertAlmostEqual(self.ps._data[s:e, 1].max().item(), 13.0)

    def test_view_behavior_normalize(self):
        sub = self.ps[0]
        sub.normalize(scale=1.0, in_place=True)
        s, e = self.ps._offsets[0].item(), self.ps._offsets[1].item()
        self.assertAlmostEqual(self.ps._data[s:e, 1].max().item(), 1.0)

    def test_sort_by_mz(self):
        # --- Shuffle peaks within each spectrum (segment-wise) ---
        shuffled = self.ps.copy()
        rng = torch.Generator().manual_seed(42)
        for i in range(len(shuffled)):
            s, e = shuffled._offsets[i].item(), shuffled._offsets[i + 1].item()
            perm = torch.randperm(e - s, generator=rng) + s
            shuffled._data[s:e] = shuffled._data[perm]
            if shuffled._metadata is not None:
                shuffled._metadata.iloc[s:e] = shuffled._metadata.iloc[perm.tolist()].values

        # --- Case 1: in_place=False (returns a new PeakSeries) ---
        sorted_ps = shuffled.sort_by_mz(ascending=False, in_place=False)

        for i in range(len(sorted_ps)):
            s, e = sorted_ps._offsets[i].item(), sorted_ps._offsets[i + 1].item()
            mz_values = sorted_ps._data[s:e, 0]
            # Ensure ascending order
            self.assertTrue(torch.all(torch.diff(mz_values) <= 0), f"Not sorted in ascending order: {mz_values}")

        # Original shuffled should remain unsorted
        still_unsorted = False
        for i in range(len(shuffled)):
            s, e = shuffled._offsets[i].item(), shuffled._offsets[i + 1].item()
            mz_values = shuffled._data[s:e, 0]
            if not torch.all(torch.diff(mz_values) >= 0):
                still_unsorted = True
                break
        self.assertTrue(still_unsorted, "Shuffled object should remain unsorted when in_place=False")

        # --- Case 2: in_place=True (modifies shuffled directly) ---
        shuffled.sort_by_mz(ascending=True, in_place=True)

        for i in range(len(shuffled)):
            s, e = shuffled._offsets[i].item(), shuffled._offsets[i + 1].item()
            mz_values = shuffled._data[s:e, 0]
            # Ensure ascending order
            self.assertTrue(torch.all(torch.diff(mz_values) >= 0), f"Not sorted in ascending order: {mz_values}")

    def test_sorted_by_intensity(self):
        # --- Shuffle peaks within each spectrum (segment-wise) ---
        shuffled = self.ps.copy()
        rng = torch.Generator().manual_seed(123)
        for i in range(len(shuffled)):
            s, e = shuffled._offsets[i].item(), shuffled._offsets[i + 1].item()
            perm = torch.randperm(e - s, generator=rng) + s
            shuffled._data[s:e] = shuffled._data[perm]
            if shuffled._metadata is not None:
                shuffled._metadata.iloc[s:e] = shuffled._metadata.iloc[perm.tolist()].values

        # --- Case 1: in_place=False ---
        sorted_ps = shuffled.sorted_by_intensity(ascending=False, in_place=False)

        for i in range(len(sorted_ps)):
            s, e = sorted_ps._offsets[i].item(), sorted_ps._offsets[i + 1].item()
            intensities = sorted_ps._data[s:e, 1]

            # Allow tiny numerical tolerance
            diffs = torch.diff(intensities)
            self.assertTrue(torch.all(diffs <= 1e-6), f"Not sorted descending: {intensities}")

        # --- Case 2: in_place=True ---
        shuffled.sorted_by_intensity(ascending=True, in_place=True)

        for i in range(len(shuffled)):
            s, e = shuffled._offsets[i].item(), shuffled._offsets[i + 1].item()
            intensities = shuffled._data[s:e, 1]

            # Again allow small tolerance
            diffs = torch.diff(intensities)
            self.assertTrue(torch.all(diffs >= -1e-6), f"Not sorted descending: {intensities}")


    def test_view_behavior_sort(self):
        sub = self.ps[1]  # spectrum 2
        sub._data[:] = torch.flip(sub._data, dims=[0])
        if sub._metadata is not None:
            sub._metadata = sub._metadata.iloc[::-1].reset_index(drop=True)
        sub.sort_by_mz(in_place=True)
        s, e = self.ps._offsets[1].item(), self.ps._offsets[2].item()
        mz_values = self.ps._data[s:e, 0]
        self.assertTrue(torch.all(torch.diff(mz_values) >= 0))

    def test_copy(self):
        ps_copy = self.ps.copy()
        torch.testing.assert_close(ps_copy._data, self.ps._data)
        self.assertTrue(ps_copy._metadata.equals(self.ps._metadata))
        self.assertIsNot(ps_copy._data, self.ps._data)
        self.assertIsNot(ps_copy._metadata, self.ps._metadata)

    def test_copy_behavior(self):
        sub = self.ps[0].copy()
        sub.normalize(scale=1.0, in_place=True)
        s, e = self.ps._offsets[0].item(), self.ps._offsets[1].item()
        self.assertNotAlmostEqual(self.ps._data[s:e, 1].max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
