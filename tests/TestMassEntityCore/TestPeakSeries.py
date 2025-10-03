import unittest
import torch
import pandas as pd
from msentity.core.PeakSeries import PeakSeries, SpectrumPeaks


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
        self.metadata = pd.DataFrame({"id": list(range(1, 21))})

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
        self.assertIsInstance(sub, SpectrumPeaks)
        self.assertEqual(len(sub), 3)  # 3 peaks
        torch.testing.assert_close(
            sub.data[:, 0],
            torch.tensor([315.0, 301.0, 310.0], device=self.ps.device)
        )

    def test_getitem_slice(self):
        sub = self.ps[0:2]  # spectra 1+2
        self.assertIsInstance(sub, PeakSeries)
        self.assertEqual(len(sub), 2)
        self.assertEqual(sub.n_all_peaks, 10)

    def test_getitem_list(self):
        sub = self.ps[[1, 3]]
        self.assertIsInstance(sub, PeakSeries)
        self.assertEqual(len(sub), 2)
        self.assertEqual(sub.n_all_peaks, 6 + 7)

    def test_iter_returns_spectra(self):
        spectra = list(self.ps)
        self.assertEqual(len(spectra), len(self.ps))
        for sp in spectra:
            self.assertIsInstance(sp, SpectrumPeaks)
        for i, sp in enumerate(spectra):
            self.assertEqual(len(sp), self.ps.n_peaks(i))
            torch.testing.assert_close(sp.data, self.ps[i].data)

    def test_normalize_vectorized(self):
        ps_norm = self.ps.normalize(scale=1.0, method="vectorized", in_place=False)
        for i in range(len(ps_norm)):
            s, e = ps_norm._offsets[i].item(), ps_norm._offsets[i + 1].item()
            self.assertAlmostEqual(ps_norm._data[s:e, 1].max().item(), 1.0)
        for i in range(len(self.ps)):
            s, e = self.ps._offsets[i].item(), self.ps._offsets[i + 1].item()
            orig_max = self.ps._data[s:e, 1].max().item()
            self.assertNotAlmostEqual(orig_max, 1.0)
        self.ps.normalize(scale=13.0, method="vectorized", in_place=True)
        for i in range(len(self.ps)):
            s, e = self.ps._offsets[i].item(), self.ps._offsets[i + 1].item()
            self.assertAlmostEqual(self.ps._data[s:e, 1].max().item(), 13.0)

    def test_normalize_for(self):
        ps_norm = self.ps.normalize(scale=1.0, method="for", in_place=False)
        for i in range(len(ps_norm)):
            s, e = ps_norm._offsets[i].item(), ps_norm._offsets[i + 1].item()
            self.assertAlmostEqual(ps_norm._data[s:e, 1].max().item(), 1.0)
        for i in range(len(self.ps)):
            s, e = self.ps._offsets[i].item(), self.ps._offsets[i + 1].item()
            orig_max = self.ps._data[s:e, 1].max().item()
            self.assertNotAlmostEqual(orig_max, 1.0)
        self.ps.normalize(scale=13.0, method="for", in_place=True)
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
        sorted_ps = shuffled.sort_by_intensity(ascending=False, in_place=False)

        for i in range(len(sorted_ps)):
            s, e = sorted_ps._offsets[i].item(), sorted_ps._offsets[i + 1].item()
            intensities = sorted_ps._data[s:e, 1]

            # Allow tiny numerical tolerance
            diffs = torch.diff(intensities)
            self.assertTrue(torch.all(diffs <= 1e-6), f"Not sorted descending: {intensities}")

        # --- Case 2: in_place=True ---
        shuffled.sort_by_intensity(ascending=True, in_place=True)

        for i in range(len(shuffled)):
            s, e = shuffled._offsets[i].item(), shuffled._offsets[i + 1].item()
            intensities = shuffled._data[s:e, 1]

            # Again allow small tolerance
            diffs = torch.diff(intensities)
            self.assertTrue(torch.all(diffs >= -1e-6), f"Not sorted descending: {intensities}")


    def test_view_behavior_sort(self):
        # Take spectrum 2 as a SpectrumPeaks view
        sub = self.ps[1]

        # Shuffle peaks directly in parent PeakSeries
        s, e = sub._s, sub._e
        self.ps._data[s:e] = torch.flip(self.ps._data[s:e], dims=[0])
        if self.ps._metadata_ref is not None:
            self.ps._metadata_ref.iloc[s:e] = self.ps._metadata_ref.iloc[s:e].iloc[::-1].values

        # Sort back via PeakSeries
        self.ps.sort_by_mz(in_place=True)

        # Verify that spectrum 2 is now sorted ascending by m/z
        mz_values = self.ps._data[s:e, 0]
        self.assertTrue(torch.all(torch.diff(mz_values) >= 0))

    def test_copy(self):
        ps_copy = self.ps.copy()
        torch.testing.assert_close(ps_copy._data, self.ps._data)
        self.assertTrue(ps_copy._metadata.equals(self.ps._metadata))
        self.assertIsNot(ps_copy._data, self.ps._data)
        self.assertIsNot(ps_copy._metadata, self.ps._metadata)

    def test_copy_behavior(self):
        ps_copy = self.ps.copy()
        ps_copy.normalize(scale=1.0, in_place=True)
        # Original self.ps should not be changed
        for i in range(len(self.ps)):
            s, e = self.ps._offsets[i].item(), self.ps._offsets[i + 1].item()
            orig_max = self.ps._data[s:e, 1].max().item()
            self.assertNotAlmostEqual(orig_max, 1.0)

    def test_reorder(self):
        # Reorder spectra: original indices [0,1,2,3] -> [2,0,3,1]
        order = [2, 0, 3, 1]
        reordered = self.ps.reorder(order)

        # Check 1: reordered has the same length as the original
        self.assertEqual(len(reordered), len(self.ps))

        # Check 2: number of peaks in each spectrum matches the original spectra
        for new_i, old_i in enumerate(order):
            self.assertEqual(
                reordered.n_peaks(new_i),
                self.ps.n_peaks(old_i),
                f"Mismatch in n_peaks at reordered index {new_i}"
            )

        # Check 3: internal index tensor matches the expected order
        self.assertTrue(torch.equal(reordered._index, self.ps._index[order]))

        # Check 4: invalid permutations should raise ValueError
        with self.assertRaises(ValueError):
            self.ps.reorder([0, 0, 1, 2])  # duplicate index
        with self.assertRaises(ValueError):
            self.ps.reorder([0, 1, 2])     # missing index


if __name__ == "__main__":
    unittest.main()
