import unittest
import numpy as np
import pandas as pd
from MassEntityCore.PeakSeries import PeakSeries

class TestPeakSeries(unittest.TestCase):

    def setUp(self):
        # Prepare test data: 2 spectra (offsets=[0,3,6])
        self.data = np.array([
            [100.0, 10.0],
            [101.0, 20.0],
            [102.0, 5.0],
            [200.0, 50.0],
            [201.0, 100.0],
            [202.0, 25.0],
        ])
        self.offsets = np.array([0, 3, 6], dtype=np.int64)
        self.metadata = pd.DataFrame({
            "formula": ["A", "B", "C", "D", "E", "F"]
        })

        self.ps = PeakSeries(self.data, self.offsets, self.metadata.copy())

    def test_len_and_counts(self):
        # Check count of spectra and peaks
        self.assertEqual(len(self.ps), 2)
        self.assertEqual(self.ps.n_all_peaks, 6)
        self.assertEqual(self.ps.n_peaks(0), 3)
        self.assertEqual(self.ps.n_peaks(1), 3)

    def test_getitem_int(self):
        # Select single spectrum by index
        sub = self.ps[0]
        self.assertEqual(len(sub), 1)
        self.assertEqual(sub.n_all_peaks, 3)
        np.testing.assert_array_equal(sub._data[:, 0], [100.0, 101.0, 102.0])

    def test_getitem_slice(self):
        # Select spectra by slice
        sub = self.ps[0:2]
        self.assertEqual(len(sub), 2)
        self.assertEqual(sub.n_all_peaks, 6)

    def test_getitem_list(self):
        # Select spectra by list of indices
        sub = self.ps[[1]]
        self.assertEqual(len(sub), 1)
        np.testing.assert_array_equal(sub._data[:, 0], [200.0, 201.0, 202.0])

    def test_normalize(self):
        # Normalize intensities using vectorized implementation
        ps_norm = self.ps.normalize(scale=1.0)
        for i in range(len(ps_norm)):
            s, e = ps_norm._offsets[i], ps_norm._offsets[i+1]
            self.assertAlmostEqual(ps_norm._data[s:e, 1].max(), 1.0)

    def test_normalize_by_for(self):
        # Normalize intensities using for-loop implementation
        ps_norm = self.ps.normalize_by_for(scale=10.0)
        for i in range(len(ps_norm)):
            s, e = ps_norm._offsets[i], ps_norm._offsets[i+1]
            self.assertAlmostEqual(ps_norm._data[s:e, 1].max(), 10.0)

    def test_sort_by_mz(self):
        # Shuffle peaks within each spectrum (segment-wise)
        shuffled = self.ps.copy()

        rng = np.random.default_rng(seed=42)  # reproducible shuffle
        for i in range(len(shuffled)):
            s, e = shuffled._offsets[i], shuffled._offsets[i+1]
            perm = rng.permutation(e - s) + s
            shuffled._data[s:e] = shuffled._data[perm]
            if shuffled._metadata is not None:
                shuffled._metadata.iloc[s:e] = shuffled._metadata.iloc[perm].values

        # Apply sort_by_mz
        sorted_ps = shuffled.sort_by_mz()

        # Check each segment
        for i in range(len(sorted_ps)):
            s, e = sorted_ps._offsets[i], sorted_ps._offsets[i+1]

            # 1. m/z must be sorted ascending
            mz_values = sorted_ps._data[s:e, 0]
            self.assertTrue(np.all(np.diff(mz_values) >= 0))

            # 2. metadata must still correspond to peaks
            for j in range(s, e):
                mz, intensity = sorted_ps._data[j]
                formula = sorted_ps._metadata.iloc[j]["formula"]
                # find the original row in self.ps (the ground truth)
                original_row = self.ps._metadata[
                    (self.ps._data[:, 0] == mz) & (self.ps._data[:, 1] == intensity)
                ]
                self.assertEqual(formula, original_row["formula"].values[0])

    def test_sorted_by_intensity(self):
        # Shuffle peaks within each spectrum (segment-wise)
        shuffled = self.ps.copy()

        rng = np.random.default_rng(seed=123)  # reproducible shuffle
        for i in range(len(shuffled)):
            s, e = shuffled._offsets[i], shuffled._offsets[i+1]
            perm = rng.permutation(e - s) + s
            shuffled._data[s:e] = shuffled._data[perm]
            if shuffled._metadata is not None:
                shuffled._metadata.iloc[s:e] = shuffled._metadata.iloc[perm].values

        # Apply sorted_by_intensity
        sorted_ps = shuffled.sorted_by_intensity(ascending=False)

        # Check each segment
        for i in range(len(sorted_ps)):
            s, e = sorted_ps._offsets[i], sorted_ps._offsets[i+1]

            # 1. intensity must be sorted descending
            intensities = sorted_ps._data[s:e, 1]
            self.assertTrue(np.all(np.diff(intensities) <= 0))

            # 2. metadata must still correspond to peaks
            for j in range(s, e):
                mz, intensity = sorted_ps._data[j]
                formula = sorted_ps._metadata.iloc[j]["formula"]
                # find the original row in self.ps (the ground truth)
                original_row = self.ps._metadata[
                    (self.ps._data[:, 0] == mz) & (self.ps._data[:, 1] == intensity)
                ]
                self.assertEqual(formula, original_row["formula"].values[0])


    def test_copy(self):
        # Ensure copy creates independent objects
        ps_copy = self.ps.copy()
        np.testing.assert_array_equal(ps_copy._data, self.ps._data)
        self.assertTrue(ps_copy._metadata.equals(self.ps._metadata))
        self.assertIsNot(ps_copy._data, self.ps._data)      # different object
        self.assertIsNot(ps_copy._metadata, self.ps._metadata)  # different object


if __name__ == "__main__":
    unittest.main()
