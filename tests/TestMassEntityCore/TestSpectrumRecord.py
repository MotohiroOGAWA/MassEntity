import unittest
import torch
import pandas as pd

from MassEntity.MassEntityCore.PeakSeries import PeakSeries
from MassEntity.MassEntityCore.MSDataset import MSDataset, SpectrumRecord
from MassEntity.MassEntityCore.PeakSeries import PeakSeries, SpectrumPeaks


class TestSpectrumRecord(unittest.TestCase):

    def setUp(self):
        # Peak data: 2 spectra
        # Spectrum 1: 3 peaks, Spectrum 2: 2 peaks
        self.data = torch.tensor([
            [100.0, 10.0],
            [101.0, 20.0],
            [102.0, 15.0],
            [200.0, 30.0],
            [201.0, 25.0],
        ], dtype=torch.float32)

        self.offsets = torch.tensor([0, 3, 5], dtype=torch.int64)

        self.metadata = pd.DataFrame({
            "peak_id": [1, 2, 3, 4, 5],
            "note": [f"p{i}" for i in range(1, 6)]
        })

        peak_series = PeakSeries(self.data, self.offsets, self.metadata)

        # Spectrum-level metadata (2 spectra)
        spectrum_meta = pd.DataFrame({
            "spectrum_id": ["s1", "s2"],
            "group": ["A", "B"],
            "rt": [1.23, 2.34],
        })

        self.ds = MSDataset(spectrum_meta, peak_series)
        self.record1 = SpectrumRecord(self.ds, 0)
        self.record2 = SpectrumRecord(self.ds, 1)

    def test_metadata_access(self):
        self.assertEqual(self.record1["spectrum_id"], "s1")
        self.assertEqual(self.record2["group"], "B")
        self.assertTrue("rt" in self.record1)

    def test_metadata_update(self):
        self.record1["group"] = "Z"
        self.assertEqual(self.record1["group"], "Z")
        # Ensure original dataset updated
        self.assertEqual(self.ds._spectrum_meta_ref.iloc[0]["group"], "Z")

    def test_metadata_add_new_column(self):
        # Add a new column via SpectrumRecord
        self.record1["new_col"] = "abc"
        self.assertIn("new_col", self.ds._spectrum_meta_ref.columns)
        self.assertEqual(self.record1["new_col"], "abc")
        # Other rows should remain NaN
        self.assertTrue(pd.isna(self.record2["new_col"]))

    def test_str_and_repr(self):
        s = str(self.record1)
        r = repr(self.record1)
        pass

    def test_peaks_property(self):
        peaks = self.record1.peaks
        self.assertIsInstance(peaks, SpectrumPeaks)
        self.assertEqual(len(peaks), 3)

    def test_getitem_valid_key(self):
        # Access valid metadata columns
        self.assertEqual(self.record1["spectrum_id"], "s1")
        self.assertEqual(self.record1["group"], "A")
        self.assertAlmostEqual(self.record1["rt"], 1.23, places=2)

        self.assertEqual(self.record2["spectrum_id"], "s2")
        self.assertEqual(self.record2["group"], "B")

    def test_getitem_invalid_key(self):
        # Access with nonexistent column name
        with self.assertRaises(KeyError):
            _ = self.record1["nonexistent"]


    def test_invalid_key(self):
        with self.assertRaises(KeyError):
            _ = self.record1["nonexistent"]

    def test_is_int_mz(self):
        self.assertTrue(self.record1.is_int_mz)
        # Modify one m/z to float
        self.ds._peak_series._data_ref[0, 0] = 100.5
        self.assertFalse(self.record1.is_int_mz)

    def test_copy(self):
        rec_copy = self.record1.copy()
        self.assertIsInstance(rec_copy, SpectrumRecord)
        # Should not share the same metadata reference
        self.assertIsNot(rec_copy._ms_dataset._spectrum_meta_ref, self.ds._spectrum_meta_ref)
        # Editing copy metadata should not affect original
        rec_copy["spectrum_id"] = "xx"
        self.assertNotEqual(rec_copy["spectrum_id"], self.record1["spectrum_id"])


if __name__ == "__main__":
    unittest.main()
