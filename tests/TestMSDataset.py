import unittest
import torch
import pandas as pd
from MassEntityCore.PeakSeries import PeakSeries
from MassEntityCore.MSDataset import MSDataset


class TestMSDataset(unittest.TestCase):

    def setUp(self):
        # --- Prepare PeakSeries test data ---
        # Spectrum 1: 3 peaks, Spectrum 2: 4 peaks, Spectrum 3: 2 peaks (total 9 peaks)
        data = torch.tensor([
            [100.0, 10.0],
            [101.0, 20.0],
            [102.0, 15.0],

            [200.0, 30.0],
            [201.0, 40.0],
            [202.0, 35.0],
            [203.0, 45.0],

            [300.0, 55.0],
            [301.0, 65.0],
        ], dtype=torch.float32)

        offsets = torch.tensor([0, 3, 7, 9], dtype=torch.int64)

        metadata = pd.DataFrame({
            "peak_id": list(range(1, 10)),
            "note": [f"p{i}" for i in range(1, 10)]
        })

        self.peak_series = PeakSeries(data, offsets, metadata)

        # --- Prepare spectrum-level metadata (3 spectra, multiple columns) ---
        spectrum_meta = pd.DataFrame({
            "spectrum_id": ["s1", "s2", "s3"],
            "group": ["A", "B", "A"],
            "rt": [1.23, 2.34, 3.45],        # retention time
            "intensity_sum": [45.0, 150.0, 120.0]
        })

        # Initialize MSDataset
        self.ds = MSDataset(spectrum_meta, self.peak_series)

    def test_len_and_shape(self):
        # Check dataset length and shape
        self.assertEqual(len(self.ds), 3)
        self.assertEqual(self.ds.shape, (3, 4))
        self.assertIn("MSDataset", repr(self.ds))

    def test_spectrum_meta_view(self):
        # Check spectrum metadata view
        meta = self.ds.meta_copy
        self.assertEqual(meta.shape, (3, 4))
        self.assertListEqual(meta["spectrum_id"].tolist(), ["s1", "s2", "s3"])
        self.assertListEqual(meta["group"].tolist(), ["A", "B", "A"])

    # def test_spectrum_meta_setter(self):
    #     # Replace spectrum metadata and check propagation to reference
    #     new_meta = pd.DataFrame({
    #         "spectrum_id": ["x1", "x2", "x3"],
    #         "group": ["C", "D", "E"],
    #         "rt": [9.99, 8.88, 7.77],
    #         "intensity_sum": [999.0, 888.0, 777.0]
    #     })
    #     self.ds.meta_copy = new_meta
    #     self.assertListEqual(
    #         self.ds._spectrum_meta_ref["spectrum_id"].tolist(),
    #         ["x1", "x2", "x3"]
    #     )

    def test_getitem_subset(self):
        # Take subset (only spectrum 2)
        sub_ds = self.ds[1]
        self.assertEqual(len(sub_ds), 1)
        self.assertEqual(sub_ds.meta_copy.iloc[0]["spectrum_id"], "s2")
        self.assertEqual(sub_ds.peak_series.n_all_peaks, 4)

    def test_copy_independence(self):
        # Copy should create independent objects
        ds_copy = self.ds.copy()
        self.assertIsNot(ds_copy._spectrum_meta_ref, self.ds._spectrum_meta_ref)
        self.assertIsNot(ds_copy._peak_series._data_ref, self.ds._peak_series._data_ref)

        # # Editing copy must not affect the original
        # ds_copy.meta_copy.iloc[0, 0] = "zzz"
        # self.assertNotEqual(ds_copy.meta_copy.iloc[0, 0], self.ds.meta_copy.iloc[0, 0])


if __name__ == "__main__":
    unittest.main()
