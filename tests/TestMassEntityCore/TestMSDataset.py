import unittest
import os
import torch
import pandas as pd
from MassEntity.MassEntityCore.PeakSeries import PeakSeries
from MassEntity.MassEntityCore.MSDataset import MSDataset


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
        
        # Save file path
        self.test_file = "test_dataset.h5"

    def tearDown(self):
        # Clean up file if it exists
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

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

    def test_getitem_subset(self):
        # Take spectrum 2 as SpectrumRecord
        rec = self.ds[1]

        # Should be SpectrumRecord
        from MassEntity.MassEntityCore.MSDataset import SpectrumRecord
        self.assertIsInstance(rec, SpectrumRecord)

        # Metadata should match spectrum 2
        self.assertEqual(rec["spectrum_id"], "s2")
        self.assertEqual(rec["group"], "B")

        # SpectrumPeaks should contain only spectrum 2's peaks
        self.assertEqual(rec.n_peaks, 4)

        # Data consistency check (m/z values of spectrum 2)
        torch.testing.assert_close(
            rec.peaks.data[:, 0],
            torch.tensor([200.0, 201.0, 202.0, 203.0], dtype=torch.float32)
        )

    def test_copy_independence(self):
        # Copy should create independent objects
        ds_copy = self.ds.copy()
        self.assertIsNot(ds_copy._spectrum_meta_ref, self.ds._spectrum_meta_ref)
        self.assertIsNot(ds_copy._peak_series._data_ref, self.ds._peak_series._data_ref)

        # # Editing copy must not affect the original
        # ds_copy.meta_copy.iloc[0, 0] = "zzz"
        # self.assertNotEqual(ds_copy.meta_copy.iloc[0, 0], self.ds.meta_copy.iloc[0, 0])


    def test_save_and_load(self):
        for i in range(2):  # Test twice to ensure file overwrite works
            if i == 0:
                self.ds.to_hdf5(self.test_file)
                ds_loaded = MSDataset.from_hdf5(self.test_file)

            # --- Spectrum metadata check ---
            pd.testing.assert_frame_equal(
                ds_loaded.meta_copy.reset_index(drop=True),
                self.ds.meta_copy.reset_index(drop=True)
            )

            # --- PeakSeries data check ---
            torch.testing.assert_close(
                ds_loaded.peak_series._data,
                self.ds.peak_series._data
            )
            torch.testing.assert_close(
                ds_loaded.peak_series._offsets,
                self.ds.peak_series._offsets
            )

            # --- Peak metadata check ---
            if self.ds.peak_series._metadata_ref is not None:
                pd.testing.assert_frame_equal(
                    ds_loaded.peak_series._metadata_ref.reset_index(drop=True),
                    self.ds.peak_series._metadata_ref.reset_index(drop=True)
                )

            # Ensure loaded is a new object, not a reference
            self.assertIsNot(ds_loaded._spectrum_meta_ref, self.ds._spectrum_meta_ref)
            self.assertIsNot(ds_loaded.peak_series._data_ref, self.ds.peak_series._data_ref)



if __name__ == "__main__":
    unittest.main()
