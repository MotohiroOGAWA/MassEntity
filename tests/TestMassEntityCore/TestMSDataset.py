import unittest
import os
import torch
import pandas as pd
import numpy as np
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

    def test_add_new_column_full_dataset(self):
        self.ds["score"] = [10, 20, 30]
        self.assertIn("score", self.ds._spectrum_meta_ref.columns)
        self.assertTrue(np.allclose(self.ds._spectrum_meta_ref["score"].values, [10, 20, 30]))

    def test_getitem_variants(self):
        # Case 1: integer index -> SpectrumRecord
        rec = self.ds[0]
        from MassEntity.MassEntityCore.MSDataset import SpectrumRecord
        self.assertIsInstance(rec, SpectrumRecord)
        self.assertEqual(rec["spectrum_id"], "s1")

        # Case 2: string index -> pandas.Series with correct length and values
        rt_series = self.ds["rt"]
        self.assertIsInstance(rt_series, pd.Series)
        self.assertEqual(len(rt_series), len(self.ds))
        self.assertListEqual(rt_series.tolist(), [1.23, 2.34, 3.45])

        # Case 3: slice -> MSDataset subset
        sub_ds = self.ds[1:3]
        self.assertIsInstance(sub_ds, MSDataset)
        self.assertEqual(len(sub_ds), 2)
        self.assertListEqual(sub_ds["spectrum_id"].tolist(), ["s2", "s3"])

        # Case 4: list of indices -> MSDataset subset
        sub_ds2 = self.ds[[0, 2]]
        self.assertEqual(len(sub_ds2), 2)
        self.assertListEqual(sub_ds2["spectrum_id"].tolist(), ["s1", "s3"])

        # Case 5: numpy array of indices -> MSDataset subset
        sub_ds3 = self.ds[np.array([1])]
        self.assertEqual(len(sub_ds3), 1)
        self.assertEqual(sub_ds3["spectrum_id"].iloc[0], "s2")

        # Case 6: torch tensor of indices -> MSDataset subset
        sub_ds4 = self.ds[torch.tensor([2])]
        self.assertEqual(len(sub_ds4), 1)
        self.assertEqual(sub_ds4["spectrum_id"].iloc[0], "s3")

    def test_add_new_column_subset(self):
        sub_ds = self.ds[:2]
        sub_ds["flag"] = [1, 0]

        col = self.ds._spectrum_meta_ref["flag"].values
        # First two updated, last should be NaN
        self.assertTrue(np.all(col[:2] == [1, 0]))
        self.assertTrue(np.isnan(col[2]))

    def test_scalar_assignment_subset(self):
        sub_ds = self.ds[1:3]
        sub_ds["status"] = "ok"

        col = self.ds._spectrum_meta_ref["status"].values
        # Index 1 and 2 updated, index 0 should be NaN
        self.assertTrue(np.isnan(col[0]))
        self.assertEqual(col[1], "ok")
        self.assertEqual(col[2], "ok")

    def test_update_existing_column_subset(self):
        # Add first
        self.ds["label"] = ["x", "y", "z"]
        sub_ds = self.ds[1:3]
        sub_ds["label"] = ["aa", "bb"]

        col = self.ds._spectrum_meta_ref["label"].values
        self.assertEqual(col[0], "x")
        self.assertEqual(col[1], "aa")
        self.assertEqual(col[2], "bb")

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

        # Editing copy must not affect the original
        ds_copy[0]["spectrum_id"] = "modified"
        self.assertNotEqual(ds_copy.meta_copy.iloc[0, 0], self.ds.meta_copy.iloc[0, 0])

    def test_sort_by(self):
        # Add a column to sort on
        self.ds["score"] = [30, 10, 20]  # deliberately unsorted values

        # Sort by "score" ascending
        sorted_ds = self.ds.sort_by("score", ascending=True)

        # Check 1: dataset length must remain the same
        self.assertEqual(len(sorted_ds), len(self.ds))

        # Check 2: sorted values are in ascending order
        sorted_scores = sorted_ds["score"].tolist()
        self.assertEqual(sorted_scores, sorted(sorted_scores))

        # Check 3: spectrum_id order is consistent with sorting
        # Original spectrum_id with score mapping: s1->30, s2->10, s3->20
        # Ascending by score should yield [s2, s3, s1]
        self.assertListEqual(
            sorted_ds["spectrum_id"].tolist(),
            ["s2", "s3", "s1"]
        )

        # Sort by "score" descending
        sorted_ds_desc = self.ds.sort_by("score", ascending=False)

        # Check 4: sorted values are in descending order
        sorted_scores_desc = sorted_ds_desc["score"].tolist()
        self.assertEqual(sorted_scores_desc, sorted(sorted_scores_desc, reverse=True))

        # Check 5: spectrum_id order is consistent with descending sort
        # Should be [s1, s3, s2]
        self.assertListEqual(
            sorted_ds_desc["spectrum_id"].tolist(),
            ["s1", "s3", "s2"]
        )

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
                ds_loaded.peaks._data,
                self.ds.peaks._data
            )
            torch.testing.assert_close(
                ds_loaded.peaks._offsets,
                self.ds.peaks._offsets
            )

            # --- Peak metadata check ---
            if self.ds.peaks._metadata_ref is not None:
                pd.testing.assert_frame_equal(
                    ds_loaded.peaks._metadata_ref.reset_index(drop=True),
                    self.ds.peaks._metadata_ref.reset_index(drop=True)
                )

            # Ensure loaded is a new object, not a reference
            self.assertIsNot(ds_loaded._spectrum_meta_ref, self.ds._spectrum_meta_ref)
            self.assertIsNot(ds_loaded.peaks._data_ref, self.ds.peaks._data_ref)

    def test_concat(self):
        for i in range(2):  # Test twice to ensure consistency
            # --- Split dataset into subsets of size 1 ---
            ds = self.ds.copy()
            if i == 0:
                subsets = [ds[i:i+1].copy() for i in range(len(ds))]
            else:
                subsets = [ds[i:i+1] for i in range(len(ds))]

            # Each subset should contain exactly one spectrum
            self.assertTrue(all(len(sub) == 1 for sub in subsets))

            # --- Concatenate subsets back into one dataset ---
            merged = MSDataset.concat(subsets)

            # --- Check dataset length and shape ---
            self.assertEqual(len(merged), len(self.ds))
            self.assertEqual(merged.shape, self.ds.shape)

            # --- Check spectrum metadata equality ---
            pd.testing.assert_frame_equal(
                merged.meta_copy.reset_index(drop=True),
                self.ds.meta_copy.reset_index(drop=True)
            )

            # --- Check peak data equality ---
            torch.testing.assert_close(
                merged.peaks._data_ref,
                self.ds.peaks._data_ref
            )
            torch.testing.assert_close(
                merged.peaks._offsets_ref,
                self.ds.peaks._offsets_ref
            )

            # --- Check peak metadata equality ---
            if self.ds.peaks._metadata_ref is not None:
                pd.testing.assert_frame_equal(
                    merged.peaks._metadata_ref.reset_index(drop=True),
                    self.ds.peaks._metadata_ref.reset_index(drop=True)
                )


if __name__ == "__main__":
    unittest.main()
