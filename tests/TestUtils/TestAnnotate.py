import unittest
import os
import torch
import pandas as pd
import numpy as np

# Adjust these imports to your project structure
from msentity.core.MSDataset import MSDataset
from msentity.core.PeakSeries import PeakSeries
from msentity.utils.annotate import set_spec_id, set_peak_id


class TestIdAssign(unittest.TestCase):

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
        self.spectrum_meta = pd.DataFrame({
            "spectrum_id": ["s1", "s2", "s3"],
            "group": ["A", "B", "A"],
            "rt": [1.23, 2.34, 3.45],
            "intensity_sum": [45.0, 150.0, 120.0]
        })

        # Initialize MSDataset (your project's style)
        self.ds = MSDataset(self.spectrum_meta, self.peak_series)

        # Save file path (if you later want file I/O tests)
        self.test_file = os.path.join("tests", "dummy_files", "test_dataset.hdf5")
        os.makedirs(os.path.dirname(self.test_file), exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    # -----------------------------
    # set_spec_id tests
    # -----------------------------
    def test_set_spec_id_assigns_ids(self):
        ok = set_spec_id(self.ds, prefix="Spec")
        self.assertTrue(ok)

        self.assertIn("SpecID", self.ds._spectrum_meta_ref.columns)
        # n=3 -> width=1 -> Spec1, Spec2, Spec3
        self.assertListEqual(
            self.ds._spectrum_meta_ref["SpecID"].tolist(),
            ["Spec1", "Spec2", "Spec3"]
        )

    def test_set_spec_id_does_not_overwrite(self):
        self.ds._spectrum_meta_ref["SpecID"] = ["X", "Y", "Z"]

        ok = set_spec_id(self.ds, prefix="Spec")
        self.assertFalse(ok)

        self.assertListEqual(
            self.ds._spectrum_meta_ref["SpecID"].tolist(),
            ["X", "Y", "Z"]
        )

    def test_set_spec_id_prefix_type_check(self):
        with self.assertRaises(ValueError):
            set_spec_id(self.ds, prefix=123)  # type: ignore[arg-type]

    # -----------------------------
    # set_peak_id tests
    # -----------------------------
    def test_set_peak_id_assigns_local_ids(self):
        ok = set_peak_id(self.ds, col_name="PeakID", overwrite=False, start=0)
        self.assertTrue(ok)

        self.assertIsNotNone(self.ds.peaks._metadata_ref)
        self.assertIn("PeakID", self.ds.peaks._metadata_ref.columns)

        got = self.ds.peaks._metadata_ref["PeakID"].to_numpy(dtype=np.int64)

        # Spectrum1: 0,1,2 / Spectrum2: 0,1,2,3 / Spectrum3: 0,1
        expected = np.array([0, 1, 2, 0, 1, 2, 3, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(got, expected)

    def test_set_peak_id_start_offset(self):
        ok = set_peak_id(self.ds, col_name="PeakID", overwrite=True, start=1)
        self.assertTrue(ok)

        got = self.ds.peaks._metadata_ref["PeakID"].to_numpy(dtype=np.int64)

        expected = np.array([1, 2, 3, 1, 2, 3, 4, 1, 2], dtype=np.int64)
        np.testing.assert_array_equal(got, expected)

    def test_set_peak_id_skip_if_exists(self):
        # Add column first
        self.ds.peaks._metadata_ref["PeakID"] = 999

        ok = set_peak_id(self.ds, col_name="PeakID", overwrite=False, start=0)
        self.assertFalse(ok)

        got = self.ds.peaks._metadata_ref["PeakID"].to_numpy(dtype=np.int64)
        expected = np.full(9, 999, dtype=np.int64)
        np.testing.assert_array_equal(got, expected)

    def test_set_peak_id_overwrite(self):
        self.ds.peaks._metadata_ref["PeakID"] = 999

        ok = set_peak_id(self.ds, col_name="PeakID", overwrite=True, start=0)
        self.assertTrue(ok)

        got = self.ds.peaks._metadata_ref["PeakID"].to_numpy(dtype=np.int64)
        expected = np.array([0, 1, 2, 0, 1, 2, 3, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(got, expected)


if __name__ == "__main__":
    unittest.main()
