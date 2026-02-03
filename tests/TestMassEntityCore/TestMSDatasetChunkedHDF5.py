import unittest
import os
import torch
import pandas as pd
import numpy as np
import h5py

from msentity.core.PeakSeries import PeakSeries
from msentity.core.MSDataset import MSDataset


class TestMSDatasetChunkedParquet(unittest.TestCase):
    def setUp(self):
        # --- Prepare PeakSeries test data (100 peaks) ---
        n = 100
        spec_num = 10
        rows = []
        for spec_id in range(n):
            for j in range(spec_num):
                rows.append([100.0 + spec_id * 50 + j, float(j + 1)])

        data = torch.tensor(rows, dtype=torch.float32)
        offsets = torch.tensor([i * spec_num for i in range(n + 1)], dtype=torch.int64)
        peak_meta = pd.DataFrame(
            {
                "peak_id": list(range(1, n * spec_num + 1)),
                "note": [f"p{i}" for i in range(1, n * spec_num + 1)],
            }
        )
        peak_series = PeakSeries(data, offsets, peak_meta)

        spectrum_meta = pd.DataFrame(
            {
                "spectrum_id": list(range(n)),
                "rt": [float(i + 1) for i in range(n)],
            }
        )

        self.ds = MSDataset(spectrum_meta, peak_series)
        self.test_file = os.path.join("tests", "dummy_files", "test_dataset_chunked.hdf5")
        os.makedirs(os.path.dirname(self.test_file), exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_force_chunked_save_and_load_peak_meta(self):
        """
        Force chunked Parquet storage by:
          - patching _dump_parquet_to_bytes to return large bytes
          - calling _save_parquet_h5 with very small max_part_bytes
        Then verify:
          - part datasets exist
          - _load_parquet_h5 reconstructs the DataFrame correctly
        """

        try:
            # --- Monkey-patch to force large "parquet bytes" ---
            # MSDataset._dump_parquet_to_bytes = staticmethod(fake_dump_parquet_to_bytes)

            # --- Write chunked parquet into HDF5 manually ---
            with h5py.File(self.test_file, "w") as f:
                grp = f.create_group("dataset_0")
                peaks_grp = grp.create_group("peaks")

                # Save required arrays (minimal)
                peaks_grp.create_dataset("data", data=self.ds.peaks._data_ref.cpu().numpy())
                peaks_grp.create_dataset("offsets", data=self.ds.peaks._offsets_ref.cpu().numpy())
                peaks_grp.create_dataset("index", data=self.ds.peaks._index.cpu().numpy())

                # Force chunking with small max_part_bytes
                MSDataset._save_parquet_h5(
                    peaks_grp,
                    "metadata_parquet",
                    self.ds.peaks._metadata_ref,
                    max_part_bytes=5000,     # << small to force multi-part
                )

            # --- Check the HDF5 structure contains parts ---
            with h5py.File(self.test_file, "r") as f:
                peaks_grp = f["dataset_0"]["peaks"]

                self.assertTrue(bool(peaks_grp.attrs.get("metadata_parquet__chunked", False)))
                num_parts = int(peaks_grp.attrs.get("metadata_parquet__num_parts", 0))
                self.assertGreaterEqual(num_parts, 2)

                self.assertIn("metadata_parquet__part_000", peaks_grp)
                self.assertIn("metadata_parquet__part_001", peaks_grp)

            # --- Counter-based decoder ---
            call_i = {"i": 0}
            rows = self.ds.peaks._metadata_ref.reset_index(drop=True)

            try:
                with h5py.File(self.test_file, "r") as f:
                    peaks_grp = f["dataset_0"]["peaks"]
                    loaded = MSDataset._load_parquet_h5(peaks_grp, "metadata_parquet")

                pd.testing.assert_frame_equal(
                    loaded.reset_index(drop=True),
                    self.ds.peaks._metadata_ref.reset_index(drop=True),
                )
            finally:
                pass

        finally:
            pass

    def test_from_hdf5_with_load_peak_meta_false(self):
        """
        Ensure from_hdf5(load_peak_meta=False) does not attempt to read metadata_parquet
        even if present, and still loads spectrum/peaks arrays.
        """
        # Save normally (small data)
        self.ds.to_hdf5(self.test_file, save_ref=False, mode="w")

        loaded = MSDataset.from_hdf5(self.test_file, load_peak_meta=False)

        # Peak meta should be skipped
        self.assertIsNone(loaded.peaks._metadata_ref)

        # Arrays should still be present
        torch.testing.assert_close(loaded.peaks._data_ref, self.ds.peaks._data_ref)
        torch.testing.assert_close(loaded.peaks._offsets_ref, self.ds.peaks._offsets_ref)
        torch.testing.assert_close(loaded.peaks._index, self.ds.peaks._index)


if __name__ == "__main__":
    unittest.main()
