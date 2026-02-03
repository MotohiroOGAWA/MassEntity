import unittest
import os
import io
import h5py
import numpy as np

# Adjust this import path to your project structure
from msentity.utils.inspect import print_hdf5_structure


class TestHDF5Inspect(unittest.TestCase):
    def setUp(self):
        # --- Prepare a temporary HDF5 file under tests/dummy_files ---
        self.test_dir = os.path.join("tests", "dummy_files")
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_file = os.path.join(self.test_dir, "test_dummy_structure.hdf5")

        # --- Create a small dummy HDF5 structure ---
        with h5py.File(self.test_file, "w") as f:
            meta = f.create_group("metadata")
            meta.attrs["description"] = "dummy dataset"
            meta.attrs["attributes_json"] = "{}"
            meta.attrs["tags_json"] = "[]"

            ds0 = f.create_group("dataset_0")
            peaks = ds0.create_group("peaks")
            peaks.create_dataset("data", data=np.zeros((3, 2), dtype=np.float32))
            peaks.create_dataset("offsets", data=np.array([0, 1, 3], dtype=np.int64))
            peaks.create_dataset("index", data=np.array([0, 1], dtype=np.int64))

            # --- Simulate new-style parquet bytes storage as uint8 array ---
            peaks.create_dataset("metadata_parquet", data=np.array([1, 2, 3, 4], dtype=np.uint8))
            peaks["metadata_parquet"].attrs["bytes_format"] = "uint8_1d"
            peaks["metadata_parquet"].attrs["nbytes"] = 4

            ds0.create_dataset("spectrum_meta_parquet", data=np.array([9, 9, 9], dtype=np.uint8))

    def tearDown(self):
        # --- Cleanup the temporary file ---
        try:
            os.remove(self.test_file)
        except FileNotFoundError:
            pass

    def _capture_stdout(self, fn, *args, **kwargs) -> str:
        """Capture stdout from a function call and return it as a string."""
        buf = io.StringIO()
        import sys
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            fn(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    def test_print_hdf5_structure_outputs_expected_content(self):
        out = self._capture_stdout(
            print_hdf5_structure,
            self.test_file,
            show_attrs=True,
            show_datasets=True,
            max_depth=None,
        )

        # --- Basic assertions: file header and key groups exist ---
        self.assertIn("HDF5 file:", out)

        # Note: print_hdf5_structure() may not print the root group "/"
        self.assertIn("[Group] dataset_0", out)
        self.assertIn("[Group] dataset_0/peaks", out)
        self.assertIn("[Group] metadata", out)

        # --- Attribute assertions ---
        self.assertIn("@attr description: dummy dataset", out)
        self.assertIn("@attr attributes_json: {}", out)
        self.assertIn("@attr tags_json: []", out)

        # --- Dataset info assertions (shape/dtype) ---
        self.assertIn("[Dataset] dataset_0/peaks/data", out)
        self.assertIn("shape=(3, 2)", out)
        self.assertIn("dtype=float32", out)

        # --- Parquet-ish dataset presence ---
        self.assertIn("dataset_0/peaks/metadata_parquet", out)
        self.assertIn("dataset_0/spectrum_meta_parquet", out)

    def test_print_hdf5_structure_respects_max_depth(self):
        # With max_depth=0, only top-level objects (no nested paths) should appear.
        out = self._capture_stdout(
            print_hdf5_structure,
            self.test_file,
            show_attrs=True,
            show_datasets=True,
            max_depth=0,
        )

        # --- Top-level groups should exist ---
        self.assertIn("[Group] dataset_0", out)
        self.assertIn("[Group] metadata", out)

        # --- Nested paths should NOT appear ---
        self.assertNotIn("dataset_0/peaks", out)
        self.assertNotIn("dataset_0/peaks/data", out)
        self.assertNotIn("dataset_0/spectrum_meta_parquet", out)


if __name__ == "__main__":
    unittest.main()
