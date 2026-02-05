import os
import unittest

from msentity.core import MSDataset
from msentity.io.prepare_ms_data import load_ms_data


class TestLoadMSData(unittest.TestCase):
    def setUp(self):
        # --- Dummy input files ---
        self.msp_file = os.path.join("tests", "dummy_files", "test_dummy.msp")
        self.mgf_file = os.path.join("tests", "dummy_files", "test_dummy.mgf")
        self.hdf5_file = os.path.join("tests", "dummy_files", "test_dummy.hdf5")

        # --- A file with an unknown extension for negative test ---
        self.unknown_file = os.path.join("tests", "dummy_files", "test_dummy.unknown")

    def test_load_ms_data_msp_auto_detect(self):
        ds = load_ms_data(self.msp_file, spec_id_prefix="TEST", file_type=None)
        self.assertIsInstance(ds, MSDataset)

    def test_load_ms_data_msp_explicit(self):
        ds = load_ms_data(self.msp_file, spec_id_prefix="TEST", file_type="msp")
        self.assertIsInstance(ds, MSDataset)

    def test_load_ms_data_mgf_auto_detect(self):
        ds = load_ms_data(self.mgf_file, spec_id_prefix="TEST", file_type=None)
        self.assertIsInstance(ds, MSDataset)

    def test_load_ms_data_mgf_explicit(self):
        ds = load_ms_data(self.mgf_file, spec_id_prefix="TEST", file_type="mgf")
        self.assertIsInstance(ds, MSDataset)

    def test_load_ms_data_hdf5_auto_detect(self):
        ds = load_ms_data(self.hdf5_file, spec_id_prefix="TEST", file_type=None)
        self.assertIsInstance(ds, MSDataset)

    def test_load_ms_data_hdf5_explicit(self):
        ds = load_ms_data(self.hdf5_file, spec_id_prefix="TEST", file_type="hdf5")
        self.assertIsInstance(ds, MSDataset)

    def test_load_ms_data_unknown_extension_raises(self):
        with self.assertRaises(ValueError):
            load_ms_data(self.unknown_file, spec_id_prefix="TEST", file_type=None)

    def test_load_ms_data_unsupported_file_type_raises(self):
        with self.assertRaises(ValueError):
            load_ms_data(self.msp_file, spec_id_prefix="TEST", file_type="mzml")

if __name__ == "__main__":
    unittest.main()
