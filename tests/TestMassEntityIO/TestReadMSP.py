import os
import unittest
import torch
import pandas as pd
import tempfile
from msentity.io.msp import read_msp, write_msp
from msentity.core import MSDataset, PeakSeries


class TestReadMSP(unittest.TestCase):
    def setUp(self):
        # --- Create a dummy MSP file ---
        self.test_file = os.path.join("tests", "dummy_files", "test_dummy.msp")
        self.test_file_with_error = os.path.join("tests", "dummy_files", "test_dummy_with_error.msp")
        self.test_file_with_peak_meta = os.path.join("tests", "dummy_files", "test_dummy_with_peak_meta.msp")
        self.test_file_with_quoted = os.path.join("tests", "dummy_files", "test_dummy_with_quoted.msp")

    def test_read_msp_file(self):
        # --- Read file ---
        ds = read_msp(self.test_file, show_progress=False)

        # Check type
        self.assertIsInstance(ds, MSDataset)


        # --- Spectrum metadata checks ---
        meta = ds.meta_copy
        # Expected 5 spectra in the dummy MSP
        self.assertEqual(len(ds), 5)
        # Check essential columns
        expected_cols = ["Name", "PrecursorMZ", "AdductType",
                         "IonMode", "Formula", "SMILES", "InChIKey"]
        for col in expected_cols:
            self.assertIn(col, meta.columns)

        # Spot check values
        self.assertEqual(meta.loc[0, "Name"], "MassSpecID0000001")
        self.assertAlmostEqual(float(meta.loc[0, "PrecursorMZ"]), 288.1225, places=4)

        # --- PeakSeries checks ---
        ps = ds.peaks
        self.assertIsInstance(ps, PeakSeries)
        # Total peaks should equal sum of NumPeaks in file (8+6+12+20+6 = 52)
        self.assertEqual(ps.n_all_peaks, 52)
        # First spectrum should have 8 peaks
        self.assertEqual(ps.n_peaks(0), 8)
        # Last spectrum should have 6 peaks
        self.assertEqual(ps.n_peaks(4), 6)

        # Check m/z values roughly match expectations
        first_spectrum_data = ps[0].data
        self.assertAlmostEqual(first_spectrum_data[0, 0].item(), 91.0542, places=4)
        self.assertAlmostEqual(first_spectrum_data[-1, 0].item(), 246.1125, places=4)

    def test_read_msp_file_with_peak_meta(self):
        # --- Read file ---
        ds = read_msp(self.test_file_with_peak_meta, show_progress=False)

        # Check type
        self.assertIsInstance(ds, MSDataset)


        # --- Spectrum metadata checks ---
        meta = ds.meta_copy
        # Expected 5 spectra in the dummy MSP
        self.assertEqual(len(ds), 5)
        # Check essential columns
        expected_cols = ["Name", "PrecursorMZ", "AdductType",
                         "IonMode", "Formula", "SMILES", "InChIKey"]
        for col in expected_cols:
            self.assertIn(col, meta.columns)

        # Spot check values
        self.assertEqual(meta.loc[0, "Name"], "MassSpecID0000001")
        self.assertAlmostEqual(float(meta.loc[0, "PrecursorMZ"]), 288.1225, places=4)

        # --- PeakSeries checks ---
        ps = ds.peaks
        self.assertIsInstance(ps, PeakSeries)
        # Total peaks should equal sum of NumPeaks in file (8+6+12+20+6 = 52)
        self.assertEqual(ps.n_all_peaks, 52)
        # First spectrum should have 8 peaks
        self.assertEqual(ps.n_peaks(0), 8)
        # Last spectrum should have 6 peaks
        self.assertEqual(ps.n_peaks(4), 6)

        # Check m/z values roughly match expectations
        first_spectrum_data = ps[0].data
        self.assertAlmostEqual(first_spectrum_data[0, 0].item(), 91.0542, places=4)
        self.assertAlmostEqual(first_spectrum_data[-1, 0].item(), 246.1125, places=4)

        peak_meta = ds.peaks._metadata
        self.assertEqual(peak_meta.columns[0], 'Formula')
        self.assertEqual(peak_meta.columns[1], 'ppm')

        self.assertEqual(peak_meta.iloc[9, 0], 'C6H5O3')
        self.assertEqual(peak_meta.iloc[20, 1], '-3.50')
        self.assertEqual(peak_meta.iloc[48, 1], '-2.92')
        self.assertEqual(peak_meta.iloc[49, 1], '-2.48')
        pass

    def test_read_msp_file_with_error(self):
        # --- Read file ---
        with tempfile.TemporaryDirectory(dir=os.path.dirname(self.test_file_with_error)) as tmpdir:
            error_log_file = os.path.join(tmpdir, "error_log.txt")
            # ds = read_msp(self.test_file_with_error, error_log_level=2)
            ds = read_msp(self.test_file_with_error, error_log_level=2, error_log_file=error_log_file, show_progress=False)

        # Check type
        self.assertIsInstance(ds, MSDataset)


        # --- Spectrum metadata checks ---
        meta = ds.meta_copy
        # Expected 5 spectra in the dummy MSP
        self.assertEqual(len(ds), 3)
        # Check essential columns
        expected_cols = ["Name", "PrecursorMZ", "AdductType",
                         "IonMode", "Formula", "SMILES", "InChIKey"]
        for col in expected_cols:
            self.assertIn(col, meta.columns)

        # Spot check values
        self.assertEqual(meta.loc[0, "Name"], "MassSpecID0000002")


    def test_write_and_read_back(self):
        # Read dummy dataset
        ds = read_msp(self.test_file, show_progress=False)

        with tempfile.TemporaryDirectory(dir=os.path.dirname(self.test_file)) as tmpdir:
            out_path = os.path.join(tmpdir, "out.msp")

            # Write dataset to MSP
            write_msp(ds, out_path)

            # Read back the written file
            ds2 = read_msp(out_path, show_progress=False)

            # Compare dataset 
            self.assertEqual(len(ds), len(ds2))
            for i in range(len(ds)):
                self.assertEqual(ds[i], ds2[i])
            
    def test_write_and_read_back_with_peak_meta(self):
        # Read dummy dataset
        ds = read_msp(self.test_file_with_peak_meta, show_progress=False)

        with tempfile.TemporaryDirectory(dir=os.path.dirname(self.test_file_with_peak_meta)) as tmpdir:
            out_path = os.path.join(tmpdir, "out.msp")

            # Write dataset to MSP
            write_msp(ds, out_path)

            # Read back the written file
            ds2 = read_msp(out_path, show_progress=False)

            # Compare dataset 
            self.assertEqual(len(ds), len(ds2))
            for i in range(len(ds)):
                self.assertEqual(ds[i], ds2[i])

    def test_write_with_header_map_from_read(self):
        # Read dataset and get header_map
        ds, header_map = read_msp(self.test_file, return_header_map=True, show_progress=False)

        with tempfile.TemporaryDirectory(dir=os.path.dirname(self.test_file)) as tmpdir:
            out_path = os.path.join(tmpdir, "out_header.msp")

            # Write using header_map obtained from read_msp
            write_msp(ds, out_path, header_map=header_map)

            # Read file again and get new header_map
            ds2, header_map2 = read_msp(out_path, return_header_map=True, show_progress=False)

            # Compare dataset 
            self.assertEqual(len(ds), len(ds2))
            for i in range(len(ds)):
                self.assertEqual(ds[i], ds2[i])


                
    def test_read_msp_file_with_quoted(self):
        # --- Read file ---
        ds = read_msp(self.test_file_with_quoted, show_progress=False)

        # Check type
        self.assertIsInstance(ds, MSDataset)


        # --- Spectrum metadata checks ---
        meta = ds.meta_copy
        # Expected 5 spectra in the dummy MSP
        self.assertEqual(len(ds), 2)
    
        # --- PeakSeries checks ---
        ps = ds.peaks
        self.assertIsInstance(ps, PeakSeries)
        # Total peaks should equal sum of NumPeaks in file (8+6+12+20+6 = 52)
        self.assertEqual(ps.n_all_peaks, 12)
        self.assertEqual(ps.n_peaks(0), 6)
        self.assertEqual(ps.n_peaks(1), 6)

        # Check m/z values roughly match expectations
        self.assertAlmostEqual(ds[0].peaks[0].mz, 53.0387, places=4)
        self.assertAlmostEqual(ds[1].peaks[2].intensity, 11.8, places=4)
        self.assertEqual(ds[0].peaks[5]['column1'], 'C2H6NNaO7=p-C13H9O/-8.7ppm 8/25')
        self.assertEqual(ds[1].peaks[2]['column1'], 'C9H6NNa=p-C6H9O8/1.1ppm;C8H7O3=p-C7H8NNaO5/2.8ppm 6/24')

    def test_write_and_read_back_with_quoted(self):
        # Read dummy dataset
        ds = read_msp(self.test_file_with_quoted, show_progress=False)

        with tempfile.TemporaryDirectory(dir=os.path.dirname(self.test_file_with_quoted)) as tmpdir:
            out_path = os.path.join(tmpdir, "out.msp")

            # Write dataset to MSP
            write_msp(ds, out_path)

            # Read back the written file
            ds2 = read_msp(out_path, show_progress=False)

            # Compare dataset 
            self.assertEqual(len(ds), len(ds2))
            for i in range(len(ds)):
                self.assertEqual(ds[i], ds2[i])
            

if __name__ == "__main__":
    unittest.main()
