import os
import unittest
import torch
import pandas as pd
import tempfile
from msentity.io.mgf import read_mgf, write_mgf
from msentity.core import MSDataset, PeakSeries


class TestReadMGF(unittest.TestCase):
    def setUp(self):
        # --- Create dummy MGF file paths ---
        self.test_file = os.path.join("tests", "dummy_files", "test_dummy.mgf")
        self.test_file_with_error = os.path.join("tests", "dummy_files", "test_dummy_with_error.mgf")
        self.test_file_with_peak_meta = os.path.join("tests", "dummy_files", "test_dummy_with_peak_meta.mgf")
        self.test_file_with_quoted = os.path.join("tests", "dummy_files", "test_dummy_with_quoted.mgf")

    def test_read_mgf_file(self):
        # --- Read file ---
        ds = read_mgf(self.test_file, show_progress=False)

        # Type check
        self.assertIsInstance(ds, MSDataset)

        # --- Spectrum metadata checks ---
        meta = ds.meta
        # Expected 6 spectra
        self.assertEqual(len(ds), 6)
        # Check essential columns
        expected_cols = ["IDENTIFIER", "SMILES", "InChIKey", "Formula", "PRECURSORFORMULA", "PARENTMASS",
                         "PrecursorMZ", "ADDUCT", "InstrumentType", "CollisionEnergy"]
        for col in expected_cols:
            self.assertIn(col, meta.columns)

        # Spot check values
        self.assertEqual(meta.loc[0, "IDENTIFIER"], "MassSpecID0000001")
        self.assertAlmostEqual(float(meta.loc[1, "PrecursorMZ"]), 288.1225, places=4)

        # --- PeakSeries checks ---
        ps = ds.peaks
        self.assertIsInstance(ps, PeakSeries)
        # Total peaks = 52 (sum of all NumPeaks)
        self.assertEqual(ps.n_all_peaks, 72)
        # First spectrum should have 8 peaks
        self.assertEqual(ps.n_peaks(0), 8)
        # Last spectrum should have 6 peaks
        self.assertEqual(ps.n_peaks(4), 6)

        # Check m/z values roughly match expectations
        first_spec_data = ps[0].data
        self.assertAlmostEqual(first_spec_data[0, 0].item(), 91.0542, places=4)
        self.assertAlmostEqual(first_spec_data[-1, 0].item(), 246.1125, places=4)

    def test_read_mgf_file_with_peak_meta(self):
        ds = read_mgf(self.test_file_with_peak_meta, show_progress=False)
        self.assertIsInstance(ds, MSDataset)

        self.assertEqual(len(ds), 6)

        # --- Peak metadata checks ---
        peak_meta = ds.peaks.metadata
        self.assertEqual(peak_meta.columns[0], 'Formula')
        self.assertEqual(peak_meta.columns[1], 'ppm')
        self.assertEqual(ds[0].peaks[0]['Formula'], '')
        self.assertEqual(ds[1].peaks[3]['Formula'], 'C13H13O')
        self.assertEqual(ds[2].peaks[6]['ppm'], '-3.50')
        self.assertEqual(ds[4].peaks[0]['Formula'], 'C7H7')
        self.assertEqual(ds[4].peaks[0]['ppm'], '-6.32')
        self.assertEqual(ds[4].peaks[2]['Formula'], '')
        self.assertEqual(ds[4].peaks[2]['ppm'], '-2.92')

    def test_read_mgf_file_with_error(self):
        with tempfile.TemporaryDirectory(dir=os.path.dirname(self.test_file_with_error)) as tmpdir:
            error_log_file = os.path.join(tmpdir, "error_log.txt")
            ds = read_mgf(
                self.test_file_with_error,
                error_log_level=2,
                error_log_file=error_log_file,
                show_progress=False
            )

        self.assertIsInstance(ds, MSDataset)

        meta = ds.meta
        self.assertEqual(len(ds), 4)

        # Spot check
        self.assertEqual(meta.loc[0, "IDENTIFIER"], "MassSpecID0000002")

    def test_write_and_read_back(self):
        ds = read_mgf(self.test_file, show_progress=False)

        with tempfile.TemporaryDirectory(dir=os.path.dirname(self.test_file)) as tmpdir:
            out_path = os.path.join(tmpdir, "out.mgf")

            write_mgf(ds, out_path, show_progress=False)
            ds2 = read_mgf(out_path, show_progress=False)

            self.assertEqual(len(ds), len(ds2))
            for i in range(len(ds)):
                self.assertEqual(ds[i], ds2[i])

    def test_write_and_read_back_with_peak_meta(self):
        ds = read_mgf(self.test_file_with_peak_meta, show_progress=False)

        with tempfile.TemporaryDirectory(dir=os.path.dirname(self.test_file_with_peak_meta)) as tmpdir:
            out_path = os.path.join(tmpdir, "out.mgf")

            write_mgf(ds, out_path, show_progress=False)
            ds2 = read_mgf(out_path, show_progress=False)

            self.assertEqual(len(ds), len(ds2))
            for i in range(len(ds)):
                self.assertEqual(ds[i], ds2[i])

    def test_write_with_header_map_from_read(self):
        ds, header_map = read_mgf(self.test_file, return_header_map=True, show_progress=False)

        with tempfile.TemporaryDirectory(dir=os.path.dirname(self.test_file)) as tmpdir:
            out_path = os.path.join(tmpdir, "out_header.mgf")

            write_mgf(ds, out_path, header_map=header_map, show_progress=False)
            ds2, header_map2 = read_mgf(out_path, return_header_map=True, show_progress=False)

            self.assertEqual(len(ds), len(ds2))
            for i in range(len(ds)):
                self.assertEqual(ds[i], ds2[i])
                
    def test_read_mgf_file_with_quoted(self):
        # --- Read file ---
        ds = read_mgf(self.test_file_with_quoted, show_progress=False)

        # Check type
        self.assertIsInstance(ds, MSDataset)


        # --- Spectrum metadata checks ---
        meta = ds.meta
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
        ds = read_mgf(self.test_file_with_quoted, show_progress=False)

        with tempfile.TemporaryDirectory(dir=os.path.dirname(self.test_file_with_quoted)) as tmpdir:
            out_path = os.path.join(tmpdir, "out.mgf")

            # Write dataset to MGF
            write_mgf(ds, out_path, show_progress=False)

            # Read back the written file
            ds2 = read_mgf(out_path, show_progress=False)

            # Compare dataset 
            self.assertEqual(len(ds), len(ds2))
            for i in range(len(ds)):
                self.assertEqual(ds[i], ds2[i])


if __name__ == "__main__":
    unittest.main()
