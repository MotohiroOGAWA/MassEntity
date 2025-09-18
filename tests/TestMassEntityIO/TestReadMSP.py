import os
import unittest
import torch
import pandas as pd
# from MassEntity.MassEntityCore import MSDataset, PeakSeries
# from MassEntity.MassEntityIO import read_msp
from MassEntity.MassEntityIO.msp import read_msp
from MassEntity.MassEntityCore import MSDataset, PeakSeries


class TestReadMSP(unittest.TestCase):
    def setUp(self):
        # --- Create a dummy MSP file ---
        self.test_file = os.path.join("MassEntity", "tests", "TestMassEntityIO", "dummy_files", "test_dummy.msp")

    def test_read_msp_file(self):
        # --- Read file ---
        ds = read_msp(self.test_file)

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
        self.assertEqual(meta.loc[0, "Name"], "MassSpecGymID0000001")
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

if __name__ == "__main__":
    unittest.main()
