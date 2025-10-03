import unittest
import torch
import pandas as pd
from msentity.core.PeakSeries import PeakSeries
from msentity.core.MSDataset import MSDataset
from msentity.core.SpecCondition import *  # SpecCondition, AllIntegerMZ


class TestSpecCondition(unittest.TestCase):
    def setUp(self):
        # Build 4 spectra: sizes (3, 2, 3, 2) = 10 peaks
        self.data = torch.tensor([
            # Spectrum 0 (all integer m/z)
            [100.0, 10.0],
            [105.0, 50.0],
            [110.0, 5.0],

            # Spectrum 1 (non-integer m/z)
            [200.123, 80.0],
            [205.0, 20.0],

            # Spectrum 2 (all integer m/z)
            [300.0, 5.0],
            [310.0, 90.0],
            [320.0, 15.0],

            # Spectrum 3 (non-integer m/z)
            [400.5, 70.0],
            [405.0, 25.0],
        ], dtype=torch.float32)

        self.offsets = torch.tensor([0, 3, 5, 8, 10], dtype=torch.int64)
        self.metadata = pd.DataFrame({"spec_id": list(range(4))})
        self.ps = PeakSeries(self.data, self.offsets, None)
        self.ds = MSDataset(self.metadata, self.ps)

    def test_and_condition(self):
        cond1 = AllIntegerMZ()
        cond2 = ~AllIntegerMZ()
        combined = cond1 & cond2
        mask = combined.evaluate(self.ds)

        # Should always be False because cond1 and cond2 are mutually exclusive
        expected = torch.zeros(4, dtype=torch.bool)
        torch.testing.assert_close(mask, expected)

    def test_or_condition(self):
        cond1 = AllIntegerMZ()
        cond2 = ~AllIntegerMZ()
        combined = cond1 | cond2
        mask = combined.evaluate(self.ds)

        # Should always be True because at least one is always satisfied
        expected = torch.ones(4, dtype=torch.bool)
        torch.testing.assert_close(mask, expected)

    def test_not_condition(self):
        cond = ~AllIntegerMZ()
        mask = cond.evaluate(self.ds)

        # Expect spectra 1 and 3 to be True (contain non-integer m/z)
        expected = torch.tensor([False, True, False, True])
        torch.testing.assert_close(mask, expected)
        
    def test_all_integer_mz_condition(self):
        cond = ~AllIntegerMZ()
        filtered_ds = self.ds.filter(cond)

        # Expect spectra 0 and 2 to be True (all m/z integers)
        self.assertEqual(len(filtered_ds), 2)
        expected_spec_ids = [1, 3]
        self.assertListEqual(filtered_ds._spectrum_meta_ref['spec_id'].tolist(), expected_spec_ids)
        self.assertEqual(filtered_ds.peaks.n_all_peaks, 4)  # 2 peaks from each of the 2 spectra


if __name__ == "__main__":
    unittest.main()
