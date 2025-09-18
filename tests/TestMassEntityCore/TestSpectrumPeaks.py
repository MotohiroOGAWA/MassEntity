import unittest
import torch
import pandas as pd

from MassEntity.MassEntityCore.PeakSeries import PeakSeries, SpectrumPeaks
from MassEntity.MassEntityCore.PeakEntry import PeakEntry


class TestSpectrumPeaks(unittest.TestCase):

    def setUp(self):
        # Peak data: 3 spectra
        # Spectrum 1: 3 peaks, Spectrum 2: 3 peaks, Spectrum 3: 3 peaks
        self.data = torch.tensor([
            [100.0, 10.0],
            [101.0, 20.0],
            [102.0, 15.0],

            [200.0, 50.0],
            [201.0, 25.0],
            [202.0, 30.0],

            [300.0, 5.0],
            [301.0, 15.0],
            [302.0, 10.0],
        ], dtype=torch.float32)

        self.offsets = torch.tensor([0, 3, 6, 9], dtype=torch.int64)

        self.metadata = pd.DataFrame({
            "peak_id": list(range(1, 10)),
            "note": [f"p{i}" for i in range(1, 10)]
        })

        self.peak_series = PeakSeries(self.data, self.offsets, self.metadata)

        # Spectrum views
        self.spec1 = SpectrumPeaks(self.peak_series, 0)
        self.spec2 = SpectrumPeaks(self.peak_series, 1)
        self.spec3 = SpectrumPeaks(self.peak_series, 2)

    def test_len(self):
        self.assertEqual(len(self.spec1), 3)
        self.assertEqual(len(self.spec2), 3)
        self.assertEqual(len(self.spec3), 3)

    def test_getitem(self):
        p0 = self.spec1[0]
        self.assertIsInstance(p0, PeakEntry)
        self.assertAlmostEqual(p0._mz, 100.0)
        self.assertAlmostEqual(p0._intensity, 10.0)
        self.assertEqual(p0.metadata["note"], "p1")

        # Last peak of spectrum 2
        p_last = self.spec2[2]
        self.assertAlmostEqual(p_last._mz, 202.0)
        self.assertAlmostEqual(p_last._intensity, 30.0)

        # Out of range
        with self.assertRaises(IndexError):
            _ = self.spec1[5]

    def test_setitem_scalar(self):
        # Assign scalar value to metadata column
        self.spec1["new_col"] = "test_value"
        meta = self.spec1.metadata
        self.assertIn("new_col", meta.columns)
        self.assertTrue(all(meta["new_col"] == "test_value"))
        self.assertEqual(self.metadata["new_col"].isna().sum(), 6)  # other spectra remain NaN

    def test_setitem_sequence(self):
        # Assign sequence value to existing metadata column
        values = ["a", "b", "c"]
        self.spec2["note"] = values
        meta = self.spec2.metadata
        self.assertListEqual(meta["note"].tolist(), values)
        self.assertListEqual(self.metadata["note"].tolist(), ["p1", "p2", "p3", "a", "b", "c", "p7", "p8", "p9"])

    def test_setitem_sequence_length_mismatch(self):
        # Length mismatch should raise ValueError
        with self.assertRaises(ValueError):
            self.spec3["note"] = ["x", "y"]  # only 2 values, but spectrum has 3 peaks

    def test_iter(self):
        peaks = list(self.spec3)
        self.assertEqual(len(peaks), 3)
        self.assertTrue(all(isinstance(p, PeakEntry) for p in peaks))
        self.assertEqual(peaks[1]._mz, 301.0)

    def test_repr(self):
        r = repr(self.spec1)
        self.assertIn("SpectrumPeaks", r)
        self.assertIn("n_peaks=3", r)

    def test_metadata_property(self):
        meta2 = self.spec2.metadata
        self.assertEqual(len(meta2), 3)
        self.assertListEqual(meta2["peak_id"].tolist(), [4, 5, 6])

    def test_data_property(self):
        d3 = self.spec3.data
        self.assertEqual(d3.shape, (3, 2))
        self.assertTrue(torch.allclose(d3[:, 0], torch.tensor([300.0, 301.0, 302.0])))

    def test_normalize_copy(self):
        # in_place=False should not change the original
        spec2_copy = self.spec2.normalize(scale=1.0, in_place=False)
        self.assertIsInstance(spec2_copy, SpectrumPeaks)

        # normalized copy has max intensity = 1.0
        self.assertAlmostEqual(spec2_copy.data[:, 1].max().item(), 1.0)

        # original remains unchanged
        self.assertAlmostEqual(self.spec2.data[:, 1].max().item(), 50.0)

    def test_normalize_inplace(self):
        # in_place=True should modify original
        self.spec3.normalize(scale=7.0, in_place=True)
        self.assertAlmostEqual(self.spec3.data[:, 1].max().item(), 7.0)


if __name__ == "__main__":
    unittest.main()
