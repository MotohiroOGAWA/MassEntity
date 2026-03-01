import unittest
import os
import torch
import numpy as np

from msentity.core.MSDataset import MSDataset

from msentity.utils.spectrum_similarity import *


class TestCosineSimilarityMatrixPaired(unittest.TestCase):
    """
    Tests for:
      cosine_similarity_matrix(ds1, index1, ds2, index2, ...)
    which returns paired cosine similarities (1D tensor [K]).
    """

    def setUp(self):
        self.test_file = os.path.join("tests", "dummy_files", "test_dummy.hdf5")

    def _load_ds_or_skip(self) -> MSDataset:
        if not os.path.isfile(self.test_file):
            self.skipTest(f"Test data file not found: {self.test_file}")
        ds = MSDataset.from_hdf5(self.test_file, load_peak_meta=False)
        return ds

    def _dataset_len(self, ds: MSDataset) -> int:
        # Try common conventions
        if hasattr(ds, "n_rows"):
            return int(ds.n_rows)
        if hasattr(ds, "__len__"):
            return int(len(ds))  # type: ignore[arg-type]
        # fallback
        if hasattr(ds, "_spectrum_meta_ref"):
            return int(ds._spectrum_meta_ref.shape[0])
        raise RuntimeError("Cannot determine dataset length (number of spectra).")

    def _find_nonzero_indices(self, ds: MSDataset, max_k: int = 16) -> torch.Tensor:
        """
        Pick up to max_k spectra that have at least one peak with intensity > 0.
        This makes the 'self similarity == 1' test robust.
        """
        n = self._dataset_len(ds)
        keep = []
        # Scan sequentially; max_k is small so this is fine
        for i in range(n):
            sp = ds.peaks[i]  # SpectrumPeaks
            if sp.intensity.numel() == 0:
                continue
            if float(sp.intensity.max().item()) > 0.0:
                keep.append(i)
                if len(keep) >= max_k:
                    break
        if not keep:
            self.skipTest("No non-zero spectra found in test_dummy.hdf5")
        return torch.tensor(keep, dtype=torch.int64)

    def test_returns_1d_and_correct_length(self):
        ds = self._load_ds_or_skip()
        idx = self._find_nonzero_indices(ds, max_k=8)

        sim = cosine_similarity_matrix(
            ds, idx,
            ds, idx,
            bin_width=0.01,
            intensity_exponent=1.0,
            device=None,
        )

        self.assertIsInstance(sim, torch.Tensor)
        self.assertEqual(sim.ndim, 1)
        self.assertEqual(sim.numel(), idx.numel())

    def test_self_similarity_is_one_for_nonzero_spectra(self):
        ds = self._load_ds_or_skip()
        idx = self._find_nonzero_indices(ds, max_k=8)

        sim = cosine_similarity_matrix(
            ds, idx,
            ds, idx,
            bin_width=0.01,
            intensity_exponent=1.0,
            device=None,
        )

        # identical spectrum paired with itself should give ~1
        sim_np = sim.detach().cpu().numpy()
        np.testing.assert_allclose(sim_np, np.ones_like(sim_np), atol=1e-6, rtol=0)

    def test_output_is_in_valid_range(self):
        ds = self._load_ds_or_skip()
        idx = self._find_nonzero_indices(ds, max_k=8)

        sim = cosine_similarity_matrix(
            ds, idx,
            ds, idx,
            bin_width=0.01,
            intensity_exponent=0.5,  # sqrt transform path
            device=None,
        )

        self.assertTrue(torch.all(sim >= -1.0).item())
        self.assertTrue(torch.all(sim <= 1.0).item())

    def test_shuffled_pairs_not_all_one(self):
        ds = self._load_ds_or_skip()
        idx = self._find_nonzero_indices(ds, max_k=8)

        if idx.numel() < 2:
            self.skipTest("Need at least 2 spectra to test shuffled pairing.")

        idx2 = idx[torch.randperm(idx.numel())]

        sim = cosine_similarity_matrix(
            ds, idx,
            ds, idx2,
            bin_width=0.01,
            intensity_exponent=1.0,
            device=None,
        )

        # We expect at least one pair to differ from 1.0 in typical datasets.
        # If the dataset contains many identical spectra, this could be all ones;
        # so we accept either outcome, but assert the computation ran and is finite.
        self.assertTrue(torch.isfinite(sim).all().item())

        # "Not all ones" is a useful sanity check but not guaranteed.
        # We'll assert it only if there exists at least one index mismatch.
        if not torch.equal(idx, idx2):
            all_ones = torch.allclose(sim, torch.ones_like(sim), atol=1e-6)
            self.assertFalse(all_ones, "All similarities were 1.0 even after shuffling pairs (possible duplicates).")

    def test_empty_indices(self):
        ds = self._load_ds_or_skip()
        idx = torch.empty((0,), dtype=torch.int64)

        sim = cosine_similarity_matrix(
            ds, idx,
            ds, idx,
            bin_width=0.01,
            intensity_exponent=1.0,
            device=None,
        )

        self.assertEqual(sim.ndim, 1)
        self.assertEqual(sim.numel(), 0)



class TestCosineSimilarityAllPairsMatrix(unittest.TestCase):
    """
    Tests for:
      cosine_similarity_all_pairs_matrix(ds1, ds2, ...)
    which returns a dense similarity matrix of shape [N1, N2].
    """

    def setUp(self):
        self.test_file = os.path.join("tests", "dummy_files", "test_dummy.hdf5")

    def _load_ds_or_skip(self) -> MSDataset:
        if not os.path.isfile(self.test_file):
            self.skipTest(f"Test data file not found: {self.test_file}")
        return MSDataset.from_hdf5(self.test_file, load_peak_meta=False)

    def _dataset_len(self, ds: MSDataset) -> int:
        if hasattr(ds, "n_rows"):
            return int(ds.n_rows)
        return int(len(ds))

    def test_shape(self):
        ds = self._load_ds_or_skip()

        n = self._dataset_len(ds)
        if n == 0:
            self.skipTest("Dataset is empty.")

        # Use ds vs ds to get [N, N]
        S = cosine_similarity_all_pairs_matrix(
            ds, ds,
            bin_width=0.01,
            intensity_exponent=1.0,
            device=None,
            max_pairs_per_call=200_000,  # force chunking on medium datasets
        )

        self.assertIsInstance(S, torch.Tensor)
        self.assertEqual(S.ndim, 2)
        self.assertEqual(tuple(S.shape), (n, n))

    def test_diagonal_is_one_for_nonzero_spectra(self):
        ds = self._load_ds_or_skip()
        n = self._dataset_len(ds)
        if n == 0:
            self.skipTest("Dataset is empty.")

        S = cosine_similarity_all_pairs_matrix(
            ds, ds,
            bin_width=0.01,
            intensity_exponent=1.0,
            device=None,
            max_pairs_per_call=200_000,
        )

        # Some spectra may be all-zero (or empty), in which case cosine is defined as 0 in our implementation.
        # We only assert diag==1 for spectra with non-zero L2 norm.
        norms = []
        for i in range(n):
            sp = ds.peaks[i]
            it = sp.intensity
            norms.append(float((it.float() * it.float()).sum().item()))
        norms = np.array(norms, dtype=np.float64)

        diag = torch.diagonal(S, 0).detach().cpu().numpy()

        # For non-zero spectra: diag ~ 1
        nonzero = norms > 0.0
        if not np.any(nonzero):
            self.skipTest("All spectra have zero norm; cannot test diag==1.")
        np.testing.assert_allclose(diag[nonzero], np.ones_like(diag[nonzero]), atol=1e-6, rtol=0)

        # For zero spectra: diag should be 0 (by definition in our cosine impl)
        zero = ~nonzero
        if np.any(zero):
            np.testing.assert_allclose(diag[zero], np.zeros_like(diag[zero]), atol=1e-6, rtol=0)

    def test_symmetry_when_same_dataset(self):
        ds = self._load_ds_or_skip()
        n = self._dataset_len(ds)
        if n == 0:
            self.skipTest("Dataset is empty.")

        S = cosine_similarity_all_pairs_matrix(
            ds, ds,
            bin_width=0.01,
            intensity_exponent=1.0,
            device=None,
            max_pairs_per_call=200_000,
        )

        self.assertTrue(torch.allclose(S, S.T, atol=1e-6))

    def test_matches_paired_function_for_subset(self):
        ds = self._load_ds_or_skip()
        n = self._dataset_len(ds)
        if n < 2:
            self.skipTest("Need at least 2 spectra to run subset check.")

        # Pick a small subset to compare (avoid huge compute)
        m = min(8, n)
        idx = torch.arange(m, dtype=torch.int64)

        # Full matrix on subset
        ds_sub = ds[idx]
        S = cosine_similarity_all_pairs_matrix(
            ds_sub, ds_sub,
            bin_width=0.01,
            intensity_exponent=1.0,
            device=None,
            max_pairs_per_call=10_000,  # force chunking even for small data
        )

        # Compare a few entries against the paired API
        # (i,j) -> paired indices arrays
        pairs = [(0, 0), (0, 1), (1, 0), (m - 1, m - 1)]
        for i, j in pairs:
            i_t = torch.tensor([i], dtype=torch.int64)
            j_t = torch.tensor([j], dtype=torch.int64)

            s_ij = cosine_similarity_matrix(
                ds_sub, i_t,
                ds_sub, j_t,
                bin_width=0.01,
                intensity_exponent=1.0,
                device=None,
            )[0].item()

            self.assertAlmostEqual(float(S[i, j].item()), float(s_ij), places=6)

    def test_chunked_equals_single_call_when_possible(self):
        ds = self._load_ds_or_skip()
        n = self._dataset_len(ds)
        if n == 0:
            self.skipTest("Dataset is empty.")

        # Use only a small prefix so that a single-call computation is feasible
        m = min(10, n)
        ds_sub = ds[torch.arange(m, dtype=torch.int64)]

        # Compute with a high threshold (single call path)
        S_single = cosine_similarity_all_pairs_matrix(
            ds_sub, ds_sub,
            bin_width=0.01,
            intensity_exponent=1.0,
            device=None,
            max_pairs_per_call=10_000_000,
        )

        # Compute with a low threshold (force chunk path)
        S_chunk = cosine_similarity_all_pairs_matrix(
            ds_sub, ds_sub,
            bin_width=0.01,
            intensity_exponent=1.0,
            device=None,
            max_pairs_per_call=20,  # force many chunks
        )

        self.assertTrue(torch.allclose(S_single, S_chunk, atol=1e-6))


if __name__ == "__main__":
    unittest.main()