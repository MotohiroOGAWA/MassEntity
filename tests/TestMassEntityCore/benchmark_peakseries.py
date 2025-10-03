import time
import torch
import pandas as pd
from msentity.core.PeakSeries import PeakSeries


def make_peakseries(n_spectra: int, peaks_per_spectrum: int, device: str = "cpu") -> PeakSeries:
    """
    Generate a synthetic PeakSeries with random data on the given device.
    """
    total_peaks = n_spectra * peaks_per_spectrum

    # Random m/z and intensity
    mz = torch.rand(total_peaks, device=device) * 950 + 50   # [50, 1000)
    intensity = torch.rand(total_peaks, device=device) * 1e5 + 1

    data = torch.stack([mz, intensity], dim=1)

    # Offsets
    offsets = torch.arange(0, total_peaks + 1, peaks_per_spectrum, dtype=torch.int64, device=device)

    # Metadata (simple IDs)
    metadata = pd.DataFrame({"id": list(range(total_peaks))})

    return PeakSeries(data, offsets, metadata, is_sorted=False)


def benchmark(n_spectra: int, peaks_per_spectrum: int, device: str = "cpu"):
    """
    Benchmark normalize with 'vectorized' and 'for' methods.
    If device='cuda' and available, also measure GPU speed.
    """
    ps = make_peakseries(n_spectra, peaks_per_spectrum, device=device)

    # --- Vectorized normalize ---
    start = time.perf_counter()
    ps.normalize(scale=1.0, method="vectorized", in_place=False)
    torch.cuda.synchronize() if device == "cuda" else None
    t_vec = time.perf_counter() - start

    # --- For-loop normalize ---
    start = time.perf_counter()
    ps.normalize(scale=1.0, method="for", in_place=False)
    torch.cuda.synchronize() if device == "cuda" else None
    t_loop = time.perf_counter() - start

    print(f"[{device.upper()}] {n_spectra} spectra × {peaks_per_spectrum} peaks")
    print(f"  Vectorized normalize: {t_vec:.6f} sec")
    print(f"  For-loop normalize:   {t_loop:.6f} sec\n")


# python -m MassEntity.tests.TestMassEntityCore.benchmark_peakseries
if __name__ == "__main__":
    # CPU benchmark
    benchmark(1000, 100)
    benchmark(100000, 100)

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        benchmark(1000, 100, device="cuda")
        benchmark(100000, 100, device="cuda")
