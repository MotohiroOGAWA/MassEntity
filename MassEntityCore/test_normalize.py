import numpy as np
import pandas as pd
import time
from .PeakSeries import PeakSeries  # 実際のプロジェクトに合わせて import を修正

def make_peakseries(n_spectra: int, peaks_per_spectrum: int) -> PeakSeries:
    """
    Generate a synthetic PeakSeries with random data.
    """
    total_peaks = n_spectra * peaks_per_spectrum
    mz = np.random.uniform(50, 1000, size=total_peaks)
    intensity = np.random.uniform(1, 1e5, size=total_peaks)
    data = np.column_stack([mz, intensity])

    offsets = np.arange(0, total_peaks + 1, peaks_per_spectrum, dtype=np.int64)

    metadata = pd.DataFrame({
        "formula": [f"X{i}" for i in range(total_peaks)]
    })

    return PeakSeries(data, offsets, metadata, is_sorted=False)

def benchmark(n_spectra: int, peaks_per_spectrum: int):
    ps = make_peakseries(n_spectra, peaks_per_spectrum)

    # Measure normalize (vectorized)
    start = time.perf_counter()
    ps.normalize(scale=1.0, in_place=False)
    t_vec = time.perf_counter() - start

    # Measure normalize_by_for (loop)
    start = time.perf_counter()
    ps.normalize_by_for(scale=1.0, in_place=False)
    t_loop = time.perf_counter() - start

    print(f"{n_spectra} spectra × {peaks_per_spectrum} peaks")
    print(f"  Vectorized normalize:     {t_vec:.6f} sec")
    print(f"  Loop normalize_by_for:    {t_loop:.6f} sec")
    print()

if __name__ == "__main__":
    # Try different dataset sizes
    benchmark(100, 100)     # 1e4 peaks
    benchmark(1000, 100)    # 1e5 peaks
    benchmark(10000000, 100)    # 1e8 peaks
