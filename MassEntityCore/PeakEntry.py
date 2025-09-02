from typing import Any, Dict

class PeakEntry:
    """
    Represents a single mass spectral peak with m/z and intensity.
    """

    def __init__(self, mz: float, intensity: float, extra: Dict[str, Any] = None):
        self.mz = mz
        self.intensity = intensity
        self.extra = extra if extra is not None else {}

    def __repr__(self):
        return f"PeakEntry({self.__str__()})"
    
    def __str__(self):
        if len(self.extra) == 0:
            return f"m/z={self.mz}, intensity={self.intensity}"
        else:
            return f"m/z={self.mz}, intensity={self.intensity}, extra={self.extra}"
    
    def __iter__(self):
        """
        Iterate over the m/z and intensity values.
        """
        yield self.mz
        yield self.intensity

    @property
    def neutral_loss(self) -> float:
        """
        Calculate the neutral loss for the peak entry.
        """
        assert "PrecursorMZ" in self.extra, "Precursor m/z not available"
        return self.extra["PrecursorMZ"] - self.mz