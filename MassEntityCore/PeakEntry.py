from typing import Any, Dict

class PeakEntry:
    """
    Represents a single mass spectral peak with m/z and intensity.
    """

    def __init__(self, mz: float, intensity: float, metadata: Dict[str, Any] = None):
        self._mz = mz
        self._intensity = intensity
        self._metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return f"PeakEntry({self.__str__()})"
    
    def __str__(self):
        if len(self._metadata) == 0:
            return f"m/z={self._mz}, intensity={self._intensity}"
        else:
            return f"m/z={self._mz}, intensity={self._intensity}, extra={self._metadata}"
    
    def __iter__(self):
        """
        Iterate over the m/z and intensity values.
        """
        yield self._mz
        yield self._intensity

    @property
    def mz(self) -> float:
        """Return the m/z value of the peak."""
        return self._mz
    
    @property
    def intensity(self) -> float:
        """Return the intensity value of the peak."""
        return self._intensity
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Return the metadata dictionary of the peak."""
        return self._metadata
    

    @property
    def neutral_loss(self) -> float:
        """
        Calculate the neutral loss for the peak entry.
        """
        assert "PrecursorMZ" in self._metadata, "Precursor m/z not available"
        return self._metadata["PrecursorMZ"] - self._mz